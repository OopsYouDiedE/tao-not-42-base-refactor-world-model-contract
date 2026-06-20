using Godot;
using System;
using System.IO.MemoryMappedFiles;
using System.Threading;
using System.Threading.Tasks;

// 训练编排器(Main)：把环境场景(env_spotlight_discrete.tscn)克隆 NumEnvs 份，
// 用事件握手把 40 个环境的观测(图像+元数据)推给 Python，并把 Python 的动作分发回各环境。
//
// 与 GDScript 环境(ModelBase 子类)的契约：
//   set_action(cont[10], disc[30]) -> step_render(steps, dt) -> get_obs(图像) / get_reward_done()
// 环境只管按动作更新自己；Main 负责加载、握手、回读纹理、写共享内存、分发动作。
//
// 共享内存布局(字节)：
//   [图像区]  NumEnvs * 128*128*3
//   [元数据]  NumEnvs * 5 * float32  = [frameCount, steps, sim_dt(=steps*dt), reward, done]
//   [连续动作] NumEnvs * 10 * float32 (Python 写)
//   [离散动作] NumEnvs * 30 * int32   (Python 写)
public partial class Main : Node
{
    private const string MapName = "GodotRL_SharedMem";
    private const string ObsReadyName = "GodotRL_ObsReady";
    private const string ActReadyName = "GodotRL_ActReady";
    private const string EnvScenePath = "res://env_spotlight_discrete.tscn";

    private const int NumEnvs = 40;
    private const int ImageWidth = 128, ImageHeight = 128, Channels = 3;
    private const int ImageSize = ImageWidth * ImageHeight * Channels;   // 49152
    private const int TotalImagesSize = NumEnvs * ImageSize;             // 1,966,080

    private const int MetaPerEnv = 5;                                   // frameCount, steps, sim_dt, reward, done
    private const int TotalMetaSize = NumEnvs * MetaPerEnv * 4;          // 800

    // 通用动作接口(与 ModelBase 一致)：10 连续 + 30 离散。
    private const int ContDim = 10, DiscDim = 30;
    private const int ContBytes = NumEnvs * ContDim * 4;                 // 1600
    private const int DiscBytes = NumEnvs * DiscDim * 4;                 // 4800

    private const int MetaOffset = TotalImagesSize;
    private const int ContOffset = TotalImagesSize + TotalMetaSize;
    private const int DiscOffset = ContOffset + ContBytes;
    // 异步模式用的 seqlock 序号(int32)：Godot 写 obs 前置奇、写完置偶；Python 读到奇或前后不等则重读，防撕裂帧。
    private const int SeqOffset = DiscOffset + DiscBytes;
    private const int TotalSize = SeqOffset + 4;

    private const int WaitActMs = 200;

    // 步进模式：Fixed=每帧固定步数；Decoupled=每帧随机[1,Max]步。两者都不按真实时间等。
    public enum StepMode { Fixed, Decoupled }
    [Export] public StepMode Mode = StepMode.Fixed;
    [Export] public int FixedStepsPerRender = 4;
    [Export] public int MaxStepsPerRender = 8;
    [Export] public float PhysicsHz = 240f;

    private MemoryMappedFile _mmf;
    private MemoryMappedViewAccessor _accessor;
    private EventWaitHandle _obsReady, _actReady;

    private Node[] _envs = new Node[NumEnvs];
    private SubViewport[] _viewports = new SubViewport[NumEnvs];
    private SubViewport _atlasViewport;   // 把 40 个环境视口纵向合成到一张大纹理，单次 GPU 回读(A)

    private float[] _meta = new float[NumEnvs * MetaPerEnv];
    private float[] _contAll = new float[NumEnvs * ContDim];
    private int[] _discAll = new int[NumEnvs * DiscDim];
    private float[] _contEnv = new float[ContDim];
    private int[] _discEnv = new int[DiscDim];
    private int[] _frameCounts = new int[NumEnvs];
    private float[] _rewards = new float[NumEnvs];
    private float[] _dones = new float[NumEnvs];

    private float _physicsDt = 1f / 240f;
    private int _stepsThisRender = 0;
    private bool _awaitingAction = false;
    // 流水线(固定1步延迟,重叠 Godot/Python)。默认【关】：Godot 单线程渲染+回读使重叠只能覆盖~5ms物理，
    // 实测仅 +6%，不值 1 步动作延迟。保留 RL_PIPELINE=1 供实验。
    private bool _pipeline = false;

    // 异步(自由跑)模式：Godot 完全不等 Python——每 tick 读 SHM 最新动作→步进→把 obs+累积reward 写回 SHM。
    // 两侧时钟解耦：Godot 跑满速、Python 自取最新帧，动作延迟可变(用户接受)。RL_ASYNC=1 开启。
    private bool _asyncMode = false;
    private float[] _cumReward = new float[NumEnvs];   // 每环境本回合累积奖励(Python 读后做差分)
    private float[] _cumSimDt = new float[NumEnvs];    // 每环境本回合累积 sim_dt
    private int _seq = 0;                              // seqlock 序号
    private RandomNumberGenerator _rng = new RandomNumberGenerator();
    private double _fpsTimer = 0;

    // 性能计时：分别累计「40× GPU 回读+取奖励」与「并行格式转换+写图像」的耗时。
    private readonly System.Diagnostics.Stopwatch _sw = new System.Diagnostics.Stopwatch();
    private double _readbackMsAccum = 0, _writeMsAccum = 0;
    private int _pubCount = 0;

    private static string EnvGet(string k) => System.Environment.GetEnvironmentVariable(k);

    public override void _Ready()
    {
        _rng.Randomize();

        var modeEnv = EnvGet("RL_STEP_MODE");
        if (!string.IsNullOrEmpty(modeEnv))
            Mode = modeEnv.ToLower().StartsWith("dec") ? StepMode.Decoupled : StepMode.Fixed;
        if (int.TryParse(EnvGet("RL_FIXED_STEPS"), out var fs)) FixedStepsPerRender = fs;
        if (int.TryParse(EnvGet("RL_MAX_STEPS"), out var ms)) MaxStepsPerRender = ms;
        if (float.TryParse(EnvGet("RL_PHYSICS_HZ"), out var hz)) PhysicsHz = hz;
        var pipeEnv = EnvGet("RL_PIPELINE");
        if (!string.IsNullOrEmpty(pipeEnv)) _pipeline = !(pipeEnv == "0" || pipeEnv.ToLower() == "false");
        var asyncEnv = EnvGet("RL_ASYNC");
        if (!string.IsNullOrEmpty(asyncEnv)) _asyncMode = !(asyncEnv == "0" || asyncEnv.ToLower() == "false");
        FixedStepsPerRender = Mathf.Max(1, FixedStepsPerRender);
        MaxStepsPerRender = Mathf.Max(1, MaxStepsPerRender);
        _physicsDt = 1f / Mathf.Max(1f, PhysicsHz);

        // 渲染端不限速：回合速率由 Python 握手节奏决定。
        DisplayServer.WindowSetVsyncMode(DisplayServer.VSyncMode.Disabled);
        Engine.MaxFps = 0;

        // 汇总视口：128 ×(NumEnvs*128)，把 40 个环境视口纵向堆叠合成 → 每回合只回读 1 次(A：消除 40 次串行 GPU 同步)。
        _atlasViewport = new SubViewport
        {
            Size = new Vector2I(ImageWidth, NumEnvs * ImageHeight),
            RenderTargetUpdateMode = SubViewport.UpdateMode.Always,
            RenderTargetClearMode = SubViewport.ClearMode.Always,
        };
        AddChild(_atlasViewport);

        // 克隆 NumEnvs 份环境场景；每份自带 SubViewport(独立 World3D，互不干扰)。
        // 环境实例挂到 atlas 之下 → 其 3D 视口先渲染、atlas 后合成，保证读到的是当前帧(无差 1 帧陈旧)。
        var packed = GD.Load<PackedScene>(EnvScenePath);
        for (int i = 0; i < NumEnvs; i++)
        {
            var env = packed.Instantiate();
            _atlasViewport.AddChild(env);   // 触发环境 _ready：standalone=false → 不自驱动，等 Main 调用
            _envs[i] = env;
            var vp = env.GetNode<SubViewport>("SubViewport");
            _viewports[i] = vp;
            // 独立模式才用的调试 CanvasLayer 会画进 atlas 污染合成 → 批量模式关掉。
            if (env.HasNode("CanvasLayer"))
                env.GetNode<CanvasLayer>("CanvasLayer").Visible = false;
            // 把该 env 的视口纹理贴到 atlas 第 i 个纵向格(y=i*128)。
            var spr = new Sprite2D { Texture = vp.GetTexture(), Centered = false, Position = new Vector2(0, i * ImageHeight) };
            _atlasViewport.AddChild(spr);
        }

        _mmf = MemoryMappedFile.CreateOrOpen(MapName, TotalSize);
        _accessor = _mmf.CreateViewAccessor(0, TotalSize);
        _obsReady = new EventWaitHandle(false, EventResetMode.AutoReset, ObsReadyName);
        _actReady = new EventWaitHandle(false, EventResetMode.AutoReset, ActReadyName);

        GD.Print($"[Main] 克隆 {NumEnvs} 个环境 '{EnvScenePath}'。共享内存 {TotalSize}B。" +
                 $" 模式={Mode} Fixed={FixedStepsPerRender} Max={MaxStepsPerRender} dt={_physicsDt:F4}。等待 Python action。");
    }

    public override void _Process(double delta)
    {
        if (_asyncMode)
        {
            ProcessAsync();
        }
        else if (_pipeline)
        {
            // 流水线(固定1步动作延迟)：发布观测后【不等】新动作，先用上一 tick 已就绪的动作推进物理，
            // 使 Godot 的回读/物理与 Python 计算重叠；随后才阻塞取 Python 对【刚发布观测】算出的动作存为 pending。
            // 1) 发布观测(回读上一 tick 物理产生、本帧已渲染的画面) + 置 ObsReady → Python 立刻开始算动作。
            PublishObservation();
            // 2) 用 pending 动作(上一 tick 读入；首帧为 0)推进物理。其渲染在本帧末，与 Python 计算重叠。
            ApplyPendingAndStep();
            // 3) 阻塞取 Python 的新动作 → 存为下一 tick 的 pending。
            if (_actReady.WaitOne(WaitActMs))
            {
                _accessor.ReadArray(ContOffset, _contAll, 0, NumEnvs * ContDim);
                _accessor.ReadArray(DiscOffset, _discAll, 0, NumEnvs * DiscDim);
            }
        }
        else
        {
            // 锁步(对照基线)：发布 → 等动作 → 步进，无重叠。
            if (!_awaitingAction)
            {
                PublishObservation();
                _awaitingAction = true;
            }
            if (_actReady.WaitOne(WaitActMs))
            {
                _awaitingAction = false;
                _accessor.ReadArray(ContOffset, _contAll, 0, NumEnvs * ContDim);
                _accessor.ReadArray(DiscOffset, _discAll, 0, NumEnvs * DiscDim);
                ApplyPendingAndStep();
            }
        }

        _fpsTimer += delta;
        if (_fpsTimer >= 1.0)
        {
            _fpsTimer = 0;
            double rb = _pubCount > 0 ? _readbackMsAccum / _pubCount : 0;
            double wr = _pubCount > 0 ? _writeMsAccum / _pubCount : 0;
            GD.Print($"[Main] 回合速率={Engine.GetFramesPerSecond():F1}/s 步数/帧={_stepsThisRender} " +
                     $"回读(atlas单次)={rb:F2}ms 写图={wr:F2}ms 累计帧={_frameCounts[0]}");
            _readbackMsAccum = 0; _writeMsAccum = 0; _pubCount = 0;
        }
    }

    // 用当前 pending 动作(_contAll/_discAll)推进每个环境 steps 个物理步并触发渲染。
    private void ApplyPendingAndStep()
    {
        int steps = (Mode == StepMode.Fixed)
            ? FixedStepsPerRender
            : _rng.RandiRange(1, MaxStepsPerRender);
        _stepsThisRender = steps;
        for (int i = 0; i < NumEnvs; i++)
        {
            Array.Copy(_contAll, i * ContDim, _contEnv, 0, ContDim);
            Array.Copy(_discAll, i * DiscDim, _discEnv, 0, DiscDim);
            _envs[i].Call("set_action", _contEnv, _discEnv);
            _envs[i].Call("step_render", steps, _physicsDt);
        }
    }

    // 异步(自由跑)一帧：读最新动作(不等待)→步进→发布累积观测。Godot 全程不阻塞。
    private void ProcessAsync()
    {
        _accessor.ReadArray(ContOffset, _contAll, 0, NumEnvs * ContDim);
        _accessor.ReadArray(DiscOffset, _discAll, 0, NumEnvs * DiscDim);
        ApplyPendingAndStep();
        PublishObservationAsync();
    }

    // 异步发布：奖励/sim_dt 按回合累积写出(Python 读后做差分)，用 seqlock 防撕裂帧。
    private void PublishObservationAsync()
    {
        for (int i = 0; i < NumEnvs; i++)
        {
            var rd = _envs[i].Call("get_reward_done").AsVector2();
            _cumReward[i] += rd.X;
            _cumSimDt[i] += _stepsThisRender * _physicsDt;
            _dones[i] = rd.Y;
            _frameCounts[i] += 1;
        }
        var atlasImg = _atlasViewport.GetTexture().GetImage();
        if (atlasImg.GetFormat() != Image.Format.Rgb8)
            atlasImg.Convert(Image.Format.Rgb8);
        var data = atlasImg.GetData();
        for (int i = 0; i < NumEnvs; i++)
        {
            int b = i * MetaPerEnv;
            _meta[b + 0] = _frameCounts[i];
            _meta[b + 1] = _stepsThisRender;
            _meta[b + 2] = _cumSimDt[i];     // 累积 sim_dt
            _meta[b + 3] = _cumReward[i];    // 累积 reward
            _meta[b + 4] = _dones[i];
        }
        // seqlock：写前置奇、写后置偶；Python 读到奇或前后不等则重读。
        _seq++; _accessor.Write(SeqOffset, _seq);
        System.Threading.Thread.MemoryBarrier();
        _accessor.WriteArray(0, data, 0, TotalImagesSize);
        _accessor.WriteArray(MetaOffset, _meta, 0, NumEnvs * MetaPerEnv);
        System.Threading.Thread.MemoryBarrier();
        _seq++; _accessor.Write(SeqOffset, _seq);

        // 结束的环境：重置并清累积(下一帧起从 0 累积新回合)。
        for (int i = 0; i < NumEnvs; i++)
            if (_dones[i] > 0.5f)
            {
                _envs[i].Call("reset");
                _cumReward[i] = 0f;
                _cumSimDt[i] = 0f;
            }
        _obsReady.Set();
    }

    private void PublishObservation()
    {
        // 1) 取 reward/done(廉价的 GDScript 调用，不碰 GPU)。
        for (int i = 0; i < NumEnvs; i++)
        {
            var rd = _envs[i].Call("get_reward_done").AsVector2();
            _rewards[i] = rd.X;
            _dones[i] = rd.Y;
            _frameCounts[i] += 1;
        }
        // 2) 单次 GPU 回读：整张 atlas(已合成 40 个环境，纵向堆叠 → 字节布局即 env-major 连续)。
        _sw.Restart();
        var atlasImg = _atlasViewport.GetTexture().GetImage();
        _readbackMsAccum += _sw.Elapsed.TotalMilliseconds;
        // 3) 单次格式转换 + 单次整块写入(取代旧的 40 次回读 + Parallel.For)。
        _sw.Restart();
        if (atlasImg.GetFormat() != Image.Format.Rgb8)
            atlasImg.Convert(Image.Format.Rgb8);
        var data = atlasImg.GetData();
        _accessor.WriteArray(0, data, 0, TotalImagesSize);
        _writeMsAccum += _sw.Elapsed.TotalMilliseconds;
        _pubCount++;
        // 3) 元数据：[frameCount, steps, sim_dt(=steps*dt), reward, done]。
        for (int i = 0; i < NumEnvs; i++)
        {
            int b = i * MetaPerEnv;
            _meta[b + 0] = _frameCounts[i];
            _meta[b + 1] = _stepsThisRender;
            _meta[b + 2] = _stepsThisRender * _physicsDt;   // ← 务必传出的“物理步数×步长”
            _meta[b + 3] = _rewards[i];
            _meta[b + 4] = _dones[i];
        }
        _accessor.WriteArray(MetaOffset, _meta, 0, NumEnvs * MetaPerEnv);

        // 4) 自动重置已结束(命中)的环境：终止观测已发布，下回合从新回合开始。
        for (int i = 0; i < NumEnvs; i++)
            if (_dones[i] > 0.5f)
                _envs[i].Call("reset");

        _obsReady.Set();
    }

    public override void _ExitTree()
    {
        _accessor?.Dispose();
        _mmf?.Dispose();
        _obsReady?.Dispose();
        _actReady?.Dispose();
        GD.Print("[Main] 共享内存与事件已释放。");
    }
}
