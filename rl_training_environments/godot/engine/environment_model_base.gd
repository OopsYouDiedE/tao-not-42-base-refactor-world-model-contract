class_name EnvironmentModelBase
extends Node
##
## 所有采集环境的【基础模型类】。
##
## 约定：每个环境 = 一个继承本类的 .gd + 一个 .tscn。
## 本类封装【所有模型通用】的强化学习接口，子类只需填任务逻辑：
##   - 通用动作空间：10 个连续变量 + 30 个离散变量（各环境按需使用其中几个）。
##   - 通用相机：固定分辨率 CAM_W×CAM_H；俯仰夹紧 ±60°，偏航无限旋转。
##   - 通用观测附带消息：sim_dt = 本次渲染推进的“物理步数 × 步长”（务必传给模型）。
##
## main(C#) 与本类的契约（批量模式）：
##   set_action(cont, disc) -> physics_step(dt)×N -> get_obs_image() / get_info()
## 独立运行（直接把本场景当主场景跑）时，本类用键盘自驱动以便人工验证。
##

# === 通用动作接口（所有环境共享，固定不变）===
const CONT_DIM := 10   # 连续变量槽位
const DISC_DIM := 30   # 离散变量槽位

# === 通用相机（所有环境共享同一分辨率）===
const CAM_W := 128
const CAM_H := 128
const PITCH_LIMIT := deg_to_rad(60.0)   # 俯仰 ±60°，超出夹紧
const DEFAULT_PHYSICS_DT := 1.0 / 240.0

# 角运动手感（可被子类/检视器覆盖）
@export var max_ang_vel := deg_to_rad(90.0)   # 角速度上限：90°/秒 ≈ 1.5708 rad/s
@export var ang_damping := 2.5          # 角速度阻尼 (1/s)，便于稳定瞄准

# 动作缓冲（main 每回合写入；独立模式由键盘写入）
var _cont := PackedFloat32Array()
var _disc := PackedInt32Array()

# 相机角状态：偏航(yaw, 绕Y, 无限) / 俯仰(pitch, 绕X, ±60)
var _yaw := 0.0
var _pitch := 0.0
var _yaw_vel := 0.0
var _pitch_vel := 0.0

# 时间记账
var physics_dt := DEFAULT_PHYSICS_DT
var _sim_elapsed := 0.0    # 本回合累计仿真时间
var _sim_dt_last := 0.0    # 最近一次渲染推进的 steps*dt —— 要随观测传给模型的消息
var _steps_last := 0       # 最近一次渲染推进的物理步数
var _reward := 0.0         # 本次渲染缓存的奖励（每渲染只算一次，避免有状态的奖励被多次调用污染）

# 子类在 _setup() 中必须设置这两个引用
var camera: Camera3D
var subviewport: SubViewport

var standalone := false    # 是否独立运行（直接跑本场景）


func _ready() -> void:
	_cont.resize(CONT_DIM)
	_cont.fill(0.0)
	_disc.resize(DISC_DIM)
	_disc.fill(0)

	# 直接运行本场景（被当作主场景）即独立模式；被 main 实例化进别处则不是。
	standalone = (get_tree().current_scene == self) or (get_parent() == get_tree().root)

	_setup()
	assert(camera != null and subviewport != null,
		"子类 _setup() 必须设置 camera 与 subviewport 引用")
	subviewport.size = Vector2i(CAM_W, CAM_H)
	subviewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS

	reset()

	# 批量模式由 main 显式调用 physics_step（不按真实时间）；独立模式才用实时 _physics_process。
	set_physics_process(standalone)


# ---- main(C#) 调用的接口 ----

## 写入本回合动作（连续 + 离散），main 每回合调一次。
func set_action(cont: PackedFloat32Array, disc: PackedInt32Array) -> void:
	for i in mini(cont.size(), CONT_DIM):
		_cont[i] = cont[i]
	for i in mini(disc.size(), DISC_DIM):
		_disc[i] = disc[i]


## 推进【一个】固定物理步（不按真实时间）。main 每回合按本帧步数调 N 次。
## update_camera：是否把姿态写进相机节点。批量多步推进时只有【最后一步】需要写（中间步不渲染、
##   奖励/done 每渲染只在末尾算一次），故 step_render 传 false 跳过每步的 transparent 节点写入，
##   24步/帧时省下 23×40 次相机 transform 更新。
## 【子类契约】_task_physics 不得依赖相机节点 transform 在多步渲染【中途】被更新——
##   需要朝向请直接用 _yaw/_pitch（其每步都更新，只是没写进相机节点）。
func physics_step(dt: float, update_camera := true) -> void:
	var acc := _angular_accel()         # Vector2(俯仰角加速度, 偏航角加速度)
	_pitch_vel = clampf(_pitch_vel + acc.x * dt, -max_ang_vel, max_ang_vel)
	_yaw_vel = clampf(_yaw_vel + acc.y * dt, -max_ang_vel, max_ang_vel)
	# 阻尼（指数衰减式），松手后平稳停下
	var k := clampf(ang_damping * dt, 0.0, 1.0)
	_pitch_vel -= _pitch_vel * k
	_yaw_vel -= _yaw_vel * k

	_pitch += _pitch_vel * dt
	_yaw += _yaw_vel * dt
	# 俯仰夹紧 ±60，撞上下界则【速度反转】反弹回来；偏航无墙、自由绕回 [-PI, PI]
	if _pitch > PITCH_LIMIT:
		_pitch = PITCH_LIMIT
		_pitch_vel = -_pitch_vel
	elif _pitch < -PITCH_LIMIT:
		_pitch = -PITCH_LIMIT
		_pitch_vel = -_pitch_vel
	_yaw = wrapf(_yaw, -PI, PI)
	if update_camera:
		camera.rotation = Vector3(_pitch, _yaw, 0.0)

	_sim_elapsed += dt
	_task_physics(dt)


## main 便捷入口：一次渲染推进 n_steps 个物理步，并记录要传出的 sim_dt = n_steps*dt。
func step_render(n_steps: int, dt: float) -> void:
	for _i in n_steps:
		physics_step(dt, false)       # 中间步不写相机节点（不渲染）
	camera.rotation = Vector3(_pitch, _yaw, 0.0)   # 仅末尾写一次：这一姿态用于渲染/奖励/done
	_steps_last = n_steps
	_sim_dt_last = n_steps * dt
	_reward = _compute_reward()       # 每渲染算一次（差分奖励在此更新内部状态）


## 帧截图（128×128 RGB）。main 读出后写入共享内存图像区。
func get_obs_image() -> Image:
	return subviewport.get_texture().get_image()


## 随观测一起传给模型的标量信息（含务必传输的 sim_dt = 物理步数×步长）。
func get_info() -> Dictionary:
	return {
		"reward": _reward,
		"done": _is_done(),
		"sim_dt": _sim_dt_last,      # ← “过了几个物理帧 × 步长”
		"steps": _steps_last,
	}


func get_reward() -> float: return _reward
func get_done() -> bool: return _is_done()
func get_sim_dt() -> float: return _sim_dt_last

## 给 main(C#) 用：一次取回 reward 与 done，省去两次跨语言调用。x=reward, y=done(0/1)。
func get_reward_done() -> Vector2:
	return Vector2(_reward, 1.0 if _is_done() else 0.0)


## 回合重置：清相机角状态 + 调子类任务重置。
func reset() -> void:
	_yaw = 0.0
	_pitch = 0.0
	_yaw_vel = 0.0
	_pitch_vel = 0.0
	_sim_elapsed = 0.0
	_sim_dt_last = 0.0
	_steps_last = 0
	_reward = 0.0
	if camera:
		camera.rotation = Vector3.ZERO
	_reset_task()


# ---- 子类要重写的虚函数（默认空实现）----

## 建场景：必须设置 camera、subviewport，并搭好任务世界。
func _setup() -> void: pass

## 由当前动作算出角加速度 Vector2(俯仰, 偏航)。子类决定用哪些动作槽位。
func _angular_accel() -> Vector2: return Vector2.ZERO

## 每个物理步的任务侧更新（移动目标、计时触发等）。
func _task_physics(_dt: float) -> void: pass

## 任务侧回合重置。
func _reset_task() -> void: pass

## 计算本步奖励。
func _compute_reward() -> float: return 0.0

## 是否回合结束（成功/失败）。
func _is_done() -> bool: return false

## 独立模式：把键盘读进 _cont/_disc（子类按自己的控制方式实现）。
func _standalone_input() -> void: pass

## 独立模式：每帧后的 HUD/调试刷新。
func _on_standalone_tick() -> void: pass


# ---- 独立运行：实时自驱动，便于人工验证（批量模式下禁用）----
func _physics_process(delta: float) -> void:
	_standalone_input()
	physics_step(delta)
	_steps_last = 1
	_sim_dt_last = delta
	_reward = _compute_reward()
	_on_standalone_tick()
