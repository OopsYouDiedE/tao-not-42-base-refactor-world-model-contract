extends ModelBase
##
## 第一个采集环境：聚光灯瞄准任务 ——【离散控制版】。
## （用路径继承基础类，避免依赖编辑器才会生成的 class_name 全局缓存；等价于 extends ModelBase。）
##
## 任务：模型固定在房间正中央，只能转动视角。回合开始后的随机 0~2 秒（仿真时间）某一刻，
## 聚光灯突然照亮场景中的某个物体；模型要【尽快】把相机转到对准该物体。
## 视角俯仰保持 ±60°（基础类已夹紧），左右无限旋转；被照物体保证落在 ±60° 可瞄范围内。
##
## 离散控制：占用 30 个离散变量里的【前 4 个】，各取 0/1，作为四个方向的“加速键”：
##   disc[0]=俯仰上  disc[1]=俯仰下  disc[2]=偏航左  disc[3]=偏航右
## 每个按下时给对应轴一个固定角加速度（控制的是加速度，不是角度）。
##
## 奖励在 Godot 内计算并随观测传出：每渲染 = 和上次瞄准角之差（靠近加分/远离扣分）。
## 差分缩放=1，故整局累计奖励 = 初始视角差 - 最终视角差 ≈【一开始相机视角与物体的视角差】。
##

# 本任务使用的离散动作槽位
const ACT_PITCH_UP := 0
const ACT_PITCH_DOWN := 1
const ACT_YAW_LEFT := 2
const ACT_YAW_RIGHT := 3

const N_TARGETS := 6
const TARGET_RADIUS := 6.0
const PITCH_SAFE_DEG := 50.0          # 目标俯仰落在 ±50°，确保在 ±60° 内可被瞄到
const ACCEL := 5.0                    # 离散按键给的角加速度 (rad/s^2)
const HIT_THRESHOLD := deg_to_rad(3.0)
const TRIGGER_MIN := 0.1               # 聚光灯触发时间下限（仿真秒）—— 物体很快出现
const TRIGGER_MAX := 0.4               # 上限：0.1~0.4s 内照亮某物体，从此刻开始计奖励
const MAX_AIM_SIM := 10.0              # 亮起后最多给 10 仿真秒，超时算放弃→回合结束（保证回合有界）
const REWARD_SCALE := 1.0             # 差分缩放=1（弧度）：整局累计奖励 = 初始视角差 - 最终视角差
const ERR_PENALTY := 0.5             # 稠密项：每仿真秒按当前误差扣分，打破"不动=0"的局部最优

# tscn 中的固定节点
@onready var _cam: Camera3D = $SubViewport/Camera3D
@onready var _vp: SubViewport = $SubViewport
@onready var _debug_view: TextureRect = $CanvasLayer/DebugView
@onready var _hud: Label = $CanvasLayer/Hud

var _rng := RandomNumberGenerator.new()
var _targets: Array[MeshInstance3D] = []
var _spotlight: SpotLight3D
var _target_idx := -1
var _trigger_time := 0.0
var _lit := false
var _lit_elapsed := 0.0               # 聚光灯亮起后经过的仿真时间（用于统计反应速度）
var _prev_aim_error := PI             # 上一渲染的瞄准角误差（差分奖励用）
var _initial_aim_error := 0.0         # 亮起瞬间的视角差 = 整局可得的最大累计奖励
var _accum_reward := 0.0              # 本回合累计奖励（HUD 验证用，应收敛到初始视角差）


func _setup() -> void:
	# 每个环境唯一种子：40 份在同一帧实例化，若都用 randomize() 会撞到同一个种子，
	# 导致 40 个环境的随机布局/颜色/目标完全相同（画面一模一样）。用唯一的 instance_id 混入。
	_rng.seed = get_instance_id() ^ Time.get_ticks_usec()
	camera = _cam
	subviewport = _vp
	_build_world()

	# 独立模式才显示模型视角与 HUD；批量模式由 main 直接读 SubViewport 纹理。
	_debug_view.visible = standalone
	_hud.visible = standalone
	if standalone:
		_debug_view.texture = _vp.get_texture()   # 把 128 的视口放大铺满窗口（obs 仍是 128）


func _build_world() -> void:
	# 世界环境：压低环境光，让“被聚光灯照亮的物体”成为唯一显著线索。
	var we := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.02, 0.02, 0.03)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.55, 0.55, 0.6)
	# 环境光抬到能看清【墙色+棋盘】以便模型定位自身朝向，但仍远低于聚光灯——被照物体依旧最亮。
	env.ambient_light_energy = 0.8
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	we.environment = env
	_vp.add_child(we)

	_build_room()

	# 聚光灯：初始关闭，触发时移到目标外侧朝向目标。
	_spotlight = SpotLight3D.new()
	_spotlight.light_energy = 10.0
	_spotlight.spot_range = 24.0
	_spotlight.spot_angle = 16.0
	_spotlight.spot_attenuation = 0.5
	_spotlight.visible = false
	_vp.add_child(_spotlight)

	# 候选物体（每回合重随机位置，其中一个会被照亮）。
	for _i in N_TARGETS:
		var m := MeshInstance3D.new()
		var s := _rng.randf_range(0.5, 0.95)
		if _rng.randf() < 0.5:
			var box_mesh := BoxMesh.new()
			box_mesh.size = Vector3(s, s, s)
			m.mesh = box_mesh
		else:
			var sphere_mesh := SphereMesh.new()
			sphere_mesh.radius = s * 0.5
			sphere_mesh.height = s
			m.mesh = sphere_mesh
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(_rng.randf(), _rng.randf(), _rng.randf())
		mat.roughness = _rng.randf_range(0.3, 0.9)
		m.material_override = mat
		_vp.add_child(m)
		_targets.append(m)


func _build_room() -> void:
	# 6 面盒子组成的房间，相机(模型)固定在原点中央。
	# 每面墙不同主色 + 棋盘纹理：给模型【稳定的朝向线索】（纯色墙无法判断偏航转到了哪里）。
	var half := 9.0
	var checker := _make_checker_texture()
	var defs := [
		[Vector3(0, -half, 0), Vector3(2 * half, 0.2, 2 * half), Color(0.30, 0.30, 0.32)],  # 地面 灰
		[Vector3(0, half, 0), Vector3(2 * half, 0.2, 2 * half), Color(0.20, 0.20, 0.24)],   # 天花板 暗灰
		[Vector3(-half, 0, 0), Vector3(0.2, 2 * half, 2 * half), Color(0.55, 0.22, 0.22)],  # -X 红
		[Vector3(half, 0, 0), Vector3(0.2, 2 * half, 2 * half), Color(0.22, 0.50, 0.25)],   # +X 绿
		[Vector3(0, 0, -half), Vector3(2 * half, 2 * half, 0.2), Color(0.25, 0.30, 0.58)],  # -Z 蓝
		[Vector3(0, 0, half), Vector3(2 * half, 2 * half, 0.2), Color(0.58, 0.52, 0.22)],   # +Z 黄
	]
	for d in defs:
		var w := MeshInstance3D.new()
		var box := BoxMesh.new()
		box.size = d[1]
		w.mesh = box
		w.position = d[0]
		var mat := StandardMaterial3D.new()
		mat.albedo_color = d[2]
		mat.albedo_texture = checker
		mat.uv1_scale = Vector3(4, 4, 4)   # 棋盘重复，给连续偏航以可分辨的竖直特征
		mat.roughness = 0.95
		w.material_override = mat
		_vp.add_child(w)


# 程序生成一张棋盘纹理：让墙面有可分辨的格子特征（否则纯色墙无法定位偏航）。
func _make_checker_texture() -> ImageTexture:
	var n := 32
	var img := Image.create(n, n, false, Image.FORMAT_RGB8)
	for y in n:
		for x in n:
			var c := 1.0 if ((x / 8 + y / 8) % 2 == 0) else 0.62
			img.set_pixel(x, y, Color(c, c, c))
	return ImageTexture.create_from_image(img)


# 给定 (yaw, pitch) 的相机前向方向（与基础类相机朝向公式一致）：
#   forward = (-sin yaw·cos pitch, sin pitch, -cos yaw·cos pitch)
# 把目标放在该方向上，相机偏航/俯仰等于该 (yaw,pitch) 时即正对目标。
func _dir_from(yaw: float, pitch: float) -> Vector3:
	return Vector3(-sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch))


func _reposition_targets() -> void:
	for m in _targets:
		var yaw := _rng.randf_range(-PI, PI)
		var pitch := deg_to_rad(_rng.randf_range(-PITCH_SAFE_DEG, PITCH_SAFE_DEG))
		m.position = _dir_from(yaw, pitch) * TARGET_RADIUS


func _reset_task() -> void:
	_lit = false
	_lit_elapsed = 0.0
	_target_idx = -1
	_prev_aim_error = PI
	_initial_aim_error = 0.0
	_accum_reward = 0.0
	if _spotlight:
		_spotlight.visible = false
	_trigger_time = _rng.randf_range(TRIGGER_MIN, TRIGGER_MAX)
	_reposition_targets()


func _light_target() -> void:
	_target_idx = _rng.randi_range(0, _targets.size() - 1)
	var t := _targets[_target_idx]
	# 把聚光灯放在目标外侧+上方一点，并【确实瞄准物体】照射。
	var outward := t.position.normalized()
	_spotlight.position = t.position + outward * 3.0 + Vector3(0, 2.0, 0)
	# 若到目标的视线接近竖直，换 up 向量避免 look_at 退化（保证灯锥对准物体）。
	var to_t := (t.global_position - _spotlight.position).normalized()
	var up := Vector3.UP if absf(to_t.dot(Vector3.UP)) < 0.99 else Vector3.FORWARD
	_spotlight.look_at(t.global_position, up)
	_spotlight.visible = true
	_lit = true
	_initial_aim_error = _aim_error()  # 记录"一开始的视角差"=整局可得的最大累计奖励
	_prev_aim_error = _initial_aim_error  # 以亮起瞬间为基准，第一帧差分≈0


func _task_physics(dt: float) -> void:
	if not _lit:
		if _sim_elapsed >= _trigger_time:
			_light_target()
	else:
		_lit_elapsed += dt


# 相机前向与“到目标方向”的夹角（弧度）。未亮起时返回 PI（视为最大误差）。
func _aim_error() -> float:
	if _target_idx < 0:
		return PI
	var fwd := -camera.global_transform.basis.z
	var to_t := (_targets[_target_idx].global_position - camera.global_position).normalized()
	return fwd.angle_to(to_t)


# 离散控制：disc[0..3] 四个加速键 -> 角加速度。
func _angular_accel() -> Vector2:
	var pitch_acc := (float(_disc[ACT_PITCH_UP]) - float(_disc[ACT_PITCH_DOWN])) * ACCEL
	var yaw_acc := (float(_disc[ACT_YAW_LEFT]) - float(_disc[ACT_YAW_RIGHT])) * ACCEL
	return Vector2(pitch_acc, yaw_acc)


func _compute_reward() -> float:
	# 灯未亮：无目标，不计奖励。
	if not _lit:
		return 0.0
	var err := _aim_error()
	var prev := _prev_aim_error
	_prev_aim_error = err              # 注意：本函数每渲染只被调用一次（见 ModelBase）
	# 差分项：和上次瞄准角之差。靠近(err变小)给正，远离(err变大)扣分（Σ = 初始视角差 - 最终视角差）。
	# 稠密项：按【当前误差 × 经过的仿真时间】扣分 = 误差对时间的积分。
	#   最优(瞬间瞄准并稳住 err≈0)时稠密项≈0 → 总分仍≈初始视角差；越慢/越偏，扣得越多 → 打破"不动=0"。
	return (prev - err) * REWARD_SCALE - err * ERR_PENALTY * _sim_dt_last


func _is_done() -> bool:
	if not _lit:
		return false
	return _aim_error() < HIT_THRESHOLD or _lit_elapsed >= MAX_AIM_SIM   # 命中 或 超时放弃


# ---- 独立运行：方向键 = 四个离散加速键，便于人工验证 ----
func _standalone_input() -> void:
	_disc[ACT_PITCH_UP] = 1 if Input.is_action_pressed("ui_up") else 0
	_disc[ACT_PITCH_DOWN] = 1 if Input.is_action_pressed("ui_down") else 0
	_disc[ACT_YAW_LEFT] = 1 if Input.is_action_pressed("ui_left") else 0
	_disc[ACT_YAW_RIGHT] = 1 if Input.is_action_pressed("ui_right") else 0


func _on_standalone_tick() -> void:
	if _lit:
		_accum_reward += _reward
	var status := ""
	if not _lit:
		status = "等待聚光灯…  %.2fs / %.2fs" % [_sim_elapsed, _trigger_time]
	elif _is_done():
		status = "命中! 亮起后用时 %.2fs" % _lit_elapsed
	else:
		status = "目标已亮! 瞄准误差 %.1f°" % rad_to_deg(_aim_error())
	_hud.text = "%s\n偏航 %.0f°  俯仰 %.0f°\nreward %+.3f   累计 %.3f / 初始视角差 %.3f\nsim_dt %.4f   info.sim_dt(传给模型)=%.4f" % [
		status, rad_to_deg(_yaw), rad_to_deg(_pitch),
		_reward, _accum_reward, _initial_aim_error, _sim_dt_last, get_sim_dt()]
	if _is_done():
		reset()      # 命中即开新回合（新延迟 + 新目标布局）
