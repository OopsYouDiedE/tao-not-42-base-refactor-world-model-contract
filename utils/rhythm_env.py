import torch


class ProceduralRhythmEnv:
    """
    纯 Tensor 驱动的微型二维音游物理引擎。
    支持数百个 Batch 环境的绝对并行。

    音符状态(每个 6 维): [active, y_head, track, color, speed, length]
      - y_head : 前缘(先到判定线的一端)像素 y,向下增大
      - speed  : 该音符自身恒定下落速度(进屏随机采样)——逼模型推断速度,而非套常数
      - length : 长按音符的长度(像素),0=点击(tap)。时长 = length / speed
    """

    def __init__(self, batch_size=32, device="cpu", tracer_mode=False,
                 speed_min=60.0, speed_max=160.0, hold_prob=0.5,
                 hold_len_min=20.0, hold_len_max=80.0, spawn_prob=2.0):
        self.B = batch_size
        self.device = device
        # tracer_mode: 每个 batch 同时至多一个音符,空了立刻补一个。
        # 用于隔离"单实体穿越遮挡"的存/预测证明(此模式下 length=0,只考验位置)。
        self.tracer_mode = tracer_mode

        # 画面参数
        self.H = 256
        self.W = 256
        self.num_tracks = 4
        self.track_xs = torch.linspace(self.W / (self.num_tracks * 2),
                                       self.W - self.W / (self.num_tracks * 2),
                                       self.num_tracks, device=device)
        self.hit_line_y = 220.0

        # 物理参数(速度改为逐音符随机)
        self.speed_min, self.speed_max = speed_min, speed_max
        self.note_size = 24.0
        self.spawn_prob = spawn_prob
        # 长按参数(tracer 模式下强制 tap)
        self.hold_prob = hold_prob
        self.hold_len_min, self.hold_len_max = hold_len_min, hold_len_max

        # 视觉遮挡盲区(此 Y 区间涂黑,强迫内部世界模型推演)
        self.blind_zone = (100.0, 160.0)

        # [active, y_head, track, color, speed, length]
        self.max_notes = 16
        self.notes = torch.zeros(self.B, self.max_notes, 6, device=device)

        self.colors = torch.tensor([
            [0.9, 0.1, 0.1], [0.1, 0.9, 0.1],
            [0.1, 0.1, 0.9], [0.9, 0.9, 0.1],
        ], device=device)

        self.current_time = torch.zeros(self.B, device=device)

        # 预计算像素整数坐标:render 热路径不再 .item(),避免 CPU-GPU 同步
        self.track_x_px = [int(round(x.item())) for x in self.track_xs]
        self.hit_y_px = int(self.hit_line_y)
        self.blind_y0, self.blind_y1 = int(self.blind_zone[0]), int(self.blind_zone[1])
        self._ys = torch.arange(self.H, device=device).float()
        self._xs = torch.arange(self.W, device=device).float()

    def reset(self):
        self.notes.zero_()
        self.current_time.zero_()

    def step(self, dt):
        """推进物理时间 dt 秒"""
        self.current_time += dt

        active_mask = self.notes[:, :, 0] > 0.5
        # 逐音符速度
        self.notes[:, :, 1] += self.notes[:, :, 4] * dt * active_mask.float()

        # 移除越界音符(尾端也掉出屏幕底部才算结束:y_head - length 越界)
        tail_y = self.notes[:, :, 1] - self.notes[:, :, 5]
        out_of_bounds = tail_y > self.H + self.note_size
        self.notes[:, :, 0] = torch.where(out_of_bounds, 0.0, self.notes[:, :, 0])

        # 生成新音符(全向量化:无 .any() guard、无 where(cond)[0],GPU 零同步)
        spawn_mask = torch.rand(self.B, device=self.device) < (self.spawn_prob * dt)
        if self.tracer_mode:
            has_active_now = (self.notes[:, :, 0] > 0.5).any(dim=1)
            spawn_mask = ~has_active_now

        free_slots = self.notes[:, :, 0] < 0.5
        has_free = free_slots.any(dim=1)
        valid_spawn = spawn_mask & has_free                       # [B] bool
        first_free_idx = free_slots.float().argmax(dim=1)         # [B] 目标空槽

        rand_track = torch.randint(0, self.num_tracks, (self.B,), device=self.device).float()
        rand_color = torch.randint(0, 4, (self.B,), device=self.device).float()
        rand_speed = (torch.rand(self.B, device=self.device)
                      * (self.speed_max - self.speed_min) + self.speed_min)
        if self.tracer_mode:
            rand_len = torch.zeros(self.B, device=self.device)
        else:
            is_hold = torch.rand(self.B, device=self.device) < self.hold_prob
            rand_len = is_hold.float() * (
                torch.rand(self.B, device=self.device)
                * (self.hold_len_max - self.hold_len_min) + self.hold_len_min)

        b_all = torch.arange(self.B, device=self.device)
        # 目标槽本就是空的(全 0);valid 时写入新值,否则写回原值(no-op)
        cur = self.notes[b_all, first_free_idx]                   # [B, 6]
        new = torch.stack([
            torch.ones_like(rand_speed),
            -self.note_size * torch.ones_like(rand_speed),
            rand_track, rand_color, rand_speed, rand_len,
        ], dim=1)                                                 # [B, 6]
        self.notes[b_all, first_free_idx] = torch.where(
            valid_spawn.unsqueeze(1), new, cur)

    def render(self):
        """渲染一帧画面 [B, 3, H, W]。全向量化:固定 N 步循环 + 纯 tensor 掩码,无 .item()。"""
        canvas = torch.zeros(self.B, 3, self.H, self.W, device=self.device)

        for ix in self.track_x_px:
            canvas[:, :, :, ix - 1:ix + 2] = 0.2
        hy = self.hit_y_px
        canvas[:, :, hy - 2:hy + 2, :] = torch.tensor(
            [0.8, 0.8, 0.8], device=self.device).view(1, 3, 1, 1)

        half_s = self.note_size / 2
        ys, xs = self._ys, self._xs                          # [H], [W]
        # 固定遍历 max_notes(不依赖数据,无同步);每个槽用掩码贴色,后者覆盖前者(z 序)
        for n in range(self.max_notes):
            active = self.notes[:, n, 0] > 0.5               # [B]
            y = self.notes[:, n, 1]
            length = self.notes[:, n, 5]
            trk = self.notes[:, n, 2].long().clamp(0, self.num_tracks - 1)
            clr = self.notes[:, n, 3].long().clamp(0, 3)
            x = self.track_xs[trk]                           # [B]
            y_top = y - half_s - length
            y_bot = y + half_s
            ymask = (ys.unsqueeze(0) >= y_top.unsqueeze(1)) & (ys.unsqueeze(0) <= y_bot.unsqueeze(1))  # [B,H]
            xmask = (xs.unsqueeze(0) >= (x - half_s).unsqueeze(1)) & (xs.unsqueeze(0) <= (x + half_s).unsqueeze(1))  # [B,W]
            box = active.view(-1, 1, 1) & ymask.unsqueeze(2) & xmask.unsqueeze(1)  # [B,H,W]
            color = self.colors[clr]                         # [B,3]
            canvas = torch.where(
                box.unsqueeze(1),
                color.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.H, self.W),
                canvas)

        canvas[:, :, self.blind_y0:self.blind_y1, :] = 0.05  # 盲区涂黑
        return canvas

    def get_tracer_state(self):
        """tracer 模式下读出被追踪音符的真实状态(供 slot-0 监督/证伪)。

        即使音符落在盲区(像素涂黑),y/speed 仍是物理真值——存世界证明的对照标尺。
        speed 用于在 harness 里算 oracle(匀速解析解)上界。
        """
        active = self.notes[:, :, 0] > 0.5
        has_active = active.any(dim=1)
        idx = active.float().argmax(dim=1)
        b = torch.arange(self.B, device=self.device)

        y = self.notes[b, idx, 1]
        track = self.notes[b, idx, 2].long()
        color = self.notes[b, idx, 3].long()
        speed = self.notes[b, idx, 4]
        occluded = has_active & (y >= self.blind_zone[0]) & (y <= self.blind_zone[1])

        return {
            "active": has_active.float(),
            "y": y, "track": track, "color": color, "speed": speed,
            "occluded": occluded,
        }

    def get_upcoming_actions(self, K):
        """多音符模式:返回每个 batch 接下来 K 个待击打动作的 GT 集合(供集合匹配监督)。

        动作 = (track, onset, duration)。onset=前缘到判定线的剩余时间;duration=length/speed。
        按 onset 升序取最近 K 个,不足处 valid=0。
        """
        active = self.notes[:, :, 0] > 0.5            # [B, N]
        y = self.notes[:, :, 1]
        speed = self.notes[:, :, 4].clamp(min=1e-3)
        length = self.notes[:, :, 5]
        track = self.notes[:, :, 2].long()

        upcoming = active & (y < self.hit_line_y)      # 前缘尚未过线
        onset_all = (self.hit_line_y - y) / speed
        dur_all = length / speed
        INF = 1e9
        onset_all = torch.where(upcoming, onset_all, torch.full_like(onset_all, INF))

        K = min(K, self.max_notes)
        vals, idx = torch.topk(onset_all, K, dim=1, largest=False)   # [B, K]
        b = torch.arange(self.B, device=self.device).unsqueeze(1).expand(-1, K)
        dur_sel = dur_all[b, idx]
        track_sel = track[b, idx]
        valid = (vals < INF * 0.5).float()

        return {
            "onset": vals * valid,        # [B, K] 秒(无效位清零)
            "duration": dur_sel * valid,  # [B, K] 秒
            "track": track_sel,           # [B, K] long
            "valid": valid,               # [B, K] 1=真动作
        }

    def get_expert_actions(self):
        """(兼容旧 API)每轨道下一次完美击打的绝对时间戳 T_k。返回 [B, num_tracks]。"""
        target_times = torch.full((self.B, self.num_tracks), float('inf'), device=self.device)
        active = self.notes[:, :, 0] > 0.5
        speed = self.notes[:, :, 4].clamp(min=1e-3)
        for trk in range(self.num_tracks):
            trk_mask = active & (self.notes[:, :, 2] == trk)
            dist_to_hit = self.hit_line_y - self.notes[:, :, 1]
            valid_mask = trk_mask & (dist_to_hit > 0)
            time_to_hit = dist_to_hit / speed
            time_to_hit = torch.where(valid_mask, time_to_hit, torch.full_like(time_to_hit, float('inf')))
            min_time_to_hit, _ = time_to_hit.min(dim=1)
            target_times[:, trk] = self.current_time + min_time_to_hit
        return target_times
