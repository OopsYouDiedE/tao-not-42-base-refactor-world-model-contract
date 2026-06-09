"""动作条件音游环境(Phigros 式扁音符 + 命中消除)。

在 ProceduralRhythmEnv 基础上加入**动作改变世界**的先天规则:
    音符头部在判定线 ±HIT_WINDOW(默认 60ms)内,对应轨道被按下 → 该音符消失(命中)。

与父类区别:
  - step(dt, action) 接收动作 [B, n_keys],先判命中消除,再推进物理;
  - render() 画**扁条音符**(宽、薄),更像 Phigros;不涂盲区(本任务考动作效果,非遮挡);
  - get_lane_state() 按轨道暴露"最近音符的 y / onset / 是否存在 / 是否可命中",供监督;
  - 记录 last_hit / last_hittable(供准确率统计)。
"""
import torch

from utils.rhythm_env import ProceduralRhythmEnv


class RhythmActionEnv(ProceduralRhythmEnv):
    HIT_WINDOW = 0.06   # ±60ms 判定窗
    MIN_GAP = 0.10      # 同 lane 相邻音符到线时间差下限(100ms)

    def __init__(self, batch_size=32, device="cpu", spawn_prob=4.0,
                 speed_min=60.0, speed_max=160.0, flat_h=6.0, flat_w=22.0):
        super().__init__(batch_size=batch_size, device=device, tracer_mode=False,
                         spawn_prob=spawn_prob, speed_min=speed_min, speed_max=speed_max,
                         hold_prob=0.0)   # tap-only,命中逻辑简洁
        self.flat_h, self.flat_w = flat_h, flat_w
        self.last_hit = torch.zeros(self.B, self.num_tracks, device=device)
        self.last_hittable = torch.zeros(self.B, self.num_tracks, device=device)

    def _lane_onset(self):
        active = self.notes[:, :, 0] > 0.5
        speed = self.notes[:, :, 4].clamp(min=1e-3)
        onset = (self.hit_line_y - self.notes[:, :, 1]) / speed   # [B,N] 到线时间(秒)
        lane = self.notes[:, :, 2].long().clamp(0, self.num_tracks - 1)
        return active, onset, lane

    def step(self, dt, action=None):
        """action: [B, n_keys] ∈[0,1],>0.5 视为按下。先判命中,再推进。"""
        self.current_time += dt
        active, onset, lane = self._lane_onset()
        in_win = active & (onset.abs() <= self.HIT_WINDOW)             # [B,N] 落在判定窗

        self.last_hit = torch.zeros(self.B, self.num_tracks, device=self.device)
        self.last_hittable = torch.zeros(self.B, self.num_tracks, device=self.device)
        for k in range(self.num_tracks):
            self.last_hittable[:, k] = (in_win & (lane == k)).any(dim=1).float()

        if action is not None:
            pressed = (action > 0.5)                                   # [B, n_keys]
            pressed_for_note = pressed.gather(1, lane)                 # [B,N] 该音符所在轨是否被按
            hit = in_win & pressed_for_note                           # [B,N]
            self.notes[:, :, 0] = torch.where(hit, torch.zeros_like(self.notes[:, :, 0]),
                                              self.notes[:, :, 0])     # 命中→消除
            for k in range(self.num_tracks):
                self.last_hit[:, k] = (hit & (lane == k)).any(dim=1).float()

        # --- 物理推进(tap-only,无长按) ---
        active = self.notes[:, :, 0] > 0.5
        self.notes[:, :, 1] += self.notes[:, :, 4] * dt * active.float()
        oob = self.notes[:, :, 1] > self.H + self.note_size
        self.notes[:, :, 0] = torch.where(oob, 0.0, self.notes[:, :, 0])

        # --- 生成新音符:允许每 lane 多个,但**同 lane 到线时间差 ≥ MIN_GAP** ---
        active = self.notes[:, :, 0] > 0.5
        cand_lane = torch.randint(0, self.num_tracks, (self.B,), device=self.device)   # 随机候选 lane
        cand_speed = (torch.rand(self.B, device=self.device)
                      * (self.speed_max - self.speed_min) + self.speed_min)
        spawn_y = -self.note_size
        cand_onset = (self.hit_line_y - spawn_y) / cand_speed                          # 新音符到线时间
        note_onset = (self.hit_line_y - self.notes[:, :, 1]) / self.notes[:, :, 4].clamp(min=1e-3)  # [B,N]
        in_lane = active & (self.notes[:, :, 2].long() == cand_lane.unsqueeze(1))      # 同 lane 现有音符
        NEG = torch.full_like(note_onset, -1e9)
        max_onset = torch.where(in_lane, note_onset, NEG).max(dim=1).values            # 同 lane 最晚到线者
        gap_ok = cand_onset > (max_onset + self.MIN_GAP)                               # 须再晚 ≥MIN_GAP 到线
        free = self.notes[:, :, 0] < 0.5
        spawn_mask = (torch.rand(self.B, device=self.device) < (self.spawn_prob * dt)) & gap_ok & free.any(dim=1)
        first_free = free.float().argmax(dim=1)
        rand_color = torch.randint(0, 4, (self.B,), device=self.device).float()
        b_all = torch.arange(self.B, device=self.device)
        cur = self.notes[b_all, first_free]
        new = torch.stack([torch.ones_like(cand_speed), spawn_y * torch.ones_like(cand_speed),
                           cand_lane.float(), rand_color, cand_speed, torch.zeros_like(cand_speed)], dim=1)
        self.notes[b_all, first_free] = torch.where(spawn_mask.unsqueeze(1), new, cur)

    def get_lane_state(self):
        """按轨道返回最近(到线)音符的状态。各为 [B, num_tracks]。"""
        active, onset, lane = self._lane_onset()
        INF = 1e9
        y = torch.zeros(self.B, self.num_tracks, device=self.device)
        present = torch.zeros(self.B, self.num_tracks, device=self.device)
        onset_lane = torch.full((self.B, self.num_tracks), INF, device=self.device)
        b_all = torch.arange(self.B, device=self.device)
        for k in range(self.num_tracks):
            m = active & (lane == k) & (onset > -self.HIT_WINDOW)      # 仍在逼近/刚到线
            on_k = torch.where(m, onset, torch.full_like(onset, INF))
            best, idx = on_k.min(dim=1)
            has = best < INF * 0.5
            present[:, k] = has.float()
            onset_lane[:, k] = torch.where(has, best, torch.full_like(best, INF))
            y[:, k] = torch.where(has, self.notes[b_all, idx, 1], torch.zeros_like(best))
        hittable = ((onset_lane.abs() <= self.HIT_WINDOW) & (present > 0.5)).float()
        return {"y": y, "onset": onset_lane, "present": present, "hittable": hittable}

    def render(self):
        """扁条音符(宽、薄),无盲区。[B,3,H,W]。"""
        canvas = torch.zeros(self.B, 3, self.H, self.W, device=self.device)
        for ix in self.track_x_px:
            canvas[:, :, :, ix - 1:ix + 2] = 0.2
        hy = self.hit_y_px
        canvas[:, :, hy - 2:hy + 2, :] = 0.8
        ys, xs = self._ys, self._xs
        for n in range(self.max_notes):
            active = self.notes[:, n, 0] > 0.5
            y = self.notes[:, n, 1]
            trk = self.notes[:, n, 2].long().clamp(0, self.num_tracks - 1)
            clr = self.notes[:, n, 3].long().clamp(0, 3)
            x = self.track_xs[trk]
            ymask = (ys.unsqueeze(0) >= (y - self.flat_h).unsqueeze(1)) & \
                    (ys.unsqueeze(0) <= (y + self.flat_h).unsqueeze(1))           # [B,H]
            xmask = (xs.unsqueeze(0) >= (x - self.flat_w).unsqueeze(1)) & \
                    (xs.unsqueeze(0) <= (x + self.flat_w).unsqueeze(1))           # [B,W]
            box = active.view(-1, 1, 1) & ymask.unsqueeze(2) & xmask.unsqueeze(1)  # [B,H,W]
            color = self.colors[clr]
            canvas = torch.where(box.unsqueeze(1),
                                 color.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.H, self.W),
                                 canvas)
        return canvas
