# 辅助工具库 (utils/)

本目录包含项目通用的辅助模块，涵盖标签提取、三维几何投影、物理一致性损失计算和训练可视化。

---

## 文件说明

### 1. [label_generator.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/label_generator.py) — GPU 标签生成器

* **核心算子**：`process_batch_on_gpu()`
* **功能说明**：全程在 GPU 上运行。使用 `scatter_reduce_`（基于 amin/amax）提取实例的 Ground-Truth 边界框，使用 `scatter_add_` 统计实例面积。无动态显存分配，无 CPU-GPU 同步点。

### 2. [geometry.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/geometry.py) — 相机重投影几何库

* **核心方法**：`inverse_warp()`、`generate_intrinsics()`
* **功能说明**：实现三维相机的反投影与重投影。根据相机内参和估计的绝对深度将像素坐标反投影为三维点，再依据预测的自运动（Ego-Pose，包含旋转和平移）将三维点投影到目标帧像素网格，通过双线性插值采样（Warping）提供自监督光度误差信号。

### 3. [losses.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/losses.py) — 损失函数库

* **核心方法**：`compute_track_loss()`、`compute_physics_loss()`
* **功能说明**：
  * 计算 SSIM 与 L1 混合的光度重投影损失、Edge-Aware 二阶深度平滑损失、光学流一致性损失。
  * **追踪损失**：在 Chunk 序列尺度上维护 GT 与 Query 之间的匈牙利匹配关系（抑制 ID Switch），在 CPU 上收集切片索引后在 GPU 上单次向量化计算 Smooth L1 与 BCE 损失。

### 4. [visualization.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/visualization.py) — 训练可视化

* **核心方法**：`save_vis_grid()`、`draw_boxes()`
* **功能说明**：在训练过程中异步 detach 梯度，将输入图像、实例分割图、预测深度图和预测光流图拼接为可视化网格并保存，不阻塞主训练线程。如遇 `torchvision` ABI 异常，自动降级回退到内置 NMS。
