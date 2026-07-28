"""开发期查看工具，不参与数据管线与训练的生产路径。

目前只有 ``action_inspector``：Gradio 界面，逐帧核对 Lumine 动作编码结果。
依赖单列在 ``tools`` extra 里，数据管线与训练机器不必安装。
"""
