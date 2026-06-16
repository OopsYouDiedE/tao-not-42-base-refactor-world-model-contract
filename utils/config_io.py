"""配置文件读取(yaml → dict)。

对外接口:
    load_yaml(path) — 读 yaml 预设为 plain dict(供 net.config.ModelConfig.from_dict)。

横向工具:只做文件 IO + yaml 解析,不含任何模型/领域知识(领域校验在 train,schema 在 net)。
"""
import yaml


def load_yaml(path):
    """读 yaml 文件为 dict。缺包(pyyaml)/缺文件直接报错(AGENTS §2:不写降级)。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
