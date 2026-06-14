"""gym3.types 的最小 vendored 替身——只实现 VPT 网络/动作头构造时用到的类型描述符。

VPT 用 gym3.types 仅作"动作空间的类型描述"(make_action_head 读 shape/eltype/n 建头,
CameraHierarchicalMapping 读 .size)。这里实现 ValType/Discrete/Real/TensorType/DictType
的构造与属性访问即可,**不引入 gym3/gym/minerl 依赖**(本仓库的硬约束:teacher 必须 minerl-free)。

只要这些类型产出的 action head 形状与原版一致(camera Discrete(121)、buttons Discrete(8641)),
OpenAI 发布的 .weights 就能按名加载——权重 key 由模块属性名决定,与类型实现无关。
"""
import numpy as np


class ValType:
    """所有类型描述符的基类(make_action_head 的类型注解/ isinstance 用)。"""
    pass


class Discrete(ValType):
    """离散元素类型:n 个类别。"""
    def __init__(self, n, **_):
        self.n = int(n)

    def __eq__(self, o):
        return isinstance(o, Discrete) and o.n == self.n

    def __hash__(self):
        return hash(("Discrete", self.n))

    def __repr__(self):
        return f"Discrete({self.n})"


class Real(ValType):
    """连续(高斯)元素类型。VPT 人类动作空间不用,仅为 make_action_head 分支完整性保留。"""
    def __init__(self, *_, **__):
        pass


class TensorType(ValType):
    """张量类型:shape + 每元素类型 eltype。"""
    def __init__(self, shape, eltype):
        self.shape = tuple(shape)
        self.eltype = eltype

    @property
    def size(self):
        return int(np.prod(self.shape)) if len(self.shape) else 1

    def __repr__(self):
        return f"TensorType(shape={self.shape}, eltype={self.eltype})"


class DictType(dict, ValType):
    """具名子类型的字典(动作空间根类型)。dict 子类 ⇒ .items()/[k] 直接可用。"""
    def __init__(self, **subtypes):
        super().__init__(subtypes)

    def __repr__(self):
        return f"DictType({dict.__repr__(self)})"
