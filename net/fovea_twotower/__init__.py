"""凹视双塔:慢塔(tower.py,需 fla.GatedDeltaNet)+ 快塔(yolo_parse.py,不需 fla)。

tower 的符号惰性载入(PEP 562 __getattr__),使"仅用快塔(YOLO 解析头)"时不被 fla 依赖
阻塞;真正取用 ContextTower/ActionTower/act_featurize 时才 import tower(此时须装 fla)。
"""
_TOWER = {"ActionTower", "ContextTower", "act_featurize", "N_ACT", "N_PATCH", "D_LAT"}


def __getattr__(name):
    if name in _TOWER:
        from . import tower
        return getattr(tower, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
