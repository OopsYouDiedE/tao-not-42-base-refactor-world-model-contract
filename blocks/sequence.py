"""L1 primitive 积木库 - 序列扫描与 λ-return (blocks/sequence.py)。

对外接口:
    static_scan                    — 通用顺序扫描(沿时间维展开 fn,聚合每步输出为字典/列表)。
    static_scan_for_lambda_return  — λ-return 专用的逆向扫描(优化实现)。
    lambda_return                  — TD(λ) 回报(λ=1→蒙特卡洛,λ=0→1-step)。

来源:原样照抄 NM512/dreamerv3-torch 的 `tools.py`(MIT,见 net/dreamer/NOTICE)。
与任务无关的可复用序列算子:RSSM 的 observe/imagine 用 static_scan 沿时间递推,
ImagBehavior 用 lambda_return 在想象轨迹上算价值目标。函数体保持 1:1 原样。
"""
import torch


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs
