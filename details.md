# awq search

```python
x = [
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
]

o0 = wx

view(-1, x.shape[-1]) =

[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

x_max = x.abs().view(-1, x.shape[-1]).mean(dim=0)

= [4, 5, 6]

for ratio in range(n_grid)/n_grid:
    scales = x_max.pow(ratio).view(-1)
    w = w * scales
    w = Q(w) / scales
    o = wx
    loss = criterion(o, o0)
    if loss < best_loss:
        best_scales = scales
return best_scales
```

```python
# 非对称量化
def Q(w):
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    scales = (max_val - min_val) / max_int
    zeros = torch.round(min_val / scales)
    w = (torch.round(w / scales - zeros) + zeros) * scales
```

```python
# auto clip
# 取值范围过大会导致量化的时候精度不够，clip 可以牺牲掉极端值的精度，提高整体精度
max_val = org_max_val * (1 - i_s / n_grid)
# 过程类似上面，总是以 q_group_size 为一组进行
```