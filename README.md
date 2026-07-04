# Skywriter 轨迹预处理（降噪 + 提取）

毕业设计 pipeline 的**第一步**：把 Skywriter（MGC3130）采集的空中手绘轨迹做**降噪**与**提取**，输出规整后的笔画与 Sketch-RNN 所需的 **stroke-3** 格式，供后续重建阶段使用。

```
原始 (t, x, y, z) → 抬笔/落笔判定 → 降噪 → 分段/重采样/归一化 → stroke-3
```

## 模块

| 文件 | 作用 |
|------|------|
| `trajectory.py` | 轨迹数据结构 + CSV 读写 |
| `pen_state.py`  | 抬笔/落笔判定（Z 高度双阈值迟滞；支持显式 pen 列） |
| `denoise.py`    | 离群跳点剔除 + 1€ 滤波平滑（实时友好） |
| `extract.py`    | 分段、弧长等间距重采样、归一化、转 stroke-3 |
| `visualize.py`  | 三联图：原始 / 降噪+落笔 / 提取笔画 |
| `demo.py`       | 模拟数据跑通全流程，或读入真实 CSV |
| `rt_filters.py` | 实时滤波链（中值/尖刺门控/1€/z 迟滞抬落笔），树莓派与离线端共用 |
| `mgc3130.py`    | MGC3130 传感器 I2C 读取，三个树莓派脚本共用（部署时随脚本一起拷） |

## 抬笔 / 落笔方案

默认用 **Z 高度双阈值迟滞**：手压低（z 小）落笔，抬高（z 大）抬笔；落笔阈值 `0.30`、抬笔阈值 `0.45`，中间区间保持上一状态以防抖动。可在 `pen_state.pen_from_z()` 调整阈值，或改用 `z_is_height=False`。
若采集时已有显式落笔标记，在 CSV 里加一列 `pen`（1=落笔，0=抬笔），会被优先使用。

## 使用

```bash
pip install -r requirements.txt      # 电脑端(离线管线)
# 树莓派端: pip install -r requirements_pi.txt(见该文件头部的部署说明)

# 1) 用模拟数据跑通（会生成 data/synthetic.csv 和 out/ 下的结果）
python demo.py

# 2) 用真实采集数据（CSV 至少含 t,x,y，可选 z,pen）
python demo.py data/your_capture.csv
```

输出：
- `out/<name>_pipeline.png`：降噪与提取效果三联图
- `out/<name>_stroke3.npy`：stroke-3 序列（下一步喂给 Sketch-RNN）

## CSV 格式

| 列 | 含义 |
|----|------|
| `t` | 时间戳（秒），缺省按帧序 |
| `x`, `y` | 归一化坐标 0~1（手不在感应区可留空） |
| `z` | 手到面板高度 0~1（用于抬笔/落笔），可选 |
| `pen` | 显式落笔标记 1/0，可选（优先于 z） |

## 下一步

stroke-3 序列将作为 Sketch-RNN 的输入，进入**降噪重建**阶段（pipeline 第二步）。
