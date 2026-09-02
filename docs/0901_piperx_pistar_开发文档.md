# PiperX PiStar/RECAP 复现进度

> 日期：2026-09-01  
> 分支：`liuzijian/pistar-piperx`  
> 代码基准：`fd06e41`  
> 本文只记录 09-01 新完成的 Value 评估、增量数据标注和 PiStar 训练输入，不重复 08-31 已记录的采集、SFT、DAgger、环境及 Value 训练流程。

## 1. Value Model 训练完成

Value Model 已完成 30,000 steps 训练，本轮选用的最终 checkpoint 为：

```text
/mnt/kpfs_juice/liuzijian/checkpoints/piperx_plug_value_v1/step_00030000
```

逐帧预测结果保存为：

```text
/mnt/kpfs_juice/liuzijian/checkpoints/piperx_plug_value_v1/value_predictions_smoke.parquet
```

## 2. Value 逐帧评估结果

本次共评估 80,585 帧。以下结果来自 Value 训练所用混合数据的全量回放，用于确认模型已经拟合标签和轨迹趋势，不作为独立测试集泛化指标。

| 指标 | 数值 |
|---|---:|
| 帧数 | 80,585 |
| `value_label` 均值 | -0.591392 |
| `vlm_value` 均值 | -0.593782 |
| `value_label` 标准差 | 0.324417 |
| `vlm_value` 标准差 | 0.318244 |
| MAE | 0.036702 |
| 绝对误差中位数 | 0.023858 |
| 最大绝对误差 | 0.624308 |

按 episode 的 `reward_max > 0.5` 区分成功和失败轨迹：

| 轨迹类型 | 平均 MAE | 平均起始 Value | 平均结束 Value |
|---|---:|---:|---:|
| 失败 | 0.044637 | -0.998979 | -0.969175 |
| 成功 | 0.035377 | -0.974403 | -0.040400 |

成功轨迹的预测值从接近 `-1` 上升到接近 `0`，失败轨迹结束时仍接近 `-1`，与当前 Value 标签定义一致。

误差尾部分布：

| 绝对误差阈值 | 帧数 | 比例 |
|---|---:|---:|
| `> 0.1` | 5,691 | 7.06% |
| `> 0.2` | 926 | 1.15% |
| `> 0.4` | 39 | 0.05% |

## 3. 新一轮数据与 Advantage 标注

新一轮 DAgger 数据与人工成功数据完成合并后，原始累计数据集为：

```text
/mnt/kpfs_juice/liuzijian/data/lerobot/piperx_plug_dagger_mix2_v1
```

用于写入 Advantage 标签的派生副本为：

```text
/mnt/kpfs_juice/liuzijian/data/lerobot/piperx_plug_dagger_mix2_v1_adv
```

本轮数据规模：

| 数据部分 | 帧数 | Value 处理方式 |
|---|---:|---|
| 全部数据 | 128,495 | - |
| 全程人工成功 demo | 28,895 | 跳过 Value 推理，保留 `positive` |
| DAgger rollout | 99,600 | 使用 `step_00030000` 执行 Value 推理并计算 Advantage |

本轮没有重新训练 Value Model，而是使用已经完成评估的 `step_00030000` 为新增 rollout 生成标签。标注规则为：

- 全程人工成功 demo 保持 `positive`；
- rollout 中 `intervention=1` 的专家接管帧标为 `positive`；
- rollout 中 `intervention=0` 的自主帧按全局 Advantage 前 30% 标为 `positive`，其余标为 `negative`。

## 4. PiStar 训练输入已固化

PiStar 训练配置为 `pi05_piperx_plug_recap1`，关键合同如下：

| 配置项 | 当前值 |
|---|---|
| 数据集 `repo_id` | `piperx_plug_dagger_mix2_v1_adv` |
| 基础权重 | `pi05_base` |
| `pistar` | `True` |
| `adv_ind_dropout` | `True` |
| 动作语义 | 7D 绝对动作，不转换为相对 delta |
| 模型动作维度 | 32D padding |
| action horizon | 50 |
| 训练步数 | 30,000 |

至此，本轮新增链路已经完成：

```text
Value 30k checkpoint
  -> 全量逐帧评估
  -> 新 DAgger 数据 Value 推理
  -> Advantage positive/negative 标注
  -> piperx_plug_dagger_mix2_v1_adv
  -> pi05_piperx_plug_recap1 训练输入
```
