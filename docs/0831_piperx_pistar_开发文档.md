# PiperX 黑色插头任务 PiStar/RECAP 复现进度

> 日期：2026-08-31  
> 分支：`liuzijian/pistar-piperx`  
> 代码基准：`d6d94be`  
> 任务：`put the black plug into the two-hole socket`  
> 当前阶段：SFT、真机推理、DAgger 和 Value 数据合并已完成，Value Model 正在 H200 上正式训练

## 1. 当前已经打通的链路

```text
PiperX MIT 专家示教
  -> LeRobot v2.1 双相机数据
  -> pi0.5 SFT norm stats 与训练
  -> WebSocket 真机策略推理
  -> DAgger 自主 rollout + 单向专家接管
  -> episode 删除与多数据集合并
  -> DAgger + 人工成功数据的 Value 训练集
  -> SigLIP 2 + Gemma3 Value Model 正式训练（当前）
```

| 阶段 | 当前结果 |
|---|---|
| CAN 与双臂控制 | `can_left_mas` / `can_left_slave` 固定映射，主从臂初始化、MIT 遥操和 MIT 回放已使用 |
| 专家示教 | 30 Hz 数据采集、60 Hz MIT 遥操、两路 `640x480` 图像、7D 绝对 state/action 已完成 |
| 普通 pi0.5 SFT | `pi05_piperx_plug_sft` 的 norm stats、训练和 checkpoint 已完成 |
| SFT 真机部署 | OpenPI WebSocket server 与 PiperX client 已完成真机推理 |
| DAgger | 策略自主运行、Space 单向切换专家、成功/失败标注和 LeRobot 落盘已完成 |
| 数据处理 | DAgger episode 删除、人工成功数据标注、多个 LeRobot v2.1 数据集合并已完成 |
| Value 数据 | 当前混合数据共 138 episodes、80,585 帧，包含 DAgger rollout/接管数据和 40 条人工成功数据 |
| Value 训练 | 本地 SigLIP 参数 21/21、Gemma 参数 200/200 加载完成，H200 已进入 30k step 正式训练 |
| Value 监控 | W&B online 已接通；约 600 step 时 loss 总体由 5.57 降至约 4.86，速度约 2.7 it/s |
| Value checkpoint | 保存 `params`、`ema_params`、`opt_state` 和 `step`，支持按 step 严格续训 |
| Value 后处理 | Value 预测导出和 Advantage 标注入口已对齐 PiperX 双相机、任务文本及 RL 字段 |

当前链路属于基于仓库 pi0.5/PiStar 实现的 RECAP 风格复现：

```text
demonstrations -> SFT -> rollout/intervention -> Value
               -> advantage labels -> PiStar policy -> next rollout
```

## 2. 当前数据合同

### 2.1 机器人输入输出

| 字段 | shape | 语义 |
|---|---:|---|
| `observation.state` | `(7,)` | 从臂 6 个关节角 rad + 归一化夹爪开度 |
| `action` | `(7,)` | 实际下发的 6 个绝对关节目标 rad + 绝对夹爪目标 |
| `observation.images.cam_head` | `(3,480,640)` | 头部 RGB 视频 |
| `observation.images.cam_wrist` | `(3,480,640)` | 腕部 RGB 视频 |
| `intervention` | `(1,)` | 策略帧为 `0`，专家帧为 `1` |
| `value_label` | `(1,)` | Value Model 监督目标，范围 `[-1,0]` |
| `reward_label` | `(1,)` | N-step Advantage 的逐帧奖励项 |
| `adv_ind` | string | 原始 DAgger 为 `none`，Value 后处理生成 `positive/negative` |

原始 7D state/action 在 OpenPI 内部 pad 到 32D。PiperX 训练和部署均使用绝对动作：

```text
7D absolute action -> pad to 32D -> pi0.5 -> output first 7D -> PiperX MIT
```

### 2.2 Value 混合数据

当前训练服务器数据目录：

```text
/mnt/kpfs_juice/liuzijian/data/lerobot/piperx_plug_dagger_mix/piperx_plug_dagger_mix
```

混合规则：

- DAgger 数据保留逐帧 `intervention/reward/reward_label/value_label`；
- 人工成功数据作为 `intervention=1` 的成功示教加入；
- 人工成功数据的成功终帧 `reward=1`，轨迹生成对应的 `value_label/reward_label`；
- Value Model 使用混合数据的 `value_label` 训练；
- 原始 DAgger 数据的 `adv_ind=none`，待 Value 训练完成后再在派生副本上生成 advantage 标签。

## 3. 部署机复现命令

部署仓库路径集中设置为：

```bash
export PISTAR_ROOT=/home/standard/workspace/pistar/openpi
cd "$PISTAR_ROOT"
source my_env.sh
```

### 3.1 激活 CAN

```bash
cd "$PISTAR_ROOT/control_your_robot"
bash scripts/piperx/2_arm_can_activate.sh
```

当前固定映射：

```text
1-4.1.1 -> can_left_mas
1-4.1.3 -> can_left_slave
bitrate  -> 1000000
```

只检查接口：

```bash
bash scripts/piperx/2_arm_can_activate.sh --check
```

### 3.2 双臂初始化

先在 `control_your_robot/scripts/piperx/2_arm_go_init.sh` 顶部设置部署仓库路径和复位速度：

```bash
REPO_ROOT=/home/standard/workspace/pistar/openpi
SPEED=15
GRIPPER=0
MOVE_WAIT_SECONDS=5
```

执行：

```bash
cd "$PISTAR_ROOT/control_your_robot"
bash scripts/piperx/2_arm_go_init.sh
```

当前顺序为从臂到任务初始位，等待后主臂到任务初始位。需要主臂先经过零位时执行：

```bash
bash scripts/piperx/2_arm_go_star.sh
```

### 3.3 MIT 专家示教采集

在 `control_your_robot/scripts/piperx/2_arm_record.sh` 顶部设置：

```bash
REPO_ID="piperx/piperx_black_plug_demo"
OUTPUT_DIR="/home/standard/agilex/lerobot"
```

开始采集：

```bash
cd "$PISTAR_ROOT/control_your_robot"
bash scripts/piperx/2_arm_record.sh
```

当前采集配置：

```text
sample FPS    = 30
teleop FPS    = 60
image         = 640x480, cam_head + cam_wrist
state/action  = 7D absolute
control mode  = MIT 0xAD
```

### 3.4 数据回放

在 `2_arm_replay.sh` 设置数据集、episode 和与元数据一致的 FPS：

```bash
DATASET_DIR="/path/to/lerobot_dataset"
EPISODE_INDEX=0
FPS=30
```

执行：

```bash
cd "$PISTAR_ROOT"
REPO_ROOT="$PISTAR_ROOT" \
  bash control_your_robot/scripts/piperx/2_arm_replay.sh
```

## 4. SFT 训练与部署复现

### 4.1 训练服务器环境变量

```bash
cd /mnt/kpfs_juice/liuzijian/code/openpi
source my_env.sh

export WANDB_ENTITY=franciscoyllydekirat870-lll
export WANDB_PROJECT=openpi
export WANDB_DIR=/mnt/kpfs_juice/liuzijian/cache/wandb
export HF_LEROBOT_HOME=/mnt/kpfs_juice/liuzijian/data/lerobot/piperx_black_plug_0825_v3
export HF_HOME=/mnt/kpfs_juice/liuzijian/cache/huggingface
export OPENPI_DATA_HOME=/mnt/kpfs_juice/liuzijian/cache/openpi
export UV_CACHE_DIR=/mnt/kpfs_juice/liuzijian/cache/uv
```

### 4.2 Norm stats

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --no-sync scripts/compute_norm_stats.py \
  --config-name pi05_piperx_plug_sft
```

### 4.3 pi0.5 SFT

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run --no-sync scripts/train.py pi05_piperx_plug_sft \
  --exp-name piperx_black_plug_0825_v3_sft_run_001 \
  --overwrite \
  --batch-size 240 \
  --num-workers 16 \
  --num-train-steps 30000 \
  --save-interval 10000 \
  --keep-period 10000 \
  --fsdp-devices 1 \
  --wandb-enabled \
  --assets-base-dir /mnt/kpfs_juice/liuzijian/code/openpi/assets \
  --checkpoint-base-dir /mnt/kpfs_juice/liuzijian/checkpoints/openpi
```

断点续训使用同一条命令，将 `--overwrite` 换为 `--resume`。

### 4.4 SFT WebSocket 服务

在 `control_your_robot/scripts/piperx/run_server.sh` 顶部填写当前 checkpoint、配置和端口：

```bash
REPO_ROOT=/mnt/kpfs_juice/liuzijian/code/openpi
CHECKPOINT_DIR=/path/to/pi05_piperx_plug_sft/step_dir
TRAIN_CONFIG=pi05_piperx_plug_sft
HOST=0.0.0.0
PORT=8000
ADV_GUIDANCE_BETA=""
```

服务端：

```bash
cd /mnt/kpfs_juice/liuzijian/code/openpi
bash control_your_robot/scripts/piperx/run_server.sh
```

部署机策略推理：

```bash
cd "$PISTAR_ROOT/control_your_robot"
bash scripts/piperx/2_arm_can_activate.sh
bash scripts/piperx/2_arm_go_init.sh
bash scripts/piperx/run_client.sh
```

普通 SFT 使用：

```bash
ADV_IND=""
ADV_GUIDANCE_BETA=""
```

## 5. DAgger 复现

先启动第 4.4 节的 SFT WebSocket 服务。在部署机修改 `2_arm_record_dagger.sh` 顶部：

```bash
REPO_ID="piperx/piperx_plug_dagger_demo"
OUTPUT_DIR="/home/standard/agilex/lerobot"
TASK_PROMPT="put the black plug into the two-hole socket"
SERVER_HOST="<POLICY_SERVER_IP>"
SERVER_PORT=8000

SAMPLE_FPS=30
TELEOP_FPS=60
CHUNK_SIZE=50
ASYNC_PREFETCH_ENABLED=false
PREFETCH_THRESHOLD=25
ADV_IND=""
```

执行：

```bash
cd "$PISTAR_ROOT/control_your_robot"
bash scripts/piperx/2_arm_can_activate.sh
bash scripts/piperx/2_arm_go_init.sh
bash scripts/piperx/2_arm_record_dagger.sh
```

单条 episode 操作：

```text
Enter -> 策略自主运行
Space -> 主臂对齐从臂并进入一次性专家接管
Enter -> 结束轨迹
Right/Left -> 成功/失败
Enter -> 保存并进入下一条
```

每帧保存真实执行的绝对 action。策略帧写 `intervention=0`，接管后的专家帧写 `intervention=1`。每个数据集同时生成：

```text
meta/dagger_episode_summary.jsonl
```

## 6. DAgger 数据删除与合并

工具仓库：

```bash
cd /home/standard/workspace/gitlab/robocoin/tools/lerobot_dataset_tools
source my_env.sh
```

### 6.1 删除指定 episodes

```bash
python 2_remove_episodes.py \
  --repo_id piperx/piperx_plug_dagger_demo \
  --root /path/to/piperx_plug_dagger_demo \
  --episodes 1,3,5-7 \
  --output_dir /path/to/piperx_plug_dagger_clean \
  --new_repo_id piperx/piperx_plug_dagger_clean
```

### 6.2 合并 DAgger 与人工成功数据

在 `3_merge_multi_v21_lerobot_datasets.py` 顶部填写：

```python
dataset_ids = [
    "piperx/piperx_plug_dagger_clean",
    "piperx/piperx_manual_success",
]
successful_demo_dataset_ids = [
    "piperx/piperx_manual_success",
]
new_repo_id = "local/piperx_plug_dagger_mix"
tolerance_s = 1e-4
```

运行：

```bash
python 3_merge_multi_v21_lerobot_datasets.py
```

该合并结果同时保留 DAgger RL 标签，并为人工成功数据生成统一的成功轨迹标签。

## 7. H200 Value Model 训练复现

### 7.1 当前路径与离线运行变量

```bash
cd /mnt/kpfs_juice/liuzijian/code/openpi
source my_env.sh

export VALUE_DATASET=/mnt/kpfs_juice/liuzijian/data/lerobot/piperx_plug_dagger_mix/piperx_plug_dagger_mix
export VALUE_CKPT=/mnt/kpfs_juice/liuzijian/checkpoints/piperx_plug_value_v1
export VALUE_ASSET_ROOT="$PWD/.offline/value_assets"

export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$VALUE_CKPT"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
```

本地 Value assets 目录结构：

```text
.offline/value_assets/
|-- tokenizer.model
|-- gemma-3-270m/
`-- siglip2-so400m-patch14-224-jax/
    `-- siglip2_so400m14_224.npz
```

### 7.2 正式训练

```bash
.offline/uv run --no-sync python scripts/train_value.py \
  --data_dir "$VALUE_DATASET" \
  --checkpoint_dir "$VALUE_CKPT" \
  --tokenizer_path "$VALUE_ASSET_ROOT/tokenizer.model" \
  --siglip_path "$VALUE_ASSET_ROOT/siglip2-so400m-patch14-224-jax/siglip2_so400m14_224.npz" \
  --gemma_checkpoint_dir "$VALUE_ASSET_ROOT/gemma-3-270m" \
  --load_pretrained \
  --freeze_mode all_backbones \
  --fsdp_devices 1 \
  --batch_size 32 \
  --num_train_steps 30000 \
  --save_interval 10000 \
  --num_workers 12 \
  --wandb_mode online \
  --wandb_project openpi \
  --wandb_entity franciscoyllydekirat870-lll \
  --wandb_run_name piperx_plug_value_v1 \
  --log_interval 100
```

### 7.3 Value 断点续训

保持第 7.1 节环境和原训练参数不变，在训练命令末尾追加：

```bash
--resume_from_checkpoint step_00010000
```

目录名按实际保存的 `step_XXXXXXXX` 选择。

## 8. Value 训练完成后的复现命令

### 8.1 导出逐帧 Value 预测

```bash
.offline/uv run --no-sync python scripts/export_vlm_values.py \
  --data_dir "$VALUE_DATASET" \
  --checkpoint_dir "$VALUE_CKPT" \
  --checkpoint_name step_00030000 \
  --tokenizer_path "$VALUE_ASSET_ROOT/tokenizer.model" \
  --base_image_col observation.images.cam_head \
  --wrist_image_col observation.images.cam_wrist \
  --batch_size 8 \
  --num_workers 12 \
  --output_path "$VALUE_CKPT/value_predictions.parquet"
```

导出表用于对照：

```text
episode / frame / intervention / reward_label
value_label / predicted value / value error / oracle advantage
```

### 8.2 在派生数据集上生成 Advantage 标签

```bash
export ADV_DATASET=/mnt/kpfs_juice/liuzijian/data/lerobot/piperx_plug_dagger_mix_adv

.offline/uv run --no-sync python scripts/label_advantage_from_vlm.py \
  --data_dir "$ADV_DATASET" \
  --checkpoint_dir "$VALUE_CKPT" \
  --checkpoint_name step_00030000 \
  --tokenizer_path "$VALUE_ASSET_ROOT/tokenizer.model" \
  --base_image_col observation.images.cam_head \
  --wrist_image_col observation.images.cam_wrist \
  --human_col intervention \
  --reward_col reward_label \
  --lookahead 50 \
  --top_percent 30 \
  --batch_size 8 \
  --num_workers 12
```

标注规则：

- 全程专家 demo 保留 positive；
- rollout 中 `intervention=1` 的专家帧标为 positive；
- rollout 中 `intervention=0` 的自主帧按全局 advantage top 30% 标为 positive，其余标为 negative。

## 9. 后续计划

### 9.1 完成并评估 Value Model

1. 完成 30k steps，保留 10k、20k、30k checkpoint；
2. 对每个候选 checkpoint 导出逐帧预测；
3. 按 episode 对照成功/失败、专家/自主阶段的 Value 曲线；
4. 选定用于 Advantage 标注的 Value checkpoint。

### 9.2 生成 PiperX PiStar 数据

1. 复制当前混合数据为派生数据集；
2. 使用选定 Value checkpoint 写入 `adv_ind`；
3. 统计 demo、intervention、自主 positive 和自主 negative 的帧数；
4. 固化用于 PiStar 训练的数据版本。

### 9.3 新增 PiperX PiStar 配置并训练

保持 PiperX 的双相机、7D 绝对动作和 32D padding 合同，新增极简配置：

```text
train: pistar=True, adv_ind_dropout=True
infer: pistar=True, adv_ind_dropout=False, guidance enabled
```

随后执行：

```text
PiStar norm stats -> PiStar policy training -> WebSocket positive-condition inference
```

### 9.4 真机对照与下一轮迭代

固定任务布置、episode 数和成功标准，对比：

```text
SFT baseline
vs
PiStar positive-condition + guidance
```

记录每组的成功数、失败数、人工接管数和完成时间，再把下一轮 rollout/接管数据加入新的混合数据集，重复 Value、Advantage 和 PiStar 训练链路。

## 10. 关键文件

| 用途 | 文件 |
|---|---|
| 专家 MIT 采集 | `control_your_robot/example/collect/collect_lerobot_master_slave_teleop.py` |
| CAN 激活 | `control_your_robot/scripts/piperx/2_arm_can_activate.sh` |
| 双臂初始化 | `control_your_robot/scripts/piperx/2_arm_go_init.sh` |
| 专家采集入口 | `control_your_robot/scripts/piperx/2_arm_record.sh` |
| 回放入口 | `control_your_robot/scripts/piperx/2_arm_replay.sh` |
| PiperX SFT config | `src/openpi/training/config.py` |
| SFT 训练 | `scripts/train.py` |
| WebSocket server | `control_your_robot/scripts/piperx/run_server.sh` |
| WebSocket client | `control_your_robot/scripts/piperx/run_client.sh` |
| DAgger collector | `control_your_robot/example/collect/collect_lerobot_dagger_websocket.py` |
| DAgger 入口 | `control_your_robot/scripts/piperx/2_arm_record_dagger.sh` |
| Value 训练 | `scripts/train_value.py` |
| Value 预测导出 | `scripts/export_vlm_values.py` |
| Advantage 标注 | `scripts/label_advantage_from_vlm.py` |
| Value 架构复核 | `docs/value_function_architecture_review.md` |

