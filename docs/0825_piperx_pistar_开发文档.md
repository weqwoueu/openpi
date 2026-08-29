# PiperX 黑色插头任务 PiStar/RECAP 开发文档

> 日期：2026-08-25  
> 分支：`liuzijian/pistar-piperx`  
> 当前 HEAD：`3a71ee8`
> 当前阶段：**107 条专家示教和 norm stats 已完成，PiperX 普通 pi0.5 SFT 正在训练；SFT/PiStar 共用的 WebSocket 推理入口已经接通**
> 任务：`put the black plug into the two-hole socket`

## 1. 当前结论

目前已经具备以下能力：

1. PiperX 主臂到从臂的 MIT 模式软件遥操；
2. 两路 RealSense 以 `640x480@30Hz` 录制视频；
3. 遥操控制线程以 `60Hz` 下发动作；
4. 数据以 LeRobot v2.1 格式保存，单臂 `state/action` 均为 7 维；
5. `action` 保存实际下发给从臂的绝对目标，不再用下一帧 state 代替；
6. PiperX 数据到 pi0.5 的 policy transform、data config 和首轮 SFT config 已接好；
7. 配置导入、命令入口、数据结构和聚焦单元测试已经通过。

当前第一版 SFT 已经开始训练。checkpoint 保存完成后，可使用本文第 10.4 节的 WebSocket 入口做真机基线推理。

目前还不能写成“完整 RECAP 已经跑通”。完整 SFT checkpoint、真机基线推理、DAgger、Value Model、Advantage 标注和 PiStar CFG 训练都还没有在本任务上完成。

这里的实现基于仓库现有 pi0.5/PiStar 代码，是 RECAP 风格的迭代流程，不应表述为已经复现官方 pi0.6 RECAP。

## 2. 相比 0817 文档的关键更新

`0817_piperx_pistar_开发文档.md` 保留为历史设计记录；从 2026-08-25 起，以本文档的当前事实为准。

| 项目 | 0817 旧口径 | 0825 当前口径 |
|---|---|---|
| 相机分辨率 | 720p 优先 | 固定 `640x480` |
| 数据采集频率 | `10Hz` | `30Hz` |
| 遥操控制频率 | 与采集循环耦合 | 独立 `60Hz` 控制线程 |
| 静止帧过滤 | `MOVE_CHECK=True` | `MOVE_CHECK=False`，保留连续的 30Hz 采样目标 |
| 从臂控制 | 可选位置/MIT | 只使用 MIT 模式 |
| action 语义 | 下一帧 state fallback | 实际同步下发的绝对命令，缺失直接报错 |
| 训练动作 | 前 6 维可转 delta | 保留原始绝对量，不做 delta 变换 |
| 首轮训练配置 | 计划 train/infer 多份配置 | 只使用 `pi05_piperx_plug_sft` |
| 路径管理 | 文档写死实验机路径 | 数据、assets、checkpoint 路径由使用者管理 |

## 3. 当前数据流

```mermaid
flowchart TD
    M[PiperX 主臂\ncan_left_mas] --> T[MIT 遥操映射\n60Hz]
    T --> S[PiperX 从臂\ncan_left_slave]
    S --> Q[从臂 7D state]
    T --> A[实际下发 7D action]
    H[头部相机\n640x480@30] --> D[LeRobot v2.1 数据集]
    W[腕部相机\n640x480@30] --> D
    Q --> D
    A --> D

    D --> QC[上传后数据质检]
    QC --> N[Norm Stats]
    N --> SFT[pi05_piperx_plug_sft]
    SFT --> B[第一版 checkpoint 与真机基线]
    B --> DG[DAgger rollout/人工接管]
    DG --> V[Value Model]
    V --> ADV[Advantage 标签]
    ADV --> STAR[PiStar/CFG 训练]
```

主线顺序是：

```text
专家示教 -> 数据质检 -> norm stats -> 普通 pi0.5 SFT
         -> 第一版权重真机推理 -> DAgger
         -> Value Model -> Advantage 标签 -> PiStar/RECAP
```

不要跳过普通 SFT 直接训练 Value。Value 数据应来自第一版策略的真实 rollout，至少包含成功、失败以及人工接管信息；只用全成功专家示教训练出的 value 对策略错误区域没有足够监督。

## 4. 真机与采集参数

### 4.1 硬件角色

| 项目 | 当前配置 |
|---|---|
| 主臂 | PiperX，`can_left_mas` |
| 从臂 | PiperX，`can_left_slave` |
| 从臂控制模式 | MIT，控制码 `0xAD` |
| 头部相机键 | `observation.images.cam_head` |
| 腕部相机键 | `observation.images.cam_wrist` |
| 相机 profile | 两路均为 `640x480@30Hz` |
| 初始关节 | `[0, 1.0, -1.0, 1.0, 0, 0]` rad |
| 初始夹爪 | `0` |

### 4.2 遥操参数

采集入口中的当前参数如下：

```python
FPS = 30
CONTROL_FPS = 60
MOVE_CHECK = False

TEACHER_ACTION_EMA_ENABLED = True
TEACHER_ACTION_EMA_ALPHA = 0.80
TEACHER_ACTION_SLEW_ENABLED = True
TEACHER_ACTION_MAX_JOINT_STEP = 0.040
TEACHER_ACTION_MAX_GRIPPER_STEP = 0.025 / 0.07
```

含义：

- `FPS=30`：图像和训练样本以 30Hz 落盘。
- `CONTROL_FPS=60`：主从跟随独立以 60Hz 运行，减小操控迟滞和阶跃感。
- `MOVE_CHECK=False`：静止时也保留帧，使视频、timestamp、state 和 action 保持连续的 30Hz 采样目标。
- `EMA_ALPHA=0.80`：输出更偏向当前目标，同时保留少量上一时刻结果用于抑制抖动。
- `MAX_JOINT_STEP=0.040`：每个 60Hz 控制 tick 单关节目标最多变化 `0.040rad`。
- `MAX_GRIPPER_STEP=0.025/0.07`：每个控制 tick 的归一化夹爪变化上限。

控制频率和采集频率不需要相同。60Hz 控制负责动作手感，30Hz 采样已经能覆盖常见桌面操作，同时比 60Hz 视频显著节省存储和训练 I/O。训练数据必须以元数据中的实际 `fps=30` 为准。

## 5. 数据集合同

### 5.1 每帧核心字段

| 字段 | shape | 语义 |
|---|---:|---|
| `observation.state` | `(7,)` | 从臂 6 个关节角 rad + 归一化夹爪开度 |
| `action` | `(7,)` | 同一采样时刻实际下发的 6 个绝对关节目标 rad + 绝对夹爪目标 |
| `observation.images.cam_head` | `(3,480,640)` | 头部 RGB 视频 |
| `observation.images.cam_wrist` | `(3,480,640)` | 腕部 RGB 视频 |
| `intervention` | `(1,)` | 人工干预标记；纯专家示教按现有采集逻辑保存 |
| `value_label` | `(1,)` | 采集器生成的辅助标签，不能等同于已训练 Value Model 的预测 |
| `reward` | `(1,)` | 逐帧 reward 字段 |
| `reward_label` | `(1,)` | 终止结果相关标签 |
| `adv_ind` | string | 当前专家 episode 默认可写 `positive`；普通 SFT 不使用它 |

### 5.2 动作语义

原始数据中的 `observation.state` 和 `action` 都是 7 维。模型配置中的 `action_dim=32` 只是 pi0.5 内部 padding 维度：

```text
dataset action: 7D absolute
        -> transform 保持绝对量
        -> PadStatesAndActions 到 32D
        -> 模型输出 32D
        -> PiperOutputs 取前 7D
```

当前 `extra_delta_transform=False`，所以训练时不会把前 6 个关节动作减去当前 state。部署端也不能再额外做一次 delta/absolute 转换。

### 5.3 最终 SFT 数据

最终训练数据为 `piperx_black_plug_0825_v3`：

| 项目 | 数值 |
|---|---:|
| episodes | 107 |
| frames | 66,358 |
| videos | 214 |
| fps | 30 |
| 图像 | 两路 `640x480` AV1 |
| state/action | 单臂 7D 绝对量 |

全局 `index` 为连续的 `0..66357`，每条 episode 的 parquet、两路视频和 metadata 长度一致。OpenPI 数据加载验证得到 `(B,32)` state、`(B,50,32)` action chunk 和两路有效图像输入。

## 6. 当前采集操作

数据目标地址在 `2_arm_record.sh` 顶部人工修改：

```bash
REPO_ID="piperx/piperx_black_plug_demo"
OUTPUT_DIR="<LEROBOT_DATA_ROOT>"
```

在部署机仓库中依次执行：

```bash
cd <PISTAR_ROOT>/control_your_robot

bash scripts/piperx/2_arm_can_activate.sh
bash scripts/piperx/2_arm_go_init.sh
bash scripts/piperx/2_arm_record.sh
```

当前 `2_arm_go_init.sh` 的顺序是：

```text
one_arm_go_init.sh can_left_slave
-> 等待
-> one_arm_go_init.sh can_left_mas
```

`2_arm_replay.sh` 仍保留旧测试参数，当前还有 `FPS=10`，且需要人工补齐仓库根目录变量。它不属于本轮 30Hz 数据的已验证入口，本文档不把“回放通过”作为开始 SFT 的前置条件。

## 7. 上传后的数据验收

最终数据已经上传到训练服务器：

```text
/mnt/kpfs_juice/liuzijian/data/lerobot/piperx_black_plug_0825_v3/piperx_black_plug_0825_v3
```

训练时的 `HF_LEROBOT_HOME` 按第 9 节设置。

### 7.1 元数据必须满足

```text
codebase_version = v2.1
robot_type = piperx
fps = 30
observation.state.shape = [7]
action.shape = [7]
cam_head.shape = [3, 480, 640]
cam_wrist.shape = [3, 480, 640]
任务文本 = put the black plug into the two-hole socket
每个 episode 对应两条可解码视频
```

### 7.2 数值检查

至少检查：

1. 所有 state/action 都是 finite，不含 `NaN/Inf`；
2. 关节 0 到 5 在实际遥操段有明显变化；
3. gripper 位于 `[0,1]` 且开合段有变化；
4. action 不是全 0，也不是机械复制下一帧 state；
5. 两路视频帧数与 episode 长度一致；
6. timestamp 单调递增，实际间隔接近 `1/30s`；
7. 随机抽查头部/腕部视频，没有长时间黑帧、冻结或相机串位；
8. 任务成功标准一致，失败/误操作 episode 不混进专家成功集。

本批最终上传数据已经通过上述验收。

## 8. PiperX SFT 适配

### 8.1 唯一首轮配置

当前只新增一个配置：

```python
TrainConfig(
    name="pi05_piperx_plug_sft",
    model=Pi0Config(
        pi05=True,
        pistar=False,
        action_dim=32,
        action_horizon=50,
    ),
    data=LeRobotPiperDataConfig(
        repo_id="piperx_black_plug_0825_v3",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=240,
    num_workers=16,
    weight_loader=CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=30_000,
)
```

这个配置沿用隔壁已多次训练的 Tianji pi0.5 配置骨架：`action_dim=32`、`action_horizon=50`、`batch_size=240`、`30k steps`。PiperX 的区别是单臂原始 7D、双相机和绝对关节动作，且当前不启用 train-time RTC。

### 8.2 为什么 SFT 不需要 infer 配置

普通 SFT 设置 `pistar=False`：

- tokenizer 不读取 `adv_ind`；
- 数据中即使存在 `adv_ind`，也会在 transform 中被移除；
- 不执行 advantage-conditioned CFG；
- 同一份 config 可以用于训练和加载该 SFT checkpoint。

因此当前没有 `pi05_piperx_plug_sft_infer`，也不应为了命名对称额外增加一份内容相同的配置。

## 9. 训练服务器环境

新建训练窗口：

```bash
tmux new -s piperx_sft
```

进入窗口后执行：

```bash
cd /mnt/kpfs_juice/liuzijian/code/openpi

unset VIRTUAL_ENV PYTHONPATH

export WANDB_ENTITY=franciscoyllydekirat870-lll
export WANDB_PROJECT=openpi
export WANDB_DIR=/mnt/kpfs_juice/liuzijian/cache/wandb
export HF_LEROBOT_HOME=/mnt/kpfs_juice/liuzijian/data/lerobot/piperx_black_plug_0825_v3
export HF_HOME=/mnt/kpfs_juice/liuzijian/cache/huggingface
export OPENPI_DATA_HOME=/mnt/kpfs_juice/liuzijian/cache/openpi
export UV_CACHE_DIR=/mnt/kpfs_juice/liuzijian/cache/uv
```

离开 tmux：`Ctrl+B` 后按 `D`。重新进入：

```bash
tmux attach -t piperx_sft
```

## 10. Norm Stats 与正式训练

### 10.1 计算 norm stats

第 9 节环境变量设置完成后，在 OpenPI 根目录执行完整数据归一化：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --no-sync scripts/compute_norm_stats.py \
  --config-name pi05_piperx_plug_sft
```

### 10.2 首次启动 SFT

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

### 10.3 断点续训

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run --no-sync scripts/train.py pi05_piperx_plug_sft \
  --exp-name piperx_black_plug_0825_v3_sft_run_001 \
  --resume \
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

首次启动使用 `--overwrite`；断点续训只使用 `--resume`，二者不能同时出现。训练真正启动的判据是日志出现 `Step 0` 及 finite loss。

### 10.4 SFT WebSocket 推理

服务端和客户端的可修改设置均集中在脚本顶部。服务端主要修改 `REPO_ROOT`、具体 step 的 `CHECKPOINT_DIR`、`TRAIN_CONFIG` 和监听端口；`CHECKPOINT_DIR` 必须直接包含 `params/` 和对应 norm stats，不能填写 run 根目录或 `params/` 子目录。

训练/推理服务器执行：

```bash
cd /mnt/kpfs_juice/liuzijian/code/openpi
bash control_your_robot/scripts/piperx/run_server.sh
```

部署机先完成 CAN 和初始姿态，再运行客户端：

```bash
cd <PISTAR_ROOT>/control_your_robot
bash scripts/piperx/2_arm_can_activate.sh
bash scripts/piperx/2_arm_go_init.sh
bash scripts/piperx/run_client.sh
```

`run_client.sh` 当前只做真机推理，不创建或写入数据集。默认设置为 `CONTROL_FREQ=30`、`CHUNK_SIZE=10`、`NUM_EPISODES=1`；`NUM_EPISODES=-1` 表示无限重复，任意阶段按 `Ctrl+C` 都作为正常退出，并关闭相机、WebSocket 和 CAN 连接。

这里的 `30Hz` 是每个已返回 action chunk 内的目标发送节拍。客户端每执行 10 步会同步请求下一个 chunk，因此两个 chunk 之间仍会包含网络和模型推理耗时，不应描述为严格无间隙的全程 30Hz 控制。

普通 SFT 保持 `ADV_IND=""`、`ADV_GUIDANCE_BETA=""`。未来 PiStar 推理使用同一套入口，但需要换成对应的 PiStar infer config/checkpoint，并设置 `ADV_IND="positive"`。DAgger 使用下面的独立客户端，推理条件和原始数据标签互不混用。

### 10.5 WebSocket DAgger 采集

先在训练/推理服务器启动与当前 checkpoint 对应的服务：

```bash
cd /mnt/kpfs_juice/liuzijian/code/openpi
bash control_your_robot/scripts/piperx/run_server.sh
```

部署机修改 `2_arm_record_dagger.sh` 顶部的数据地址、服务地址和 CAN 名称，然后执行：

```bash
cd <PISTAR_ROOT>/control_your_robot
bash scripts/piperx/2_arm_can_activate.sh
bash scripts/piperx/2_arm_go_init.sh
bash scripts/piperx/2_arm_record_dagger.sh
```

状态机固定为：

```text
READY --Enter--> AUTONOMOUS --Space--> INTERVENTION --Enter--> LABEL
LABEL --Right--> success --Enter--> save
LABEL --Left----> failure --Enter--> save
save --Enter--> next AUTONOMOUS
```

单个 episode 内 `Space` 只允许从策略切到专家一次，进入专家后不能切回策略。切换时清空剩余 action chunk、递增 generation 并丢弃迟到的旧推理结果；模式切换和主从对齐期间不采帧。自主阶段同一条策略 action 同步发送给主臂和从臂；专家阶段主臂保持拖动示教，从臂固定使用 MIT `0xAD`，60Hz 遥操控制并以 30Hz 落盘。

每帧保存实际下发的 7D 绝对 action。策略帧 `intervention=0`，接管完成后的专家帧 `intervention=1`。Right/Left 只决定 episode 的 success/failure 及其 value/reward 标签；原始 rollout/DAgger 数据无论成功失败都写 `adv_ind=none`，最终 positive/negative advantage 由后续 Value/Advantage 流程生成。`Ctrl+C` 丢弃尚未保存的 episode；若已经开始保存，则完成本次保存后退出。

默认 `MAX_STEPS=900`，即 30Hz 下单回合最多 30 秒，随后自动进入 LABEL；`NUM_EPISODES=-1` 仍表示可连续采无限多个回合。若把 `MAX_STEPS` 改为 `-1`，单回合的两路原始图像会一直驻留内存直到保存或丢弃，不适合无人看守运行。

## 11. 当前代码改动摘要

| 文件 | 当前改动 |
|---|---|
| `src/openpi/policies/piper_policy.py` | 严格检查 7D state/action；支持 HWC/CHW；映射双相机；补空的第三相机和 mask；输出前 7D |
| `src/openpi/training/config.py` | 新增 `LeRobotPiperDataConfig` 数据映射和唯一的 `pi05_piperx_plug_sft` |
| `src/openpi/training/data_loader.py` | 使用标准 `repo_id -> LeRobotDataset` 加载路径 |
| `src/openpi/models/pi0.py` | 普通 pi0.5 loss 返回 `(B,H)` 的逐时间步 MSE，保留 PiStar 加权损失接口 |
| `control_your_robot/scripts/serve_piper_single_pi05star_websocket.py` | 从根 OpenPI 加载 checkpoint，发布 SFT/PiStar metadata 和可选 guidance 参数 |
| `control_your_robot/example/deploy/piper_single_on_PI0_websocket.py` | PiperX 新输入合同、MIT 绝对 7D 动作、chunk 内 30Hz 与正常退出清理 |
| `control_your_robot/example/collect/collect_lerobot_dagger_websocket.py` | WebSocket rollout、单向专家接管、success/failure 标注和 LeRobot 写入 |
| `control_your_robot/my_robot/piper_dagger.py` | 主从 CAN 参数化、自主双臂同步、从臂 MIT 下发和实际 action 返回 |
| `control_your_robot/scripts/piperx/run_server.sh` | 服务端人工配置和启动入口 |
| `control_your_robot/scripts/piperx/run_client.sh` | 无采集功能的真机推理入口 |
| `control_your_robot/scripts/piperx/2_arm_record_dagger.sh` | DAgger 数据地址、服务地址、频率和遥操滤波参数入口 |
| `scripts/train.py` | 在 train step 对模型 loss 做总 mean |
| `src/openpi/shared/console.py` | 提供 value/label/weight loader 使用的文本日志辅助函数 |
| `src/openpi/policies/piper_policy_test.py` | 覆盖图像映射、绝对 action、shape/finite 校验和 7D 输出 |
| `src/openpi/training/piperx_config_contract_test.py` | 锁定 PiperX SFT 配置合同 |

上述内容位于 `liuzijian/pistar-piperx` 工作树；提交和推送由仓库维护者统一完成。

## 12. 已完成验证

当前已完成：

1. 107 条最终专家数据完成并上传；
2. 107 个 parquet、214 个视频和 66,358 帧通过一致性验证；
3. `pi05_piperx_plug_sft` 已指向最终 repo ID；
4. OpenPI loader 已生成 `(B,32)` state、`(B,50,32)` action 和有效图像 batch；
5. norm stats 已生成，四卡 SFT 正在训练；
6. DAgger 键盘、单向状态机、generation、MIT 和实际 action 语义通过离线测试；`control_your_robot/tests` 当前为 43 passed。

尚未完成：完整 SFT checkpoint、真机基线、DAgger 真机首轮验收、Value、Advantage 和 PiStar 训练。

## 13. RECAP 后续各阶段状态

| 阶段 | 当前状态 | 进入下一阶段前必须完成 |
|---|---|---|
| 专家示教 | 已完成并上传 | 保留最终 v3 数据不再原地修改 |
| 普通 pi0.5 SFT | norm stats 已完成，训练进行中 | 训练完成、checkpoint 推理 |
| SFT 真机基线 | 未开始 | 固定任务布置和评测标准，记录成功/失败/干预 |
| DAgger | WebSocket 采集入口和离线测试已完成，尚未真机首轮验收 | 用 SFT checkpoint 跑首轮，检查视频、action/state、intervention 和 success/failure 标签 |
| Value Model | 有实现代码，当前入口不可直接运行 | 修复本地数据加载合同、模型权重路径并完成目标任务训练 |
| Advantage 标注 | 有脚本，当前入口不可直接运行 | 修复本地数据加载合同；只在派生副本上写标签 |
| PiStar/CFG | 有模型分支，尚无 PiperX 任务配置 | 建立 PiperX PiStar config，确保每条样本都有有效 `adv_ind`，接通 unconditional guidance 输入 |
| 最终真机评测 | 未开始 | 固定 checkpoint、场景、次数、成功定义和干预统计 |

### 13.1 DAgger

普通 SFT checkpoint 部署后，使用 `2_arm_record_dagger.sh` 采集模型 rollout 和专家 pilot 接管：

```text
policy action -> 正常执行并记录 intervention=0
expert pilot 接管 -> 执行专家 action 并记录 intervention=1
episode 结束 -> 保存真实成功/失败结果
```

新入口直接连接根 OpenPI 的 WebSocket 服务，不再使用旧 vendored `PI0_SINGLE`。模型请求使用当前 `state + cam_high + cam_wrist + prompt` 合同，输出取前 7 维绝对 action。网络推理和 CAN 控制解耦，action chunk 不足时重复最近一次真实命令，并按真实执行时间轴继续采样。

首轮真机验收后检查：

- 数据集包含两个相机的 `videos/` 文件，视频帧数与 parquet 一致；
- `observation.state` 和 `action` 都是 7D，且运动维度存在合理变化；
- Space 前 `intervention=0`，切换完成后 `intervention=1`，过渡帧没有写入；
- 成功轨迹终帧 reward/value 与失败轨迹标签符合当前 collector 定义；
- 全部原始帧 `adv_ind=none`。

DAgger 数据保持为独立 repo，不与专家 v3 原地合并。后续训练是否混合专家和 rollout 数据，应由新的派生训练集或多数据源 loader 明确完成；旧 `scripts/merge_datasets.py` 仍不能直接用于当前字段。

### 13.2 Value Model

仓库里存在 Value Model 训练代码，也存在外部训练过的 value checkpoint，但这不等于当前黑色插头任务的 value 已训练完成。旧 checkpoint 的任务分布、输入字段和效果没有在本任务上验证。

当前 `scripts/train_value.py` 和 `scripts/label_advantage_from_vlm.py` 会构造 `DataConfig(local_data_dir=...)`，而当前稳定 `DataConfig` 没有该字段；因此这两条入口在修复前不能直接用于最终数据。

另外还有三处明确问题：

- Value 图像输入候选键没有完整覆盖当前 `observation.images.cam_wrist`，腕部图像可能被静默补零；
- Value weight loader 仍包含其他机器的 SigLIP/Gemma 绝对路径，需要改为显式参数，但实际路径仍由使用者管理；
- `scripts/check_value_data.py` 与当前 `ValueDataLoader` 构造参数不一致，当前会直接 `TypeError`，不能当作已经可用的验收入口。

采集器中的两个监督信号不要混淆：`value_label` 是成功轨迹约从 `-1` 线性走到 `0`、失败轨迹为 `-1`，供当前 Value 训练入口使用；`reward_label` 是 Advantage 的 N-step 累积项。

### 13.3 Advantage 与 PiStar

未来 PiStar 训练时：

- `pistar=True`，每个训练样本必须提供 `adv_ind`；
- `adv_ind_dropout=True` 时，tokenizer 以 30% 概率去掉 advantage condition，让模型同时学习有条件和无条件分支；
- 推理时应使用 `adv_ind_dropout=False`，避免同一输入随机改变条件；
- CFG 还要求生成 `tokenized_prompt_uncond`，当前通用 `ModelTransformFactory` 尚未把 `adv_guidance_input=True` 接入 PiperX 配置。

这就是 PiStar 阶段可能需要 train/infer 两份配置的原因。它与当前普通 SFT 无关，所以首轮 SFT 只保留一个 config。

当前 Advantage 脚本实现的是未折扣 N-step 形式：

```text
A_t = sum(reward_label[t:t+N]) + V(t+N) - V(t)
```

越过 episode 末端时未来 value 取 0。脚本会原地重写 parquet，因此未来必须先复制出派生数据集，再运行标注；原始专家/DAgger 数据保持不变。

## 14. 下一步执行清单

按以下 gate 顺序推进：

1. 完成当前 `pi05_piperx_plug_sft` 训练；
2. 使用统一 WebSocket 入口完成第一版权重真机推理；
3. 使用新入口完成 PiperX rollout/DAgger 真机首轮验收；
4. 用 rollout/接管数据训练目标任务 Value Model；
5. 在数据副本上生成 advantage 标签；
6. 新增并验证 PiperX PiStar train/infer config；
7. 固定评测方案，对 SFT 与 PiStar 做同条件对照。

当前最直接的工作边界是第 1 到第 3 项：**先把第一版普通 SFT 跑稳，再进入 DAgger/Value/RECAP。**

## 15. 关键文件索引

| 用途 | 文件 |
|---|---|
| 专家遥操采集 | `control_your_robot/example/collect/collect_lerobot_master_slave_teleop.py` |
| 数据集写入 | `control_your_robot/src/robot/data/collect_lerobot_rl.py` |
| PiperX LeRobot 封装 | `control_your_robot/my_robot/piper_single_lerobot.py` |
| 相机 profile | `control_your_robot/src/robot/sensor/Realsense_sensor.py` |
| 双臂 CAN 激活 | `control_your_robot/scripts/piperx/2_arm_can_activate.sh` |
| 双臂初始化 | `control_your_robot/scripts/piperx/2_arm_go_init.sh` |
| 采集入口 | `control_your_robot/scripts/piperx/2_arm_record.sh` |
| Piper policy transform | `src/openpi/policies/piper_policy.py` |
| 训练配置 | `src/openpi/training/config.py` |
| 数据加载 | `src/openpi/training/data_loader.py` |
| Norm stats | `scripts/compute_norm_stats.py` |
| 训练入口 | `scripts/train.py` |
| 推理服务端 | `control_your_robot/scripts/piperx/run_server.sh` |
| 真机推理客户端 | `control_your_robot/scripts/piperx/run_client.sh` |
| DAgger 采集 | `control_your_robot/example/collect/collect_lerobot_dagger_websocket.py` |
| DAgger 启动 | `control_your_robot/scripts/piperx/2_arm_record_dagger.sh` |
| Value 训练 | `scripts/train_value.py` |
| Advantage 标注 | `scripts/label_advantage_from_vlm.py` |
