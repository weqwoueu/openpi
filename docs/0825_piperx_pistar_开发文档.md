# PiperX 黑色插头任务 PiStar/RECAP 开发文档

> 日期：2026-08-25  
> 分支：`liuzijian/pistar-piperx`  
> 当前 HEAD：`926680c`  
> 当前阶段：**专家示教采集中；PiperX 普通 pi0.5 SFT 适配已完成，尚未生成最终 norm stats，尚未启动正式训练**  
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

因此，**完成本轮专家数据采集、上传和质检后，可以计算 norm stats 并启动第一版 SFT**。

目前还不能写成“完整 RECAP 已经跑通”。正式 SFT checkpoint、真机基线推理、DAgger、Value Model、Advantage 标注和 PiStar CFG 训练都还没有在本任务上完成。

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

### 5.3 旧 v3 测试集边界

`/home/standard/下载/piperx_black_plug_demo_v3` 已经验证过 parquet、视频和 7 维曲线能够被读取，但它不是本轮最终训练数据。其元数据仍是：

- 1 个 episode；
- `1280x720@10Hz`；
- prompt 是白色插头任务；
- 旧相机和旧采集设置。

它只能用于结构回归，不能用它的 norm stats 训练当前黑色插头任务。

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

最终数据上传到训练服务器后，先检查数据，不要立即算 norm stats。

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

只有这一批最终上传数据通过验收后，才生成正式 norm stats。

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
        repo_id="piperx/piperx_black_plug_demo",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=240,
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

隔壁 `/home/standard/workspace/gitlab/openpi` 的现有训练环境已确认包含：

```text
Python 3.11.15
JAX 0.5.3
PyTorch 2.7.1+cu126
Flax 0.10.2
LeRobot commit 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

这些核心版本与当前分支的锁文件相符。用户提交并推送当前改动后，在训练服务器切换分支：

```bash
cd /home/standard/workspace/gitlab/openpi
git fetch <remote>
git switch liuzijian/pistar-piperx
git pull --ff-only
git rev-parse --short HEAD

source .venv/bin/activate
uv sync --active --frozen
```

本仓库的 `lerobot` 仍是 Git commit 依赖，不是受 Git 跟踪的 `third_party/` 目录。若训练机本地没有该 commit 的 uv/Git 缓存，`uv sync` 仍需要能访问 GitHub；网络失败时要使用已有 uv 缓存或正常的 Git 网络代理，而不是把整个 `third_party/` 提交进当前分支。

注意：editable install 绑定具体源码路径。如果训练服务器使用另一个 checkout，只激活旧 `.venv` 不能保证导入新分支源码；切分支后必须执行一次 `uv sync --active --frozen`，再检查：

```bash
python -c "import openpi; print(openpi.__file__)"
python scripts/train.py --help | rg pi05_piperx_plug_sft
python scripts/compute_norm_stats.py --help
```

`my_env.sh` 会按当前目录设置 `UV_CACHE_DIR`、`HF_LEROBOT_HOME` 和 `OPENPI_DATA_HOME`。数据和 checkpoint 路径由使用者管理时，应先确认这些变量，不要盲目 source 后覆盖训练机原有路径：

```bash
printf '%s\n' "$UV_CACHE_DIR" "$HF_LEROBOT_HOME" "$OPENPI_DATA_HOME"
```

当前流程不需要执行 `git submodule update --init --recursive`。

## 10. Norm Stats 与正式训练

### 10.1 数据地址

在计算 norm stats 前，先确保 `pi05_piperx_plug_sft` 中的 `repo_id` 与最终上传数据一致。`train.py` 可以覆盖部分 config 字段，但当前 `compute_norm_stats.py` 的 CLI 只接收 `config_name` 和 `max_frames`，所以不要假设它会自动识别任意数据路径。

### 10.2 计算 norm stats

在 PiStar 根目录执行：

```bash
python scripts/compute_norm_stats.py \
  --config-name pi05_piperx_plug_sft
```

默认输出到：

```text
./assets/pi05_piperx_plug_sft/<repo_id>/norm_stats.json
```

验收 norm stats：

- `state` 和 `actions` 都有统计量；
- 统计基于原始 7 维数据，不是人为转换后的 delta；
- 数值全部 finite；
- 没有维度长期异常为 0；
- 使用的是本轮最终上传数据，而不是旧 v3。

### 10.3 启动训练

路径由使用者指定，命令模板如下：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_piperx_plug_sft \
  --exp-name=<EXP_NAME> \
  --assets-base-dir=<ASSETS_BASE_DIR> \
  --checkpoint-base-dir=<CHECKPOINT_BASE_DIR> \
  --overwrite
```

如果 `batch_size=240` 在目标 GPU 拓扑上不合适，再通过 config/CLI 调整；不要仅因为隔壁任务能跑就假设显存一定相同。

中断后恢复时，把 `--overwrite` 换成：

```bash
--resume
```

训练真正启动的判据是日志出现 `Step 0` 及 finite loss。只出现 `Initialized train state` 只证明模型和优化器初始化完成，不能算已经开始有效训练。

## 11. 当前代码改动摘要

| 文件 | 当前改动 |
|---|---|
| `src/openpi/policies/piper_policy.py` | 严格检查 7D state/action；支持 HWC/CHW；映射双相机；补空的第三相机和 mask；输出前 7D |
| `src/openpi/training/config.py` | 新增 `LeRobotPiperDataConfig` 数据映射和唯一的 `pi05_piperx_plug_sft` |
| `src/openpi/training/data_loader.py` | 恢复标准 `repo_id -> LeRobotDataset` 加载路径；移除当前不存在的本地目录分支和 HF monkey transform |
| `src/openpi/models/pi0.py` | 普通 pi0.5 loss 返回 `(B,H)` 的逐时间步 MSE，保留 PiStar 加权损失接口 |
| `scripts/train.py` | 在 train step 对模型 loss 做总 mean |
| `src/openpi/shared/console.py` | 补齐 value/label/weight loader 依赖的文本日志辅助函数 |
| `src/openpi/policies/piper_policy_test.py` | 覆盖图像映射、绝对 action、shape/finite 校验和 7D 输出 |
| `src/openpi/training/piperx_config_contract_test.py` | 锁定 PiperX SFT 配置合同 |

本文档生成时，上述训练适配仍在工作树中，尚未由助手提交；提交和推送由用户处理。`third_party/aloha` 的已有状态与本轮无关，未修改。

## 12. 已完成验证

当前已完成的离线验证：

1. `pi05_piperx_plug_sft` 可被 config registry 找到；
2. `scripts/train.py --help` 能正常加载并显示新配置；
3. `scripts/compute_norm_stats.py --help` 能正常加载；
4. 新增 Piper policy/config 聚焦测试：`10 passed`；
5. 联合现有 data loader 测试的聚焦回归：`15 passed`；
6. 旧 v3 测试数据能生成 `(B,32)` state 和 `(B,50,32)` action chunk；
7. 两路真实图像映射到 pi0.5，第三路图像 mask 为 false；
8. 普通 pi0.5 dummy loss 为 finite，shape 为 `(B,H)`；
9. 相关新文件通过 Ruff；
10. `git diff --check` 通过。

上述验证使用隔壁已经训练过的 OpenPI `.venv` 配合当前 PiStar 源码完成。当前 PiStar checkout 自身没有独立 `.venv`，所以这不等于训练服务器切分支后的最终环境验收。

尚未完成：

- 最终上传专家全集质检；
- 最终数据 norm stats；
- GPU 真实 batch；
- `Step 0` 和训练 loss；
- SFT checkpoint；
- 策略服务和真机推理；
- DAgger、Value、Advantage、PiStar 训练。

## 13. RECAP 后续各阶段状态

| 阶段 | 当前状态 | 进入下一阶段前必须完成 |
|---|---|---|
| 专家示教 | 进行中 | 上传最终数据并通过第 7 节质检 |
| 普通 pi0.5 SFT | 代码适配完成 | norm stats、真实 Step 0、训练完成、checkpoint 推理 |
| SFT 真机基线 | 未开始 | 固定任务布置和评测标准，记录成功/失败/干预 |
| DAgger | 有旧入口，PiperX 当前链路未重新验收 | 统一模型 config、双相机、7D 绝对 action 和采集字段 |
| Value Model | 有实现代码，当前入口不可直接运行 | 修复本地数据加载合同、模型权重路径并完成目标任务训练 |
| Advantage 标注 | 有脚本，当前入口不可直接运行 | 修复本地数据加载合同；只在派生副本上写标签 |
| PiStar/CFG | 有模型分支，尚无 PiperX 任务配置 | 建立 PiperX PiStar config，确保每条样本都有有效 `adv_ind`，接通 unconditional guidance 输入 |
| 最终真机评测 | 未开始 | 固定 checkpoint、场景、次数、成功定义和干预统计 |

### 13.1 DAgger

下一步应先部署普通 SFT checkpoint，采集模型 rollout 和专家 pilot 接管：

```text
policy action -> 正常执行并记录 intervention=0
expert pilot 接管 -> 执行专家 action 并记录 intervention=1
episode 结束 -> 保存真实成功/失败结果
```

当前旧 DAgger 代码包含 vendored OpenPI 和旧 config 约定，不能仅修改 checkpoint 路径就宣称已经可用。要先统一它与根目录的 PiperX policy/data contract。

当前具体阻塞包括：

- 默认 checkpoint、config、repo 和任务仍是旧 white-plug 版本；
- rollout 循环仍按 `10Hz` 执行动作，而当前专家数据是 `30Hz`；
- 模型 horizon 为 50，但旧逻辑只执行前 10 步，若继续按 10Hz 会把训练时序放慢约 3 倍；
- DAgger 使用 `control_your_robot` 内的 vendored OpenPI，其中没有 `pi05_piperx_plug_sft`；
- vendored inference 仍使用旧的 `observation/state`、`image`、`wrist_image` 键和错误的动作切片，与根目录当前 PiperX contract 不一致；
- `scripts/merge_datasets.py` 仍硬编码旧 `image/wrist_image/state/actions` 列，不能直接合并当前 `observation.images.cam_* / observation.state / action` 数据。

因此，第一版 SFT 训练完成后，下一项代码工作是统一 DAgger 推理、30Hz 执行和数据合并合同，而不是直接开始采 rollout。

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

1. 采完本轮黑色插头专家数据；
2. 上传后检查 metadata、两路视频、7D 数值、时间轴和任务文本；
3. 确认训练 config 的 `repo_id` 指向最终数据；
4. 计算正式 norm stats；
5. 在训练服务器完成分支/环境验收；
6. 启动 `pi05_piperx_plug_sft`，确认 `Step 0` 和 finite loss；
7. 训练出第一版权重并做离线/真机推理；
8. 统一并验收 PiperX DAgger；
9. 用 rollout/接管数据训练目标任务 Value Model；
10. 在数据副本上生成 advantage 标签；
11. 新增并验证 PiperX PiStar train/infer config；
12. 固定评测方案，对 SFT 与 PiStar 做同条件对照。

当前最直接的工作边界是第 1 到第 6 项：**先把高质量专家数据和第一版普通 SFT 跑稳，再进入 DAgger/Value/RECAP。**

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
| Value 训练 | `scripts/train_value.py` |
| Advantage 标注 | `scripts/label_advantage_from_vlm.py` |
