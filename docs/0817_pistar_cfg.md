# PiStar CFG 代码实现与当前 Pipeline 分析

> 分析日期：2026-08-17  
> 根目录版本：`d7b6ebdf41fdfeb368ef157965c1c41b7c056ddd`（`support CFG`）  
> 分析范围：仓库根目录 `src/openpi`、训练/推理入口，以及 `control_your_robot` 实际部署副本。

## 1. 结论先行

PiStar 这里的 CFG 是 **advantage classifier-free guidance**：同一个策略分别预测

1. 带 `Advantage: positive/negative` 的条件速度场 `v_cond`；
2. 去掉 `Advantage`、但仍保留图像、任务和状态信息的基线速度场 `v_uncond`；

然后在每一个 flow-matching 去噪步组合：

```text
v_cfg = v_uncond + beta * (v_cond - v_uncond)
```

根目录最新代码已经实现了双 prompt 数据结构和上面的双分支采样，但 **当前 transform 工厂没有设置 `adv_guidance_input=True`**。因此正常通过 `TrainConfig -> DataConfig -> Policy` 创建策略时，只会产生 `tokenized_prompt`，不会产生 `tokenized_prompt_uncond`。模型看到后令 `use_adv_guidance=False`，静默退化成单分支：

```text
v_used = v_cond
```

所以当前根目录 pipeline 的实际状态是：

- 训练会使用 `adv_ind` 条件，并通过 30% advantage dropout 学习“不带 advantage”的基线分布；
- 推理会使用调用方传入的 `adv_ind`，形成带 advantage 的 prompt；
- 模型端 CFG 公式存在；
- 但是正常 pipeline 没有把无 advantage prompt 送到模型，CFG 组合不执行；
- 默认 `adv_guidance_beta=2.0` 在这条路径上实际不改变动作。

另外，PiperX WebSocket 服务显式导入 `control_your_robot/src/robot/policy/openpi`。该副本甚至还没有根目录提交 `d7b6ebd` 中新增的 CFG 模型代码，因此当前真机 WebSocket 部署路径同样没有动作 CFG，只是 advantage-conditioned PiStar 推理。

## 2. 这里的“conditional / unconditional”具体指什么

这里的 `unconditional` 不是完全无条件。两条分支共享：

- 当前相机图像；
- 任务文本；
- 当前机器人状态（是否进入文本取决于 `discrete_state_input`）；
- 相同的初始动作噪声 `x_t`；
- 相同的 flow-matching 时间 `t`；
- 同一组模型参数。

两者唯一设计差异是 advantage 文本：

```text
# conditional
Task: insert the plug, Advantage: positive;
Action:

# “unconditional”，更准确地说是 dropped-advantage baseline
Task: insert the plug;
Action:
```

如果 `discrete_state_input=True`，prompt 还会包含离散化后的状态；去掉的仍然只有 `Advantage: ...`：

```text
Task: ..., State: ..., Advantage: positive;
Action:

Task: ..., State: ...;
Action:
```

因此本文后续的 `v_uncond` 应理解为“对 advantage 无条件”，不是对观测和任务无条件。

## 3. 关键配置和数据字段

| 名称 | 位置 | 当前默认值 | 作用 |
|---|---|---:|---|
| `pistar` | `Pi0Config` | `False` | 打开 PiStar advantage 输入和 PiStar 加权训练 loss |
| `adv_ind` | dataset / inference request | 无 | 字符串条件，常见值为 `positive`、`negative`；`none` 仍是一个普通字符串条件，不等于删除条件 |
| `adv_ind_input` | `TokenizePrompt` | `False` | 是否从输入读取并要求存在 `adv_ind` |
| `adv_ind_dropout` | `ModelTransformFactory` / `TokenizePrompt` | `True` | tokenize 时以 30% 概率删除 advantage 文本，用于训练无 advantage 基线 |
| `adv_guidance_input` | `TokenizePrompt` | `False` | 是否额外生成固定不含 advantage 的 `tokenized_prompt_uncond` |
| `adv_guidance_beta` | `Pi0Config` | `2.0` | 推理时组合 `v_cond` 和 `v_uncond` 的 guidance scale |
| `tokenized_prompt_uncond(_mask)` | `Observation` | `None` | 触发模型 CFG 分支的实际输入；缺失时模型静默单分支运行 |

最容易混淆的三点：

1. `adv_ind_input=True` 只表示“使用 advantage 条件”，不等于启用 CFG。
2. `adv_ind_dropout=True` 是训练时随机删条件，负责让模型学会基线分支；它也不等于推理时已经执行 CFG。
3. 真正控制推理输入能否进入双分支的是 `adv_guidance_input=True`，因为它负责构造 `tokenized_prompt_uncond`。

## 4. 训练 Pipeline 如何处理 PiStar

### 4.1 数据进入模型前的路径

以 `LeRobotLiberoDataConfig` 为例：

```text
LeRobot frame
  {image, wrist_image, state, actions, task/prompt, adv_ind, ...}
        |
        v
RepackTransform
  保留并重命名 image / wrist_image / state / actions / prompt / adv_ind
        |
        v
LiberoInputs / PiperInputs
  转换图像布局、补齐第三路图像和 mask，并继续透传 adv_ind
        |
        v
Normalize
  归一化 state / actions
        |
        v
TokenizePrompt
  读取 prompt 和 adv_ind，生成 tokenized_prompt / mask
        |
        v
PadStatesAndActions
        |
        v
Observation.from_dict + actions
        |
        v
Pi0.compute_loss(..., train=True)
```

当 `model_config.pistar=True` 时，`ModelTransformFactory` 当前会设置：

```python
TokenizePrompt(
    tokenizer,
    discrete_state_input=model_config.discrete_state_input,
    adv_ind_input=True,
    adv_ind_dropout=data_config.adv_ind_dropout,
    # 当前漏掉 adv_guidance_input
)
```

`adv_ind_input=True` 会执行 `data.pop("adv_ind")`；字段不存在就直接抛出 `ValueError("Adv_ind is required.")`。这保证 PiStar 训练数据必须包含 advantage 标签。

### 4.2 训练时的 advantage dropout

`PaligemmaTokenizer.tokenize()` 内部逻辑可以概括为：

```python
def tokenize(task, state, adv_ind, adv_ind_dropout):
    if adv_ind is not None:
        drop_adv = adv_ind_dropout and random() < 0.3
        if drop_adv:
            prompt = format_without_advantage(task, state)
        else:
            prompt = format_with_advantage(task, state, adv_ind)
    else:
        prompt = format_without_advantage(task, state)

    return sentencepiece_encode_and_pad(prompt)
```

含义是：一个 batch 内约 70% 样本训练条件策略，约 30% 样本训练同一个网络的无 advantage 基线。它不是为每条训练样本同时做 conditional/unconditional 两次 forward，而是每次随机选择其中一种 prompt，再做一次 forward。

训练配置通常保持 `adv_ind_dropout=True`；名字带 `_infer` 的配置一般显式设置 `adv_ind_dropout=False`，保证推理条件 prompt 不会随机丢失 advantage。

注意：如果训练时始终设置 `adv_ind_dropout=False`，模型没有从其他数据见过无 advantage prompt，那么即使推理接通双分支，`v_uncond` 也缺少对应训练基础，CFG 的差分方向未必可靠。

### 4.3 PiStar 的 flow-matching loss

`Pi0.compute_loss()` 先按普通 flow matching 构造：

```text
noise ~ N(0, I)
t ~ Beta(1.5, 1) * 0.999 + 0.001
x_t = t * noise + (1 - t) * action
u_t = noise - action
v_t = model(observation, x_t, t)
```

普通 pi0/pi0.5 的 loss 是全元素 MSE。PiStar 分支则先对 action 维和 horizon 求平均，再按时间加权：

```text
per_timestep_loss = mean_action_dim((v_t - u_t)^2)
per_sample_loss   = mean_horizon(per_timestep_loss)
weight(t)         = 0.5 * exp(-0.5 * (1 - t))
loss              = mean_batch(per_sample_loss * weight(t))
```

`weight(t)` 在 `t` 越接近噪声端 `1` 时越大。这里的 PiStar 特性包含两部分：

- prompt 中的 advantage 条件及其 dropout；
- 上述时间加权 flow-matching loss。

训练阶段本身不应用 `v_uncond + beta * (...)`；CFG 组合是推理算法。

## 5. 根目录代码“预期”如何执行 CFG

### 5.1 Transform 应生成两个 prompt

根目录 `TokenizePrompt` 已实现以下分支：

```python
cond_tokens = tokenizer.tokenize(
    prompt,
    state,
    adv_ind,
    adv_ind_dropout=adv_ind_dropout,
)

if adv_guidance_input and adv_ind is not None:
    uncond_tokens = tokenizer.tokenize(
        prompt,
        state,
        None,
        adv_ind_dropout=False,
    )
```

输出应包含：

```python
{
    "tokenized_prompt": cond_tokens,
    "tokenized_prompt_mask": cond_mask,
    "tokenized_prompt_uncond": uncond_tokens,
    "tokenized_prompt_uncond_mask": uncond_mask,
}
```

`Observation.from_dict()` 检查 uncond token 和 mask 必须成对出现，并将它们保存到 `Observation`。`preprocess_observation()` 在图像预处理后继续原样携带这两个字段。

### 5.2 模型在每个采样步做两次速度预测

`Pi0.sample_actions()` 的意图可以写成：

```python
beta = runtime_beta if runtime_beta is not None else config.adv_guidance_beta
x = initial_noise

cond_cache = encode_prefix(images, cond_prompt)

use_cfg = model.pistar and observation.tokenized_prompt_uncond is not None
if use_cfg:
    uncond_cache = encode_prefix(images, uncond_prompt)

for t in [1.0, 0.9, ..., 0.1]:
    v_cond = denoise(cond_cache, x, t)

    if use_cfg:
        v_uncond = denoise(uncond_cache, x, t)
        v = v_uncond + beta * (v_cond - v_uncond)
    else:
        v = v_cond

    x = x + (-1 / num_steps) * v

return x
```

两个图像/文本 prefix 分别预计算 KV cache；动作 suffix 在每个 Euler 步都要分别对条件和基线 cache 做 forward。因此真正启用 CFG 后，动作去噪主体计算量接近单分支的两倍，此外多一次 prefix 编码和一份 KV cache。

### 5.3 `beta` 的准确语义

| `beta` | 组合结果 | 含义 |
|---:|---|---|
| `0` | `v_cfg = v_uncond` | 完全使用无 advantage 基线 |
| `1` | `v_cfg = v_cond` | 与普通条件推理数学等价，但若仍做双 forward 会浪费计算 |
| `> 1` | 从 `v_uncond` 越过 `v_cond` 外推 | 放大 advantage 条件带来的速度场差异 |
| `< 0` | 朝条件差分的反方向外推 | 通常不是期望用法，当前代码也没有范围校验 |

默认 `beta=2.0`：

```text
v_cfg = 2 * v_cond - v_uncond
```

它不是简单地把最终 action 乘以 2，而是在每一个 flow step 上放大条件与基线速度场的差分，因此动作变化是非线性的。

## 6. “PiStar transform 未设置 `adv_guidance_input=True`”到底是什么意思

根目录 `TokenizePrompt` 定义了：

```python
adv_guidance_input: bool = False
```

但是 `ModelTransformFactory` 构造 PiStar tokenizer transform 时当前只传：

```python
TokenizePrompt(
    PaligemmaTokenizer(...),
    discrete_state_input=model_config.discrete_state_input,
    adv_ind_input=model_config.pistar,
    adv_ind_dropout=self.adv_ind_dropout,
)
```

没有传 `adv_guidance_input=...`，所以 dataclass 使用默认值 `False`。完整的因果链是：

```text
adv_guidance_input=False
  -> TokenizePrompt 不生成 tokenized_prompt_uncond(_mask)
  -> Observation.tokenized_prompt_uncond is None
  -> use_adv_guidance = pistar and False = False
  -> 不计算 v_uncond
  -> 不执行 v_uncond + beta * (v_cond - v_uncond)
  -> beta 不影响结果
```

这不是“CFG 效果比较弱”，而是 **CFG 分支根本没有被执行**。当前模型代码采用静默回退，没有在 `pistar=True && beta!=1 && uncond prompt 缺失` 时抛错，所以仅看服务能正常返回动作，无法证明 CFG 已启用。

还有一个配置契约不一致：`Pi0Config.inputs_spec()` 在 `pistar=True && adv_guidance_beta!=1.0` 时声明 uncond token 是必需输入；但实际 transform 不生成它们。`inputs_spec` 会影响 fake observation / fake dataset 等静态路径，却不会替代实际 transform 构造字段，因此它本身不能激活 CFG，并可能让假数据测试与真实 pipeline 表现不一致。

## 7. 当前各条 Pipeline 实际在做什么

### 7.1 仓库根目录 JAX Pipeline

当前根目录 HEAD 包含 CFG 模型实现，但 transform 未接通：

```text
request adv_ind=positive
  -> conditional prompt 包含 Advantage: positive
  -> 无 uncond prompt
  -> 单分支 flow matching
  -> advantage-conditioned action
```

准确表述应是“PiStar advantage-conditioned policy”，不能据此声称“PiStar CFG 已启用”。

### 7.2 `control_your_robot` 真机 WebSocket Pipeline

服务脚本 `control_your_robot/scripts/serve_piper_single_pi05star_websocket.py` 在开头将以下目录插到 `sys.path` 最前面：

```text
control_your_robot/src/robot/policy/openpi
control_your_robot/src/robot/policy/openpi/src
```

所以它不是导入仓库根目录的 `src/openpi`。该部署副本当前：

- `Pi0Config` 没有 `adv_guidance_beta`；
- `Observation` 没有 uncond prompt 字段；
- `TokenizePrompt` 没有 `adv_guidance_input`；
- `Pi0.sample_actions()` 没有双分支组合。

客户端传 `--adv-ind positive` 后，WebSocket 链路会把 `adv_ind` 送到服务端，服务端 metadata 也会声明 `requires_adv_ind=True`。这只能证明 advantage 条件字段被传输和 tokenize，不能证明执行了 CFG。

另一个 `control_your_robot/src/robot/policy/pistar` 副本也属于旧实现。部署前必须明确唯一源码来源，或者把根目录修复同步到实际被导入的 `policy/openpi`；只修根目录不会改变上述 WebSocket 服务行为。

### 7.3 World Model 的 CFG 不是本问题中的 CFG

`src/openpi/models/pipeline_ctrl_world.py` 和 stable-video-diffusion pipeline 也出现 `classifier_free_guidance`、`guidance_scale`。那一套是在视频扩散 world model 中组合 conditional/unconditional noise prediction：

```text
noise = noise_uncond + scale * (noise_cond - noise_uncond)
```

它由 `max_guidance_scale > 1.0` 控制，与 PiStar 动作策略的 `adv_ind`、`adv_guidance_input`、`adv_guidance_beta` 是两条独立 pipeline，不能用 world model 日志证明 PiStar 动作 CFG 已启用。

## 8. 建议如何接通

### 8.1 最小接线

根目录最小逻辑是让 `ModelTransformFactory` 显式设置：

```python
TokenizePrompt(
    ...,
    adv_ind_input=model_config.pistar,
    adv_ind_dropout=self.adv_ind_dropout,
    adv_guidance_input=(
        model_config.pistar
        and model_config.adv_guidance_beta != 1.0
    ),
)
```

这与当前 `inputs_spec()` 的条件一致。训练 batch 会额外携带 uncond token，但当前 `compute_loss()` 不读取这些字段，不会在训练阶段执行双 forward；代价主要是很小的 token 内存和额外 tokenize。

如果希望严格区分训练/推理，也可以在 `ModelTransformFactory` 或各 `DataConfig` 增加显式 `adv_guidance_input` 参数：训练 config 设 `False`，infer config 设 `True`。这种方式控制更清晰，但必须确保每个 PiStar infer config 都正确设置，漏配仍会静默降级。

### 8.2 建议增加 fail-fast 约束

仅靠可选字段触发容易再次漏接。更稳妥的模型入口逻辑是：

```python
cfg_requested = self.pistar and adv_guidance_beta != 1.0

if cfg_requested and observation.tokenized_prompt_uncond is None:
    raise ValueError(
        "PiStar CFG requested, but tokenized_prompt_uncond is missing"
    )

use_adv_guidance = cfg_requested
```

同时建议约束：

- `tokenized_prompt_uncond` 和 mask 必须同时存在；现有 `Observation.from_dict()` 已检查这一点；
- `beta == 1` 时跳过 uncond forward；
- 如果允许通过 `sample_kwargs` 运行时覆盖 `beta`，transform 是否生成 uncond prompt 不能只依赖 config 的默认 beta；
- 对生产部署记录 `cfg_enabled`、`adv_guidance_beta` 和当前 `adv_ind`，避免只从 config 名字推断。

### 8.3 同步实际部署源码

需要二选一：

1. WebSocket 服务改为导入根目录唯一的 `src/openpi`；
2. 将 CFG 相关的 `model.py`、`pi0_config.py`、`pi0.py`、`transforms.py`、`training/config.py` 和测试完整同步到 `control_your_robot/src/robot/policy/openpi`。

不要只同步 `pi0.py`。Observation schema、input spec、transform 输出和 model 分支必须作为一个接口整体保持一致。

## 9. 推荐验证方法

### 9.1 Transform 单元测试

用 stub tokenizer 避免下载 tokenizer 权重，验证一次调用产生一对不同 prompt：

```python
transform = TokenizePrompt(
    tokenizer=stub,
    adv_ind_input=True,
    adv_ind_dropout=False,
    adv_guidance_input=True,
)

out = transform({"prompt": "insert plug", "adv_ind": "positive"})

assert "tokenized_prompt_uncond" in out
assert "tokenized_prompt_uncond_mask" in out
assert stub.calls == [
    ("insert plug", None, "positive", False),
    ("insert plug", None, None, False),
]
```

还应验证 `ModelTransformFactory(Pi0Config(pistar=True, adv_guidance_beta=2))` 创建出的 `TokenizePrompt.adv_guidance_input is True`，防止 transform 自身测试通过但工厂再次漏接。

### 9.2 模型组合公式测试

将 conditional/unconditional denoise 输出替换成可控常量：

```text
v_cond   = 3
v_uncond = 1

beta=0 -> v=1
beta=1 -> v=3
beta=2 -> v=5
```

同时验证缺少 uncond prompt 且 `pistar=True, beta=2` 时明确报错，而不是静默单分支。

### 9.3 端到端离线验证

固定同一个 checkpoint、observation 和初始 noise，分别运行：

```text
beta=0
beta=1
beta=2
```

记录：

```text
cfg_enabled
adv_ind
beta
||action_beta0 - action_beta1||_2
||action_beta1 - action_beta2||_2
max_abs_delta
```

预期：transform 输出含 uncond 字段；日志确认每步执行两次 denoise；固定噪声时三组动作可重复，且通常不应完全相等。仅观察到“动作发生变化”还不够，需要同时证明变化来自 CFG 分支，而不是随机 noise。

### 9.4 真机边界

离线测试只能证明代码路径和数值组合正确，不能证明 `beta=2` 对真机安全或任务成功率更高。真机应从低 guidance、低速、短 horizon 和可急停环境开始，并分别报告：

- 成功率；
- 关节/夹爪动作范围；
- 相邻 action jump；
- 控制频率和 CFG 带来的推理延迟；
- `positive` 与 `negative` 条件的差异。

## 10. 源码证据索引

| 主题 | 文件与关键位置 |
|---|---|
| CFG 配置默认值和 input spec | `src/openpi/models/pi0_config.py:32-42, 80-86` |
| uncond Observation 字段 | `src/openpi/models/model.py:99-106, 118-136, 210-220` |
| advantage prompt 和 30% dropout | `src/openpi/models/tokenizer.py:22-70` |
| `adv_guidance_input` 默认值及 uncond tokenize | `src/openpi/transforms.py:247-290` |
| 工厂漏传 `adv_guidance_input` | `src/openpi/training/config.py:108-144` |
| PiStar 数据 repack 和 transform 构造 | `src/openpi/training/config.py:340-431, 435-530` |
| PiStar 加权 flow-matching loss | `src/openpi/models/pi0.py:190-228` |
| CFG 双 cache、双 denoise 和组合公式 | `src/openpi/models/pi0.py:230-315` |
| Policy 推理变换链 | `src/openpi/policies/policy_config.py:16-94`、`src/openpi/policies/policy.py:67-106` |
| Piper/Libero 对 `adv_ind` 的透传 | `src/openpi/policies/piper_policy.py:70-86`、`src/openpi/policies/libero_policy.py:72-86` |
| 真机服务固定导入 vendored OpenPI | `control_your_robot/scripts/serve_piper_single_pi05star_websocket.py:7-15` |
| 客户端注入 `adv_ind` | `control_your_robot/example/deploy/piper_single_on_PI0_websocket.py:76-103, 127-168` |
| CFG 引入提交 | Git commit `d7b6ebdf41fdfeb368ef157965c1c41b7c056ddd` |

## 11. 一句话回答

“PiStar transform 未设置 `adv_guidance_input=True`”的意思是：**模型虽已写好 CFG 双分支公式，但输入变换没有构造无 advantage prompt，导致模型判断 CFG 输入不存在并退化为普通的 advantage-conditioned 单分支采样；当前根目录 pipeline 和实际 PiperX vendored 部署 pipeline 都没有真正执行动作 CFG。**
