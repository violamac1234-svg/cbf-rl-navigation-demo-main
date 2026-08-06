# Algorithm 1 与单积分器复现源码逐行对照

本文档对照论文 **Algorithm 1: RL Training with Discrete-Time CBF Safety**（截图中的第 1--17 行）与本仓库当前实际执行路径。行号按 2026-08-06 工作区版本记录；训练解释器为 `D:\anaconda3\envs\cbf_learning\python.exe`。

## 先看实际调用链

`train.py:138` → `train()` → `OnPolicyRunner.learn()` → `PPO.act()` → `UnifiedNavigationEnv.step()` → `PPO.process_env_step()` → `PPO.update()`。

仓库负责环境、CBF、安全奖励和训练配置；策略采样、rollout 存储及 PPO 更新来自解释器环境中安装的 `rsl_rl`。

## 伪代码逐行映射

| 伪代码行 | 含义 | 当前源码位置 | 对应程度与说明 |
|---:|---|---|---|
| 1 | 初始化策略参数 \(\theta\)、初始构型 \(q_0\)、安全函数 \(h\) | `train.py:102-107` 创建 `OnPolicyRunner`；`rsl_rl/runners/on_policy_runner.py:67-96` 创建 Actor-Critic 与 PPO；`nav_env/unified_navigation_env.py:179` 调用 `reset()`；`nav_env/unified_navigation_env.py:535-575` 定义 `h_function()` | 分散在 runner 和环境中完成。CBF 参数在 `config.py:31-36`，经 `train.py:74-83` 传入环境。 |
| 2 | 外层训练循环 `step = 1...N_steps` | `train.py:125-127` 调用 `runner.learn(max_iterations)`；`rsl_rl/runners/on_policy_runner.py:193-200` 外层 iteration 与 rollout 循环 | 论文的“step”在实现中分成 PPO iteration 和每环境 rollout step 两层。当前为 `1500 × 48 × 4096` 个并行环境步，而不是只有 1500 个环境步，配置见 `config.py:23,66-67`。 |
| 3 | 初始化 \(q_0\)、观测 \(o_0\) | `nav_env/unified_navigation_env.py:185-434` 的 `reset()`；`nav_env/unified_navigation_env.py:436-481` 构造观测字典；`nav_env/unified_navigation_env.py:506-533` 展平观测；`rsl_rl/runners/on_policy_runner.py:166-170` 首次取观测 | 环境构造时先全量 reset；episode 结束后在 `nav_env/unified_navigation_env.py:894-904` 局部 reset。不是每个 PPO iteration 都重新初始化。 |
| 4 | 回合内循环 \(k=0...T-1\) | `rsl_rl/runners/on_policy_runner.py:199-217` | `num_steps_per_env=48` 是每次 PPO 更新前的 rollout 长度；单个 episode 最长 1000 步，见 `config.py:27,67`。两者不要混为同一个 \(T\)。 |
| 5 | \(v_k^{policy}\leftarrow\pi_\theta(o_k)\) | `rsl_rl/runners/on_policy_runner.py:201-204` 调 `self.alg.act(...)`；`rsl_rl/algorithms/ppo.py:136-148` 中 `self.policy.act(obs)` | `PPO.act()` 返回随机策略采样动作，并同时保存 value、log-prob、均值和方差。进入环境后先在 `nav_env/unified_navigation_env.py:689` 裁剪到速度范围。 |
| 6 | \(q_{k+1}^{policy}=q_k+\Delta t\,v_k^{policy}\) | 没有独立变量；最接近的是 `nav_env/unified_navigation_env.py:689` 的 `clipped_action` 与 `706-707` 的积分式 | **未按伪代码显式实现。** 当前过滤器不需要先构造 `q_policy_next`，而是在 \(q_k\) 处直接检查 \(\nabla h(q_k)^T v_k^{policy}+\alpha h(q_k)\)。若不启用过滤，`707` 才等价执行名义下一状态（另含可选扰动）。 |
| 7 | \(a_k=\nabla h(q_k),\ b_k=-\alpha h(q_k)\) | `nav_env/unified_navigation_env.py:535-575` 计算 \(h\)；`577-632` 计算活动约束梯度；`648-650` 调用二者 | 代码未显式命名 `a_k/b_k`：`grad_h` 即 \(a_k\)，`-self.cbf_alpha*h` 即 \(b_k\)。`h` 取所有圆障碍和四面墙中最小的净安全距离。 |
| 8 | 计算 CBF 条件 \(c=a_k^T v_k^{policy}-b_k\) | `nav_env/unified_navigation_env.py:652-655` | 变量名为 `psi`，即 `grad_h·v_policy + alpha*h`，与 \(a^Tv-b\) 完全等价。 |
| 9 | 若 \(c\ge0\)，令 \(v_k^{safe}=v_k^{policy}\) | `nav_env/unified_navigation_env.py:659-660,667` | `filtered_velocity` 先克隆名义速度，只筛选 `psi < 0` 的环境，因此其余环境保持不变。 |
| 10 | 否则应用闭式投影 | `nav_env/unified_navigation_env.py:660-665` | `correction=(-psi/||grad_h||²)grad_h` 后相加，等价于论文 \(v^{policy}+[(b-a^Tv^{policy})/\|a\|^2]a\)。分母用 `clamp_min(1e-12)` 做数值保护。 |
| 11 | \(q_{k+1}^{safe}=q_k+\Delta t\,v_k^{safe}\) | `nav_env/unified_navigation_env.py:697-707` | Filter Only/Dual 将 `_last_velocity` 设为 `filtered_action`，随后积分。Nominal/Reward Only 执行 `clipped_action`。DR 时实际式为 \(q_{k+1}=q_k+\Delta t(v+\epsilon)\)，扰动见 `704-707`。 |
| 12 | 环境更新并得到 \(q_{k+1}^{env},o_{k+1}\) | `nav_env/unified_navigation_env.py:710-771` 更新位置、碰撞、目标与终止；`844-846`、`901-904` 生成下一观测 | 环境更新还包括边界裁剪、终止判断和结束环境的自动 reset，所以返回的 `obs` 对 done 环境是 reset 后观测。 |
| 13 | 计算 CBF 奖励与任务奖励 | `nav_env/unified_navigation_env.py:773-820` | `reward_psi`（`805-806`）对应 \(w\min(c,0)\)；`reward_clipped_action`（`800-804`）对应 \(w[\exp(-\|v^{policy}-v^{safe}\|^2/\sigma^2)-1]\)；任务奖励 \(R\) 为 goal、碰撞、progress、alive 项（`773-794`）。参数见 `config.py:34-36`。 |
| 14 | 存储 transition | `rsl_rl/runners/on_policy_runner.py:216-217` 调 `process_env_step()`；`rsl_rl/algorithms/ppo.py:150-177` 写 reward/done 并 `storage.add_transitions(...)` | **不是按论文元组逐字段保存。** PPO storage 保存 observation、privileged observation、sampled action、value、log-prob、distribution 参数、reward、done 等；`q`、`q_policy_next`、`q_safe_next` 没有作为独立字段存储。环境诊断量则放在 `extras["log"]`，见 `nav_env/unified_navigation_env.py:848-868`。 |
| 15 | 回合内循环结束 | `rsl_rl/runners/on_policy_runner.py:200-250` 的 rollout 循环结束 | 每收集 48 步/环境后进入 return 计算和 PPO 更新；episode 可跨越多个 rollout。 |
| 16 | 更新策略参数 \(\theta\) | `rsl_rl/runners/on_policy_runner.py:255-260`；`rsl_rl/algorithms/ppo.py:179-205` 计算 returns/构造 minibatch；`305-324` PPO policy/value loss；`370-386` 反传、裁剪梯度、Adam step | 论文用抽象的策略梯度表示；当前是 PPO clipped surrogate，并非一次简单的 \(\theta\leftarrow\theta+\eta\nabla L\)。超参数在 `config.py:49-60`。 |
| 17 | 外层训练循环结束 | `rsl_rl/runners/on_policy_runner.py:193-270`；仓库入口 `train.py:121-135` | 每个 iteration 完成 rollout、GAE/returns、10 epochs × 8 minibatches 的 PPO 更新，并按周期保存/记录。 |

## 关键变量名对应

| 论文符号 | 代码变量 |
|---|---|
| \(q_k\) | `self._robot_pos` |
| \(o_k\) | `obs`；原始分量在 `obs_dict` |
| \(v_k^{policy}\) | runner 中的 `actions`；环境中裁剪后为 `clipped_action` |
| \(h(q_k)\) | `h` |
| \(a_k=\nabla h(q_k)\) | `grad_h` |
| \(b_k=-\alpha h(q_k)\) | 未单独存储 |
| \(c\) | `psi` |
| \(v_k^{safe}\) | `filtered_action` / `filtered_velocity` |
| \(q_{k+1}^{env}\) | 更新后的 `self._robot_pos` |
| \(r^c\) | `reward_psi + reward_clipped_action` |
| \(r\) | `reward` |

## 四种训练方法如何切换 Algorithm 1

`experiment_configs.py:27-33` 定义两开关的 2×2 消融；`train.py:77-83` 将开关传给环境。

- `nominal`: 不把 CBF 奖励加入总奖励，也不执行安全动作。
- `reward_only`: 计算并加入 CBF 奖励，但实际执行名义动作。
- `filter_only`: 执行安全动作，但不把 CBF 奖励加入总奖励。
- `dual`: 同时执行安全动作并加入 CBF 奖励，是 Algorithm 1 最完整的对应实现。

## 复现时必须注意的实现差异

1. 论文伪代码第 6 行的预测状态没有显式保存；当前实现使用连续时间形式的 CBF 条件在离散仿真步上过滤。
2. 策略动作先被速度裁剪，再进入 CBF 过滤器。过滤后的动作没有再次速度裁剪，因此闭式投影可能产生超过 `max_velocity` 的分量；这是当前实现行为，不应在报告中写成论文明确规定。
3. DR 扰动在每个环境、每个时间步、每个速度维度独立采样，代码为 `torch.randn_like(...) * max_velocity * noise_level`。
4. `reward_log` 在 `nav_env/unified_navigation_env.py:822-830` 刻意没有加入 `reward_psi`，而真正传给 PPO 的 `reward` 在 `812-820` 包含它。画图时若使用 `reward_log`，曲线不等于 PPO 实际优化的总奖励。
5. 当前 `h_function()` 先取单个最危险约束，再做单约束闭式投影；这不等于同时求解包含全部障碍物和墙约束的多约束 QP。

## 用指定解释器验证

```powershell
& 'D:\anaconda3\envs\cbf_learning\python.exe' -m pytest tests\test_cbf_navigation.py -q
```

其中 `tests/test_cbf_navigation.py:36-47` 直接验证过滤后的动作满足活动 CBF 约束，可作为伪代码第 8--10 行的最小单元测试。
