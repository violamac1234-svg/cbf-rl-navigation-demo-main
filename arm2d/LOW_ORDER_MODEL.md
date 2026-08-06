# 二维二连杆机械臂低阶 CBF-RL 实现说明

本文档说明仓库中**已经训练完成的低阶机械臂版本**。内容以当前源码为准，
并按“机械臂模型—运动学—任务—观测—CBF—过滤器—奖励—训练—评价”的顺序组织。

当前版本的核心结构是：

```text
PPO Actor 输出名义关节速度 u_policy
                    ↓
          构造多约束 CBF-QP
                    ↓
      得到安全关节速度 u_safe
                    ↓
     q(k+1) = q(k) + dt * u_executed
```

它仍然是低阶运动学模型，不包含机械臂质量矩阵、科氏力、重力、关节力矩或
HOCBF。但与只保护末端的最简版本不同，当前版本已经加入两根连杆的离散碰撞点、
关节角约束、速度约束和多约束 QP。

主环境实现：[vec_env.py](./vec_env.py)  
训练入口：[train_arm2d.py](../train_arm2d.py)  
四种消融配置：[experiment_configs.py](../experiment_configs.py)

## 1. 机械臂模型

机械臂为二维平面二连杆机构，状态为两个关节角：

\[
q=\begin{bmatrix}q_1\\q_2\end{bmatrix}.
\]

策略和安全过滤器的控制量都是关节速度：

\[
u=\dot q=\begin{bmatrix}\dot q_1\\\dot q_2\end{bmatrix}.
\]

采用理想关节速度模型：

\[
\dot q=u.
\]

离散更新为：

\[
q_{k+1}=q_k+\Delta t\,u_k,
\qquad \Delta t=0.025\ \mathrm{s}.
\]

两根连杆长度为：

\[
l_1=1.0\ \mathrm{m},\qquad l_2=0.8\ \mathrm{m}.
\]

关节角范围和关节速度范围为：

\[
-2.85\le q_i\le2.85\ \mathrm{rad},
\qquad
-1.6\le\dot q_i\le1.6\ \mathrm{rad/s}.
\]

对应代码：

- 环境时间步和回合长度：[vec_env.py L17-L31](./vec_env.py#L17-L31)
- 连杆、关节和速度参数：[vec_env.py L36-L46](./vec_env.py#L36-L46)
- 离散状态更新：[vec_env.py L236-L245](./vec_env.py#L236-L245)

## 2. 正运动学

肘关节位置为：

\[
p_1(q)=
\begin{bmatrix}
l_1\cos q_1\\
l_1\sin q_1
\end{bmatrix}.
\]

末端位置为：

\[
p_e(q)=p_1(q)+
\begin{bmatrix}
l_2\cos(q_1+q_2)\\
l_2\sin(q_1+q_2)
\end{bmatrix}.
\]

对应代码：[vec_env.py L65-L70](./vec_env.py#L65-L70)。

## 3. 连杆碰撞点与雅可比

当前版本不只保护末端，而是在每根连杆上设置 5 个碰撞采样点：

\[
s\in\{0.2,0.4,0.6,0.8,1.0\}.
\]

两根连杆共得到 10 个采样点。对每个采样点 \(p_i(q)\) 同时解析计算：

\[
J_i(q)=\frac{\partial p_i(q)}{\partial q}\in\mathbb R^{2\times2}.
\]

这些雅可比用于把笛卡尔空间的障碍物距离梯度转换到关节空间。

对应代码：

- 采样比例：[vec_env.py L46](./vec_env.py#L46)
- 两根连杆采样点和雅可比：[vec_env.py L72-L92](./vec_env.py#L72-L92)

## 4. 任务与场景生成

每个环境包含：

- 一个二维二连杆机械臂；
- 一个圆形障碍物；
- 一个末端目标点；
- 一个随机安全初始构型。

场景不是固定场景，而是在每个 episode 重置时随机生成：

### 4.1 初始关节角

\[
q_{1,0}\in[-1.15,-0.45],\qquad
q_{2,0}\in[0.45,1.45].
\]

### 4.2 目标构型

先采样一个目标关节构型：

\[
q_{1,g}\in[0.25,1.20],\qquad
q_{2,g}\in[-1.35,-0.45],
\]

再通过正运动学得到末端目标位置 \(p_g=p_e(q_g)\)。策略只需要末端到达目标，
不要求最终关节角等于采样的 \(q_g\)。

### 4.3 障碍物

障碍物中心被放置在初始末端到目标点连线的 40%～60% 位置附近，并加入垂直方向
随机偏移。障碍物半径随机为：

\[
r_o\in[0.13,0.18]\ \mathrm{m}.
\]

重置逻辑会拒绝以下场景：

- 起点和目标距离不足 0.65 m；
- 初始构型的最小 CBF 裕度不大于 0.10 m；
- 目标构型的最小 CBF 裕度不大于 0.08 m。

对应代码：[vec_env.py L166-L222](./vec_env.py#L166-L222)。

## 5. PPO 观测量

当前 Actor 观测是 11 维：

\[
o_k=
\begin{bmatrix}
\sin q_k\\
\cos q_k\\
u_{k-1}/u_{\max}\\
(p_g-p_e)/1.8\\
(p_o-p_e)/1.8\\
r_o/0.25
\end{bmatrix}.
\]

展开后的维数为：

| 内容 | 维数 |
|---|---:|
| \(\sin q\) | 2 |
| \(\cos q\) | 2 |
| 上一步实际执行速度 | 2 |
| 目标相对末端位置 | 2 |
| 障碍物相对末端位置 | 2 |
| 障碍物半径 | 1 |
| 合计 | 11 |

使用 \(\sin q,\cos q\) 而不是直接输入关节角，可以避免角度周期边界处的观测跳变。
当前观测没有直接包含绝对末端坐标，但策略可以通过关节角编码推断机械臂构型。

对应代码：[vec_env.py L149-L164](./vec_env.py#L149-L164)。

## 6. Actor 输出与名义动作

Actor 输出两维高斯策略动作：

\[
u_k^{\mathrm{policy}}
=\begin{bmatrix}\dot q_{1,k}^{\mathrm{policy}}\\\dot q_{2,k}^{\mathrm{policy}}\end{bmatrix}.
\]

进入 CBF 之前，环境先把动作裁剪到：

\[
-1.6\le u_{i,k}^{\mathrm{policy}}\le1.6.
\]

需要注意：当前 CBF 奖励比较的是**裁剪后的名义动作**与安全动作，策略原始输出超过
速度范围的部分不会单独受到 CBF 修正量惩罚。

对应代码：

- Actor 输出维数：[vec_env.py L27-L29](./vec_env.py#L27-L29)
- 动作预裁剪：[vec_env.py L236-L241](./vec_env.py#L236-L241)
- Actor-Critic 网络：[train_arm2d.py L37-L43](../train_arm2d.py#L37-L43)

## 7. 障碍物安全函数

连杆等效半径和额外安全裕度分别为：

\[
r_{\mathrm{link}}=0.055\ \mathrm{m},\qquad
d_{\mathrm{margin}}=0.025\ \mathrm{m}.
\]

对第 \(i\) 个连杆采样点，安全函数定义为净距离：

\[
h_i(q)=
\|p_i(q)-p_o\|
-\left(r_o+r_{\mathrm{link}}+d_{\mathrm{margin}}\right).
\]

含义为：

- \(h_i>0\)：位于 CBF 安全边界之外；
- \(h_i=0\)：位于 CBF 安全边界；
- \(h_i<0\)：进入安全裕度区域。

当前使用净距离而不是距离平方，因此 \(h\) 的单位是米，绘图中的最小安全裕度也可以
直接解释为距离。

对应代码：[vec_env.py L94-L101](./vec_env.py#L94-L101)。

## 8. 安全函数梯度

定义：

\[
\Delta p_i=p_i(q)-p_o,
\qquad
n_i=\frac{\Delta p_i}{\|\Delta p_i\|}.
\]

安全函数对关节角的梯度为：

\[
\nabla_q h_i(q)=n_i^\top J_i(q).
\]

程序将距离下限裁剪到 \(10^{-8}\)，避免采样点与障碍物中心重合时除零。

对应代码：[vec_env.py L94-L101](./vec_env.py#L94-L101)。

## 9. 一阶 CBF 约束

低阶模型满足 \(\dot q=u\)，因此：

\[
\dot h_i=\nabla_q h_i(q)^\top u.
\]

使用一阶 CBF 条件：

\[
\dot h_i+\alpha h_i\ge0,
\qquad \alpha=6.
\]

可以写成半空间约束：

\[
a_i(q)^\top u\ge b_i(q),
\]

其中：

\[
a_i=\nabla_qh_i,
\qquad
b_i=-\alpha h_i.
\]

每个环境包含 10 条障碍物 CBF 约束。

对应代码：[vec_env.py L94-L110](./vec_env.py#L94-L110)。

## 10. 关节限位和速度限位

### 10.1 关节角 CBF

每个关节包含上下限两个安全函数：

\[
h_{i,\min}=q_i-q_{i,\min},
\qquad
h_{i,\max}=q_{i,\max}-q_i.
\]

因此两个关节共增加 4 条 CBF 约束。

### 10.2 速度半空间

速度限制也被写入 QP：

\[
u_i\ge-u_{i,\max},
\qquad
-u_i\ge-u_{i,\max}.
\]

两个关节共增加 4 条速度约束。

最终 QP 一共有：

\[
10\ \text{条障碍物 CBF}
+4\ \text{条关节角 CBF}
+4\ \text{条速度限制}
=18\ \text{条约束}.
\]

对应代码：[vec_env.py L103-L114](./vec_env.py#L103-L114)。

## 11. 多约束 CBF-QP 安全过滤器

当前版本没有使用单约束闭式公式，而是求解：

\[
u_k^{\mathrm{safe}}
=\arg\min_u\frac12\|u-u_k^{\mathrm{policy}}\|^2
\]

满足：

\[
A(q_k)u\ge b(q_k).
\]

因为优化变量只有二维，程序不调用外部 QP 库，而是枚举二维凸投影的全部可能最优点：

1. 名义动作本身，共 1 个；
2. 名义动作到每条边界的正交投影，共 18 个；
3. 任意两条边界的交点，共 \(\binom{18}{2}=153\) 个。

总候选点数为：

\[
1+18+153=172.
\]

程序批量检查每个候选点是否满足全部约束，并选取可行候选中距离名义动作最近的点。
这对于二维控制量是精确的欧氏投影解，而不是逐条约束依次修正。

如果没有找到可行候选点，当前回退动作为零速度，并把 `arm/qp_feasible` 记录为 0。
当前 QP 没有松弛变量。

对应代码：[vec_env.py L116-L147](./vec_env.py#L116-L147)。

## 12. 四种方法中的动作执行

无论当前方法是否真正执行过滤动作，环境每一步都会计算：

- 裁剪后的名义动作 \(u_{\mathrm{policy}}\)；
- 安全动作 \(u_{\mathrm{safe}}\)；
- 最小 CBF 条件值；
- 动作修正量。

实际执行动作取决于训练方法：

\[
u_{\mathrm{executed}}=
\begin{cases}
u_{\mathrm{safe}},&\text{Filter Only 或 Dual},\\
u_{\mathrm{policy}},&\text{Nominal 或 Reward Only}.
\end{cases}
\]

对应代码：[vec_env.py L236-L245](./vec_env.py#L236-L245)。

## 13. 状态更新

实际执行速度通过显式欧拉法更新关节角：

\[
q_{k+1}
=\operatorname{clip}
\left(q_k+\Delta t\,u_{\mathrm{executed}},q_{\min},q_{\max}\right).
\]

外层 `clip` 是最终数值保护；正常情况下，启用过滤器的方法已经通过 QP 的关节限位
CBF 阻止状态越界。

对应代码：[vec_env.py L236-L245](./vec_env.py#L236-L245)。

## 14. 任务奖励

首先定义归一化进度：

\[
r_{\mathrm{progress}}
=\frac{
\|p_{e,k}-p_g\|-\|p_{e,k+1}-p_g\|
}{\Delta t\cdot1.8\cdot u_{\max}}.
\]

任务奖励为：

\[
r_{\mathrm{task}}
=4r_{\mathrm{progress}}
-0.005
+12\,\mathbb I_{\mathrm{success}}
-12\,\mathbb I_{\mathrm{collision}}.
\]

其中 `-0.005` 是每步时间代价，用于鼓励尽快到达目标。

当前版本**没有显式动作平方惩罚** \(-\lambda_u\|u\|^2\)。

对应代码：[vec_env.py L247-L267](./vec_env.py#L247-L267)。

## 15. CBF 安全奖励

过滤前第 \(i\) 条 CBF 条件值为：

\[
\psi_i=a_i^\top u_{\mathrm{policy}}-b_i
=\nabla h_i^\top u_{\mathrm{policy}}+\alpha h_i.
\]

程序在 10 条障碍物 CBF 和 4 条关节角 CBF 中取最小值：

\[
\psi_{\min}=\min_{i=1,\ldots,14}\psi_i.
\]

安全奖励为：

\[
r_{\mathrm{CBF}}
=w_{\mathrm{CBF}}
\left[
\min(\psi_{\min},0)
+\exp\left(
-\frac{\|u_{\mathrm{policy}}-u_{\mathrm{safe}}\|^2}{\sigma^2}
\right)-1
\right],
\]

当前参数：

\[
w_{\mathrm{CBF}}=5,\qquad\sigma=0.5.
\]

速度半空间不参与 \(\psi_{\min}\)，但参与 QP 投影。CBF 奖励只在 Reward Only 和
Dual 中加入总奖励。

对应代码：

- 最小 CBF 条件：[vec_env.py L116-L124](./vec_env.py#L116-L124)
- CBF 奖励：[vec_env.py L257-L267](./vec_env.py#L257-L267)

## 16. 终止条件与碰撞判定

episode 在以下条件之一满足时终止：

### 16.1 到达目标

\[
\|p_e-p_g\|<0.075\ \mathrm{m}.
\]

### 16.2 发生物理碰撞

CBF 安全函数包含额外安全裕度：

\[
h_{\min}=d-(r_o+r_{\mathrm{link}}+d_{\mathrm{margin}}).
\]

但物理碰撞判定不包含额外裕度：

\[
d-(r_o+r_{\mathrm{link}})<0.
\]

代码中等价写成：

\[
h_{\min}+d_{\mathrm{margin}}<0.
\]

因此进入 0.025 m 的安全缓冲区会使 CBF 裕度为负，但只有进一步接触连杆物理半径
才被统计为碰撞。

### 16.3 超时

\[
T\ge320\ \text{步}.
\]

关节角越界不会单独终止 episode，因为状态更新最终会裁剪到允许范围。

对应代码：[vec_env.py L247-L255](./vec_env.py#L247-L255)。

## 17. 完整训练流程

每个环境步执行：

```text
读取 q、上一时刻速度、目标和障碍物
                    ↓
构造 11 维 PPO 观测
                    ↓
Actor 输出两维名义关节速度
                    ↓
名义动作裁剪到速度范围
                    ↓
计算 10 个连杆点的位置和雅可比
                    ↓
构造 14 条 CBF + 4 条速度约束
                    ↓
枚举求解二维多约束 QP
                    ↓
根据方法选择 nominal 或 safe 动作执行
                    ↓
显式欧拉更新关节角
                    ↓
计算任务奖励、CBF 奖励和终止条件
                    ↓
向 PPO 返回 transition
```

环境单步实现：[vec_env.py L236-L286](./vec_env.py#L236-L286)。

## 18. 四种训练配置

当前训练了以下四种方法：

| 方法 | 训练时执行 CBF-filter | 加入 CBF 奖励 |
|---|---:|---:|
| Nominal | 否 | 否 |
| Reward Only | 否 | 是 |
| Filter Only | 是 | 否 |
| Dual | 是 | 是 |

配置定义：[experiment_configs.py L26-L31](../experiment_configs.py#L26-L31)。

测试绘图还会比较：

- Dual with runtime filter；
- Dual without runtime filter；
- Filter Only with runtime filter；
- Filter Only without runtime filter。

对应代码：[plot_arm2d_fig4_no_dr.py L19-L35](../plot_arm2d_fig4_no_dr.py#L19-L35)。

## 19. PPO 训练配置

默认训练参数为：

| 参数 | 当前值 |
|---|---:|
| 并行环境数 | 1024 |
| PPO 迭代数 | 1000 |
| 每环境 rollout 步数 | 32 |
| Actor 隐藏层 | 64, 64 |
| Critic 隐藏层 | 64, 64 |
| 激活函数 | ELU |
| 初始动作标准差 | 0.8 |
| 学习率 | \(3\times10^{-4}\) |
| 折扣因子 \(\gamma\) | 0.99 |
| GAE \(\lambda\) | 0.95 |
| PPO clip | 0.2 |
| 每次更新 epoch | 5 |
| mini-batch 数 | 8 |
| entropy coefficient | 0.003 |

对应代码：

- 命令行训练规模：[train_arm2d.py L20-L30](../train_arm2d.py#L20-L30)
- PPO 和网络参数：[train_arm2d.py L33-L63](../train_arm2d.py#L33-L63)
- 环境及 runner 初始化：[train_arm2d.py L74-L94](../train_arm2d.py#L74-L94)

## 20. 当前记录的评价量

训练环境每一步记录：

| TensorBoard 标签 | 含义 |
|---|---|
| `arm/success` | 当前步是否到达目标 |
| `arm/collision` | 当前步是否发生物理碰撞 |
| `arm/distance_to_goal` | 末端到目标距离 |
| `arm/min_safety_margin` | 10 个采样点中的最小 CBF 裕度 |
| `arm/filter_activated` | 名义动作和安全动作是否不同 |
| `arm/action_correction` | \(\|u_{\mathrm{policy}}-u_{\mathrm{safe}}\|\) |
| `arm/cbf_reward` | 当前步 CBF 奖励 |
| `arm/qp_feasible` | QP 是否找到可行候选点 |

对应代码：[vec_env.py L269-L281](./vec_env.py#L269-L281)。

需要特别注意：`arm/success` 和 `arm/collision` 是训练期间的**逐步指标**，不是独立测试
得到的 episode 成功率和碰撞率。当前轨迹脚本会记录单个场景的成功、碰撞和最小裕度，
但仓库目前还没有为机械臂实现 500/1000 episode 的完整批量统计表。

训练曲线脚本：[plot_arm2d_paper_fig3.py](../plot_arm2d_paper_fig3.py)  
无 DR 轨迹脚本：[plot_arm2d_fig4_no_dr.py](../plot_arm2d_fig4_no_dr.py)

## 21. 当前参数总表

| 参数 | 当前值 | 源码 |
|---|---:|---|
| \(l_1,l_2\) | 1.0, 0.8 m | [vec_env.py L36](./vec_env.py#L36) |
| \(\Delta t\) | 0.025 s | [vec_env.py L23-L31](./vec_env.py#L23-L31) |
| 最大步数 | 320 | [vec_env.py L23-L30](./vec_env.py#L23-L30) |
| 关节角范围 | ±2.85 rad | [vec_env.py L40-L41](./vec_env.py#L40-L41) |
| 最大关节速度 | 1.6 rad/s | [vec_env.py L42](./vec_env.py#L42) |
| 连杆等效半径 | 0.055 m | [vec_env.py L37](./vec_env.py#L37) |
| 安全裕度 | 0.025 m | [vec_env.py L38](./vec_env.py#L38) |
| 障碍物半径 | 0.13～0.18 m | [vec_env.py L194](./vec_env.py#L194) |
| 目标阈值 | 0.075 m | [vec_env.py L43](./vec_env.py#L43) |
| \(\alpha\) | 6 | [vec_env.py L39](./vec_env.py#L39) |
| \(\sigma\) | 0.5 | [vec_env.py L45](./vec_env.py#L45) |
| CBF 奖励权重 | 5 | [vec_env.py L44](./vec_env.py#L44) |
| 成功奖励 | 12 | [vec_env.py L267](./vec_env.py#L267) |
| 碰撞惩罚 | -12 | [vec_env.py L267](./vec_env.py#L267) |
| 每步时间代价 | -0.005 | [vec_env.py L267](./vec_env.py#L267) |
| 连杆采样点数 | 每根 5 个 | [vec_env.py L46](./vec_env.py#L46) |
| QP 总约束数 | 18 | [vec_env.py L48-L50](./vec_env.py#L48-L50) |
| PPO 网络 | 64,64 | [train_arm2d.py L37-L43](../train_arm2d.py#L37-L43) |

## 22. 与“只保护末端”的最简版本的关系

两者相同之处：

- 都使用二维二连杆；
- 都使用理想关节速度输入；
- 都使用显式欧拉更新；
- 都使用一阶 CBF；
- 都使用 PPO 输出名义动作；
- 都训练 Nominal、Reward Only、Filter Only、Dual 四种方法；
- 都没有完整动力学和 HOCBF。

当前实现额外加入：

- 两根连杆各 5 个碰撞点，而不是只保护末端；
- 10 条障碍物 CBF，而不是单一末端 CBF；
- 4 条关节限位 CBF；
- 4 条速度半空间；
- 多约束二维 QP，而不是单约束闭式修正；
- 随机起点、目标、障碍物中心和障碍物半径；
- 11 维周期角度观测，而不是 8 维原始关节角观测。

因此当前版本仍是“低阶模型”，但不是最简末端避障版本，而是一个**采样连杆安全的增强低阶版本**。

## 23. 局限性

当前版本能够验证：

- 多连杆采样点 CBF 是否能修正 PPO 关节速度；
- CBF 奖励是否减少策略对运行时过滤器的依赖；
- Filter Only 和 Dual 是否能在训练期间减少物理碰撞；
- 多约束二维 QP 是否能实时批量运行。

当前版本不能保证：

1. **连续整条连杆绝对安全。** 每根连杆只有 5 个点，采样点之间仍可能漏检；
2. **离散采样间绝对安全。** CBF 是连续时间条件，但环境以 0.025 s 离散更新；
3. **完整动力学安全。** 没有 \(M(q),C(q,\dot q),g(q)\) 和力矩限制；
4. **真实速度跟踪。** 仿真假设命令速度可以瞬时、精确执行；
5. **QP 永远可行。** 当前没有松弛变量，不可行时回退为零速度；
6. **多障碍物泛化。** 当前训练环境只有一个障碍物；
7. **DR 鲁棒性。** 当前这四个机械臂 checkpoint 没有使用 domain randomization；
8. **统计意义上的最终性能。** 还需要统一的多 seed、多 episode 独立评估脚本。

## 24. 源码导航

| 功能 | 代码位置 |
|---|---|
| 环境参数 | [vec_env.py L17-L60](./vec_env.py#L17-L60) |
| 正运动学 | [vec_env.py L65-L70](./vec_env.py#L65-L70) |
| 连杆点与雅可比 | [vec_env.py L72-L92](./vec_env.py#L72-L92) |
| 障碍物 CBF | [vec_env.py L94-L101](./vec_env.py#L94-L101) |
| 关节和速度约束 | [vec_env.py L103-L114](./vec_env.py#L103-L114) |
| 二维 QP 过滤器 | [vec_env.py L116-L147](./vec_env.py#L116-L147) |
| 11 维观测 | [vec_env.py L149-L164](./vec_env.py#L149-L164) |
| 随机场景生成 | [vec_env.py L166-L230](./vec_env.py#L166-L230) |
| 环境 step 和奖励 | [vec_env.py L236-L286](./vec_env.py#L236-L286) |
| 四种方法 | [experiment_configs.py L26-L31](../experiment_configs.py#L26-L31) |
| PPO 配置与训练 | [train_arm2d.py L20-L94](../train_arm2d.py#L20-L94) |
| Fig.3 训练曲线 | [plot_arm2d_paper_fig3.py](../plot_arm2d_paper_fig3.py) |
| Fig.4 固定场景轨迹 | [plot_arm2d_fig4_no_dr.py](../plot_arm2d_fig4_no_dr.py) |

