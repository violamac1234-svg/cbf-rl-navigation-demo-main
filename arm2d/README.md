# 二自由度机械臂 CBF 迁移原型

这是迁移路线的第一阶段：先不训练 PPO，用可解释的末端伺服控制器验证运动学、
连杆碰撞距离、关节限位和多约束 CBF-filter。

```powershell
D:\anaconda3\envs\cbf_learning\python.exe -m arm2d.demo
```

状态为 `q=[q1,q2]`，动作为关节速度 `q_dot`。每根连杆布置 5 个碰撞采样点；
对每个采样点与每个圆形障碍物建立
`grad_q(h) @ q_dot + alpha*h >= 0`，并同时加入关节角和关节速度约束。
由于控制量只有二维，过滤器通过枚举边界投影和边界交点精确求解凸 QP，暂不依赖
外部优化库。

下一阶段是在这一核心之上封装向量化 Gym 环境，并把现有 nominal / reward-only /
filter-only / dual 四种训练配置接入。
