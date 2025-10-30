1. 原始HER及其基础变种
1.1 HER (Hindsight Experience Replay) - 2017
原理：

将失败轨迹中实际到达的状态作为"虚拟目标"
重标注经验，将失败变为成功

Goal选择策略：

final: 使用episode最终状态
future: 从当前时刻之后随机选k个状态（最常用）
episode: 从整个episode随机选择
random: 完全随机

数学形式：
原始: (s, a, r, s', g)  其中 r = 0 (失败)
HER:  (s, a, r', s', g') 其中 g' = s_achieved, r' = 1 (成功)

2. 基于优先级/能量的变种
2.1 Energy-based HER (EbHER) - 2018
基于物理学中的功-能原理定义轨迹能量函数，假设重放具有高轨迹能量的episode对机器人强化学习更有效 UnrResearchGate。
核心思想：

不均匀采样hindsight经验
优先选择"高能量"轨迹
能量 = 势能 + 动能 + 转动能

公式：
pythonE_trajectory = Σ (E_potential + E_kinetic + E_rotational)
```

### 2.2 **Prioritized HER (PHER)** - 2018

结合Prioritized Experience Replay与HER。

**采样概率**：
```
P(i) ∝ (|TD_error_i| + ε)^α
2.3 Experience Ranking with HER - 2019
对训练经验进行排序并过滤掉大多数低排名经验，只保留高评分经验用于学习 Unr。
排序标准：

距离目标的接近度
探索质量
信息增益


3. 基于Curriculum Learning的变种
3.1 Curriculum-guided HER (CHER) - 2019 ⭐
自适应选择失败经验进行重放，基于接近真实目标的程度和探索多样化伪目标的好奇心，并逐渐改变比例：早期阶段强调好奇心，后期转向更大的目标接近度 NIPSACM Digital Library。
核心思想：

Early stage: 好奇心权重高 (探索多样化)
Late stage: 接近度权重高 (exploitation)
自动curriculum

公式：
pythoncurriculum_stage = t / T  # 训练进度

# 早期: curiosity_weight = 0.8, proximity_weight = 0.2
# 后期: curiosity_weight = 0.2, proximity_weight = 0.8

goal_score = (1-stage) * curiosity_score + stage * proximity_score
```

### 3.2 **Sequential HER (SHER)** - 2021

将复杂任务分解为一系列源任务的课程，每个源任务是下一个的简化版本，复杂度递增 。

**流程**：
```
Task: 复杂操作
├── Subtask 1 (简单)
├── Subtask 2 (中等)
├── Subtask 3 (复杂)
└── Subtask N (原始任务)
3.3 Relay HER (RHER) - 2021
分解顺序任务为多个子任务并重组为复杂度递增的新子任务，使用自我引导探索策略利用已学习的简单子任务策略引导更复杂子任务的探索 ScienceDirectScienceDirect。
关键创新：Self-Guided Exploration Strategy (SGES)

4. 基于目标生成的变种
4.1 Hindsight Goal Generation (HGG) - 2019 ⭐
生成有价值的hindsight目标，这些目标短期内容易实现，长期内也有潜力引导agent到达实际目标 ACM Digital LibraryResearchGate。
核心算法：
pythondef select_hindsight_goal(trajectory, goal):
    """选择中间目标"""
    candidates = []
    
    for state in trajectory:
        # 1. 容易达到 (短期)
        ease_score = compute_reachability(state, current_policy)
        
        # 2. 有潜力引导到真实目标 (长期)
        potential_score = -distance(state, goal)
        
        score = ease_score + λ * potential_score
        candidates.append((state, score))
    
    return max(candidates, key=lambda x: x[1])
```

**与HER的区别**：
- HER: 随机选择achieved goals
- HGG: 智能选择有价值的中间目标

### 4.2 **Graph-based HGG (G-HGG)** - 2021

基于避障图中的最短距离选择hindsight目标，该图是环境的离散表示，使HGG适用于有障碍物的操作任务 。

**改进**：
- 考虑障碍物
- 使用图搜索算法
- 更准确的距离度量

### 4.3 **Guided Goal Generation (G-HER)** - 2019

使用条件生成RNN显式建模策略级别与目标之间的关系，能够生成基于不同策略级别条件的各种目标 。

**网络结构**：
```
policy_level → RNN → goal_sequence
     ↓              ↓
  replay_transition
4.4 Hindsight Goal Filtering (HGF)
在HTRPO中提出，使用Max-Min Distance选择分散的hindsight目标。

5. 基于模型的变种
5.1 Model-based HER (MB-HER) - 2021
使用预测动力学模型预测未来状态，引入foresight relabeling，将预测的未来状态作为达到的目标 PeerJ。
关键创新：Foresight Relabeling (FR)
流程：
python# 1. 学习动力学模型
s_next_pred = dynamics_model(s, a)

# 2. 预测未来轨迹
future_trajectory = predict_trajectory(dynamics_model, s, policy)

# 3. 使用预测状态作为虚拟目标
virtual_goals = sample_from(future_trajectory)
5.2 Model-based Relay HER (MRHER) - 2023
打破连续任务为复杂度递增的子任务，使用Foresight relabeling预测hindsight状态的未来轨迹，并将期望目标重标注为虚拟未来轨迹上达到的目标 arXiv。

6. 适配On-policy算法的变种
6.1 Hindsight Policy Gradients (HPG) - 2017
将hindsight引入likelihood-ratio policy gradient方法，将这种能力推广到整个高度成功的算法类 Semantic Scholar。
核心：

将HER扩展到policy gradient
适用于REINFORCE等算法

6.2 Hindsight Trust Region Policy Optimization (HTRPO) - 2019 ⭐
使用hindsight扩展TRPO算法，在稀疏奖励条件下有效利用交互在信任区域内优化策略，同时保持学习稳定性 OpenReviewIJCAI。
核心贡献：

QKL (Quadratic KL Estimation):

python# 传统KL估计: 高方差
KL_div = E[log(π_new/π_old)]

# QKL: 二次近似，降低方差
KL_approx ≈ (1/2) * E[(π_new - π_old)²/π_old²]

Hindsight Expected Return:

pythonJ_hindsight(θ) = E[A_θ(s, a | g')]  # g'是hindsight goal
收敛性：保持TRPO的单调改进性质
6.3 HER + PPO - 2024
展示HER可以显著加速PPO算法，尽管事后修改观察到的目标违反了on-policy算法的假设 arXiv。
关键发现：

Vanilla PPO+HER在连续动作空间有效
不需要轨迹过滤或self-imitation learning


7. 处理特殊场景的变种
7.1 Dynamic HER (DHER) - 2019
用于动态目标任务，自动从两个相关失败中组装成功经验，可增强任意off-policy RL算法处理随时间变化的目标 OpenReviewPapers with Code。
核心思想：
python# 场景: 目标在移动
trajectory_1: 追踪目标到位置A (失败)
trajectory_2: 追踪目标到位置B (失败)

# DHER: 组合两个轨迹
success_traj = combine(trajectory_1, trajectory_2, dynamic_goal)
7.2 Failed Goal Aware HER (FAHER) - 2024
通过对失败目标进行聚类提高采样效率，在采样时考虑达到目标相对于失败目标的属性 PeerJ。
算法：
python# 1. 聚类失败目标
clusters = kmeans(failed_goals, n_clusters=K)

# 2. 从每个cluster采样，确保多样性
for cluster in clusters:
    sample_hindsight_experience(cluster)
改进：

CloseSlide: +3.80%
FarSlide: +22.58%
FarNarrowSlide: +7.09%

7.3 Multi-step HER - 2023
通过重标注轨迹中多个连续的transition而非单个transition来解决标准HER的固有偏差，增加非负学习信号的数量 PeerJ。
改进点：
python# 标准HER: 单步重标注
(s_t, a_t, r_t, s_{t+1}, g') 

# Multi-step HER: 多步重标注
[(s_t, a_t, r_t, s_{t+1}, g'),
 (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}, g'),
 ...,
 (s_{t+n}, a_{t+n}, r_{t+n}, s_{t+n+1}, g')]
7.4 Bias-reduced HER - 2021
优先选择虚拟目标以使agent学习更有价值的信息，并通过移除误导性样本来减少HER中的现有偏差 ScienceDirect。
两个关键改进：

Instructional-Based Strategy (IBS): 选择instructive的虚拟目标
Filtered-HER: 移除misleading samples

7.5 Sampling Rate Decay in HER - 2020
提出采样率衰减策略，在训练过程中减少hindsight经验的数量，因为固定采样率引入偏差且不允许hindsight经验在不同训练阶段具有可变重要性 PubMed。
策略：
pythonsampling_rate(t) = initial_rate * decay_factor^t
动机：

早期: 需要更多hindsight (探索)
后期: 减少hindsight (减少偏差)

7.6 Addressing Hindsight Bias (ARCHER) - 2023
解决HER引入的bias问题。
7.7 Imaginary Filtered HER (IFHER) - 2022
通过合理想象失败episode中的目标轨迹生成成功episode来增强经验，精心设计的目标、episode和质量过滤策略确保只存储高质量的增强经验 ScienceDirect。
应用场景：UAV动态目标跟踪

8. 其他重要变种
8.1 Decayed Hindsight (DH)
分析偏斜目标并诱导decayed hindsight，通过对抗exploration和hindsight replay之间的偏差实现一致的多目标经验重放 Semantic Scholar。
8.2 Multi-Objective Evolutionary HER (MOEHER) - 2024
重新制定考虑多个目标和障碍物的curriculum生成 ACM Digital Library。
特点：

考虑多目标
进化算法生成curriculum

8.3 Contact Energy Based HER Prioritization - 2023
针对机器人接触任务的能量based优先级。
8.4 Goal-Conditioned Hindsight Regularization (GCHR) - 2025
最大化off-policy GCRL的经验利用，包含Hindsight Self-Imitation Regularization (HSR)和Hindsight Goal Regularization (HGR)两个组件 arXiv。
公式：
python# HSR: 自模仿正则化
L_HSR = -log π(a|s,g') 

# HGR: 目标正则化
L_HGR = KL(π(·|s,g') || π_prior(·|s,g'))
