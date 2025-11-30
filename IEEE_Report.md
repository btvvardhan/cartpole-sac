# Soft Actor-Critic for Continuous Control: A Deep Reinforcement Learning Approach to Cart-Pole Balancing

## Abstract

This paper presents an implementation and evaluation of the Soft Actor-Critic (SAC) algorithm for solving the cart-pole balancing task in continuous control. SAC is an off-policy, model-free deep reinforcement learning algorithm that combines maximum entropy reinforcement learning with actor-critic methods. We train SAC agents on the DeepMind Control Suite cart-pole balance environment and achieve stable performance with an average evaluation reward of 993.19 ± 5.03 across three independent training runs. The implementation utilizes twin Q-networks for value estimation, a stochastic policy with entropy regularization, and experience replay for sample-efficient learning. Our results demonstrate SAC's ability to learn effective control policies in continuous action spaces, achieving near-optimal balancing performance with low variance across multiple random seeds.

**Keywords**: Deep Reinforcement Learning, Soft Actor-Critic, Continuous Control, Cart-Pole, Maximum Entropy Reinforcement Learning

---

## I. INTRODUCTION

Deep Reinforcement Learning (DRL) has emerged as a powerful paradigm for solving complex control problems in continuous action spaces [1]. Traditional Q-learning methods face challenges when dealing with continuous actions, as they require maximizing over a continuous action space, which is computationally intractable. Actor-critic methods address this limitation by maintaining separate policy (actor) and value (critic) networks, enabling efficient learning in continuous control domains [2].

The Soft Actor-Critic (SAC) algorithm, introduced by Haarnoja et al. [3], [4], extends actor-critic methods by incorporating maximum entropy reinforcement learning principles. SAC maximizes both the expected return and the entropy of the policy, promoting exploration while maintaining stability. This approach has demonstrated state-of-the-art performance across various continuous control benchmarks in the MuJoCo simulation environment [4].

In this work, we implement and evaluate SAC on the cart-pole balancing task from the DeepMind Control Suite [6]. The cart-pole system is a classic benchmark in control theory and reinforcement learning, involving a cart that must balance an upright pole through continuous force application. This task requires precise control and serves as an excellent testbed for continuous control algorithms.

### A. Problem Statement

The cart-pole balancing problem involves maintaining an inverted pendulum (pole) in an upright position by applying continuous forces to the cart. The system's state space includes the cart's position and velocity, as well as the pole's angle and angular velocity. The action space consists of continuous force values applied to the cart. The objective is to learn a policy that maximizes the cumulative reward by keeping the pole balanced for as long as possible.

### B. Contributions

The main contributions of this work are:

1. **Implementation**: A complete implementation of the SAC algorithm with proper handling of Dict observation spaces from the DeepMind Control Suite environment.

2. **Evaluation**: Comprehensive evaluation across multiple random seeds demonstrating stable convergence and reproducible results.

3. **Analysis**: Detailed analysis of the learning dynamics, convergence properties, and model architecture.

4. **Reproducibility**: Open-source implementation with GPU-accelerated training and comprehensive documentation.

## II. RELATED WORK

### A. Continuous Control in Reinforcement Learning

Traditional reinforcement learning algorithms like Deep Q-Network (DQN) [7] are designed for discrete action spaces. Extending these methods to continuous control requires modifications such as discretization of the action space or policy gradient methods [8].

### B. Actor-Critic Methods

Actor-critic methods combine the benefits of value-based and policy-based approaches. The actor learns a policy, while the critic estimates the value function, providing guidance for policy updates. Deep Deterministic Policy Gradient (DDPG) [9] was an early successful actor-critic method for continuous control, using deterministic policies and off-policy learning.

### C. Maximum Entropy Reinforcement Learning

Maximum entropy reinforcement learning [10], [11] augments the standard RL objective with an entropy term, encouraging the policy to explore more broadly. This approach has been shown to improve sample efficiency and robustness. SAC builds upon this principle, combining it with actor-critic methods.

### D. Soft Actor-Critic

SAC [3], [4] integrates maximum entropy RL with actor-critic methods, using a stochastic policy and twin Q-networks to reduce overestimation bias. It has demonstrated superior sample efficiency and stability compared to previous methods like DDPG and Trust Region Policy Optimization (TRPO) [12].

## III. METHODOLOGY

### A. Soft Actor-Critic Algorithm

SAC learns a stochastic policy $\pi_{\phi}$ and estimates Q-functions $Q_{\theta_1}$ and $Q_{\theta_2}$ (twin networks) through the following objective functions.

#### 1) Policy Objective

The policy is trained to maximize the expected return while maximizing entropy:

$$J_{\pi}(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \mathbb{E}_{a_t \sim \pi_{\phi}} \left[ \alpha \log \pi_{\phi}(a_t|s_t) - Q_{\theta}(s_t, a_t) \right] \right]$$

where $\alpha$ is the temperature parameter that controls the trade-off between exploration and exploitation, $\mathcal{D}$ is the replay buffer, and $Q_{\theta}(s_t, a_t) = \min(Q_{\theta_1}(s_t, a_t), Q_{\theta_2}(s_t, a_t))$ to reduce overestimation.

#### 2) Critic Objective

The Q-functions are updated to minimize the Bellman error:

$$J_Q(\theta_i) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim \mathcal{D}} \left[ \left( Q_{\theta_i}(s_t, a_t) - y_t \right)^2 \right]$$

where the target is:

$$y_t = r_t + \gamma \left( Q_{\bar{\theta}}(s_{t+1}, \tilde{a}_{t+1}) - \alpha \log \pi_{\phi}(a_{t+1}|s_{t+1}) \right)$$

Here, $\tilde{a}_{t+1} \sim \pi_{\phi}(\cdot|s_{t+1})$, $\gamma$ is the discount factor, and $\bar{\theta}$ represents target network parameters updated via soft updates.

#### 3) Reparameterization Trick

For continuous actions, SAC uses the reparameterization trick [13] for differentiable policy gradients:

$$a_t = f_{\phi}(\epsilon_t; s_t)$$

where $\epsilon_t \sim \mathcal{N}(0, 1)$ is a noise vector, enabling backpropagation through the policy network.

### B. Network Architecture

#### 1) Actor Network

The actor network implements a stochastic Gaussian policy:

- **Input Layer**: 5-dimensional state vector (cart position, cart velocity, pole angle, pole angular velocity, normalized reward signal)
- **Hidden Layers**: Two fully connected layers with 256 units each, using ReLU activations
- **Output Layers**: 
  - Mean head: Linear layer outputting 1D mean
  - Log-standard-deviation head: Linear layer outputting 1D log-standard-deviation (clamped between -20 and 2)

The action is sampled from the Gaussian distribution and squashed through a tanh function to ensure it lies within the action bounds.

#### 2) Critic Network

The critic implements twin Q-networks (double Q-learning) to reduce overestimation:

- **Input Layer**: Concatenation of 5D state and 1D action (6D total)
- **Hidden Layers**: Two fully connected layers with 256 units each, using ReLU activations
- **Output Layer**: Linear layer outputting a single Q-value

Two independent Q-networks ($Q_{\theta_1}$ and $Q_{\theta_2}$) share the same architecture but are trained independently.

#### 3) Target Networks

Separate target networks for the critics are maintained and updated via soft updates:

$$\bar{\theta}_i \leftarrow \tau \theta_i + (1 - \tau) \bar{\theta}_i$$

where $\tau = 0.005$ is the soft update coefficient.

### C. Experience Replay

SAC uses off-policy learning with an experience replay buffer [7] of capacity 1,000,000 transitions. During training, mini-batches of 256 transitions are sampled uniformly from the buffer to decorrelate updates and improve sample efficiency.

### D. Observation Space Handling

The DeepMind Control Suite environment returns observations as a Dict containing keys such as 'position', 'velocity', etc. We flatten this dictionary into a 5-dimensional vector for network input, handling the conversion automatically through utility functions.

## IV. EXPERIMENTAL SETUP

### A. Environment

We use the `dm_control/cartpole-balance-v0` environment from the DeepMind Control Suite [6], accessed through the Gymnasium interface via the Shimmy compatibility layer [14]. The environment features:

- **State Space**: Dict with 5-dimensional flattened observation space
- **Action Space**: Continuous 1D force on the cart, bounded to $[-1, 1]$
- **Reward**: Continuous reward signal proportional to how upright the pole remains
- **Episode Length**: Maximum 1000 steps (time limit removed for evaluation)

### B. Training Configuration

#### 1) Hyperparameters

Our implementation uses the following hyperparameters, consistent with the original SAC paper [3]:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 256 | Hidden layer dimension |
| `lr` | 3×10⁻⁴ | Learning rate for Adam optimizer |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft update coefficient |
| `alpha` | 0.2 | Entropy temperature (fixed) |
| `batch_size` | 256 | Mini-batch size |
| `buffer_size` | 1,000,000 | Replay buffer capacity |

#### 2) Training Procedure

Training is conducted across three independent random seeds (0, 1, 2) for statistical significance:

- **Training Episodes**: 300 episodes per seed
- **Total Training Episodes**: 900
- **Update Frequency**: After each step, if buffer contains at least `batch_size` samples
- **Evaluation**: 10 episodes after training with seed 10

#### 3) Hardware

All experiments are run on GPU (CUDA) for accelerated neural network computations. The implementation enforces GPU-only execution to ensure consistent, fast training.

### C. Evaluation Protocol

After training, agents are evaluated on 10 independent episodes using a deterministic policy (mean action without exploration noise). The evaluation uses a separate random seed (10) to ensure unbiased performance assessment.

## V. RESULTS

### A. Training Performance

Table I shows the training results for each seed, measured as the mean reward over the final 10 episodes:

**TABLE I. TRAINING RESULTS (LAST 10 EPISODES)**

| Seed | Mean Reward | Std Dev | Best Episode |
|------|-------------|---------|--------------|
| 0    | 948.69      | 2.07    | 955.55       |
| 1    | 944.15      | 3.53    | 958.32       |
| 2    | 944.45      | 4.49    | 957.11       |
| **Overall** | **945.76** | **4.07** | **958.32** |

The results demonstrate stable convergence across all seeds with low variance (σ = 4.07), indicating consistent learning behavior.

### B. Evaluation Performance

Table II presents the evaluation results:

**TABLE II. EVALUATION RESULTS (10 EPISODES)**

| Seed | Evaluation Reward | Status |
|------|-------------------|--------|
| 0    | 997.14           | Best   |
| 1    | 986.10           | -      |
| 2    | 996.34           | -      |
| **Mean** | **993.19**    | -      |
| **Std** | **5.03**       | -      |

The evaluation performance (993.19 ± 5.03) exceeds training performance, indicating good generalization. The best model achieves 997.14 reward, demonstrating near-optimal balancing capability.

### C. Learning Dynamics

The learning curve (Figure 1, referenced in `results/learning_curve.png`) shows:

- **Initial Performance**: ~360-370 reward (episode 10)
- **Convergence**: ~600-700 reward (episode 150-200)
- **Final Performance**: ~945-950 reward (last 10 episodes)

The agent demonstrates stable learning with monotonic improvement and low variance across seeds after convergence.

### D. Model Statistics

The trained model consists of:

- **Actor Parameters**: 67,842
- **Critic Parameters**: 135,682 (combined Q1 and Q2)
- **Total Parameters**: 203,524
- **Model Size**: 2.87 MB

The compact model size demonstrates SAC's efficiency in parameter usage.

### E. Convergence Analysis

The SAC algorithm converges reliably across all seeds:
- **Stability**: All seeds show consistent, stable learning
- **Variance**: Low variance (±4.07) in final performance
- **Reproducibility**: Similar final performance across seeds (944-949 range)

## VI. DISCUSSION

### A. Algorithm Performance

SAC successfully learns effective balancing policies, achieving evaluation rewards near the theoretical maximum (~1000). The algorithm's combination of maximum entropy RL and actor-critic methods enables both stable learning and effective exploration.

### B. Hyperparameter Sensitivity

The fixed entropy temperature (α = 0.2) worked well for this task. Adaptive temperature methods [4] could potentially improve performance further but were not necessary for achieving strong results.

### C. Twin Q-Networks

The use of twin Q-networks helps reduce overestimation bias, contributing to stable learning. The minimum of Q1 and Q2 used in the policy objective prevents the policy from exploiting overestimated Q-values.

### D. Sample Efficiency

SAC demonstrates good sample efficiency, achieving near-optimal performance within 300 episodes. The experience replay buffer and off-policy learning contribute to this efficiency by reusing past experiences.

### E. Limitations

While our implementation achieves strong performance, several limitations exist:

1. **Fixed Entropy Temperature**: Adaptive temperature tuning could improve exploration-exploitation balance
2. **Network Architecture**: The architecture is fixed; hyperparameter optimization could yield improvements
3. **Single Task**: Evaluation is limited to one environment; broader evaluation would assess generalization

### F. Comparison with Baselines

While direct comparison with other algorithms is not provided in this work, SAC's performance (993.19 ± 5.03) aligns with state-of-the-art results on similar continuous control tasks [3], [4]. The low variance and stable convergence demonstrate SAC's reliability.

## VII. CONCLUSION

This paper presented an implementation and evaluation of the Soft Actor-Critic algorithm on the cart-pole balancing task. Our results demonstrate:

1. **Effectiveness**: SAC successfully learns stable balancing policies, achieving evaluation rewards of 993.19 ± 5.03
2. **Stability**: Consistent performance across multiple random seeds with low variance
3. **Sample Efficiency**: Near-optimal performance within 300 training episodes
4. **Reproducibility**: Open-source implementation with comprehensive documentation

The implementation handles the complexities of continuous control, Dict observation spaces, and GPU acceleration, providing a solid foundation for further research and applications. Future work could explore adaptive entropy temperature, architecture search, and evaluation on additional continuous control benchmarks.

## ACKNOWLEDGMENT

The authors acknowledge the use of the DeepMind Control Suite, Gymnasium, and PyTorch frameworks in this work.

## REFERENCES

[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," *Commun. ACM*, vol. 60, no. 6, pp. 84-90, 2017.

[2] V. Mnih et al., "Asynchronous methods for deep reinforcement learning," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2016, pp. 1928-1937.

[3] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2018, pp. 1861-1870.

[4] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, and S. Levine, "Soft actor-critic algorithms and applications," *arXiv preprint arXiv:1812.05905*, 2018.

[5] Y. Tassa, N. Mansard, and E. Todorov, "Control-limited differential dynamic programming," in *Proc. IEEE Int. Conf. Robot. Automat. (ICRA)*, 2014, pp. 1168-1175.

[6] Y. Tassa et al., "DeepMind Control Suite," *arXiv preprint arXiv:1801.00690*, 2018.

[7] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529-533, 2015.

[8] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[9] T. P. Lillicrap et al., "Continuous control with deep reinforcement learning," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2016.

[10] B. D. Ziebart, "Modeling purposeful adaptive behavior with the principle of maximum causal entropy," Ph.D. dissertation, Carnegie Mellon University, 2010.

[11] E. Todorov, "Linearly-solvable Markov decision problems," in *Advances in Neural Information Processing Systems (NIPS)*, 2006, pp. 1369-1376.

[12] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, "Trust region policy optimization," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2015, pp. 1889-1897.

[13] D. P. Kingma and M. Welling, "Auto-encoding variational bayes," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2014.

[14] J. K. Terry et al., "Shimmy: Gymnasium Compatibility for Popular External Simulation Suites," *arXiv preprint arXiv:2307.00386*, 2023.

---

**Note**: This report follows IEEE conference paper format guidelines. Figures and tables referenced in the text should be included separately. The learning curve plot is available in `results/learning_curve.png`, and model comparison visualization is in `results/model_comparison.png`.

