# 🧠 Solving MDPs: Policy Iteration vs Value Iteration

This repository contains two Python scripts that demonstrate two foundational algorithms in **Reinforcement Learning** for solving **Markov Decision Processes (MDPs)**:

1. `Policy_Iteration_Evaluation_Improvement.py` — Monte Carlo-based **Policy Iteration**
2. `Value_Iteration_with_Bellman_Updates.py` — **Value Iteration** using **Bellman Optimality Updates**

---

## 📁 Overview of Files

### 1. `Policy_Iteration_Evaluation_Improvement.py`

🔹 **Method**: Policy Iteration  
🔹 **Policy Evaluation**: Done using **Monte Carlo (First-Visit)** sampling  
🔹 **Policy Improvement**: Greedy selection based on estimated `Q(s, a)`  
🔹 **Exploration**: Simulates episodes under the current policy  
🔹 **Stochastic**: Uses sampling → results may vary slightly run-to-run  
🔹 **Convergence**: Occurs when the policy no longer changes

#### 🔧 Algorithm Steps:
- Simulate episodes following the current policy
- Estimate returns `G` for each state (First-Visit Monte Carlo)
- Compute `V(s)` as the average return
- Improve the policy using:
  \[
  \pi(s) = \arg\max_a [r + \gamma V(s')]
  \]
- Repeat until policy stabilizes

---

### 2. `Value_Iteration_with_Bellman_Updates.py`

🔹 **Method**: Value Iteration  
🔹 **Policy Evaluation and Improvement**: Combined via **Bellman Optimality Equation**  
🔹 **Deterministic**: Fully model-based and analytical  
🔹 **Convergence**: Updates value function `V(s)` until the change (`delta`) is below a small threshold  
🔹 **Final Policy Extraction**: Greedy with respect to final value function

#### 🔧 Algorithm Steps:
- Initialize `V(s) = 0`
- Iteratively update using:
  \[
  V(s) = \max_a \left[ r + \gamma V(s') \right]
  \]
- After convergence, extract policy:
  \[
  \pi(s) = \arg\max_a \left[ r + \gamma V(s') \right]
  \]

---

## 📊 Key Differences

| Feature                      | Policy Iteration (MC)                          | Value Iteration (Bellman)                   |
|-----------------------------|------------------------------------------------|---------------------------------------------|
| **Evaluation Method**       | Monte Carlo (episode-based sampling)          | Bellman updates (exact recursive updates)   |
| **Exploration**             | Simulates full episodes                       | No episodes, uses full model directly       |
| **Stability**               | Can have randomness due to sampling           | Deterministic, consistent results           |
| **When to Use**             | When environment model is not fully known     | When full transition model is available     |
| **Implementation Style**    | Model-free (sample-based)                     | Model-based (equation-based)                |

---

## ⚙️ Running the Scripts

Both scripts are self-contained and can be run with:

```bash
python Policy_Iteration_Evaluation_Improvement.py
python Value_Iteration_with_Bellman_Updates.py
