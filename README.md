# 📊 Comparison of Q-learning, Policy Iteration, and Value Iteration

This document compares three reinforcement learning methods implemented in the Python scripts:

- `Policy_Iteration_Evaluation_Improvement.py`
- `Value_Iteration_with_Bellman_Updates.py`
- `Q_Learning.py`

---

## 🧠 Summary Table

| Feature                        | **Policy Iteration**                                | **Value Iteration**                                   | **Q-learning**                                         |
|-------------------------------|-----------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|
| 📄 **Script**                  | `Policy_Iteration_Evaluation_Improvement.py`        | `Value_Iteration_with_Bellman_Updates.py`              | `Q_Learning.py`                                        |
| 🔧 **Type**                   | Model-based (requires transition table)             | Model-based (requires transition table)                | Model-free (no transition probabilities required)       |
| 🧠 **Learns**                 | State-value function `V(s)` and policy `π(s)`       | State-value function `V(s)` → then derive `π(s)`       | Action-value function `Q(s, a)` directly                |
| 🔄 **Core Equation**          | Bellman Expectation Equation (via Monte Carlo)       | Bellman Optimality Equation                            | Bellman Optimality Equation (sample-based)              |
| 📈 **Policy Evaluation**      | First-visit Monte Carlo sampling                    | Direct Bellman updates                                | Temporal-difference updates via episodes                |
| 🧭 **Exploration**            | None (follows current policy)                       | None (full sweep over state space)                     | ε-greedy exploration                                    |
| 🔁 **Policy Update**         | Greedy improvement based on `Q(s, a)`                | Greedy improvement from `V(s)`                         | Derived from learned `Q(s, a)`                          |
| 📦 **Memory Used**            | `V(s)`, `returns[s]`, `policy[s]`                    | `V(s)`, `policy[s]`                                    | `Q(s, a)`                                              |
| 🧪 **Sample Efficiency**      | Requires many episodes for evaluation               | Efficient with known model                             | Needs many episodes but works with real environments    |
| 📚 **Convergence Type**       | Until policy is stable                              | Until value function change < threshold                | Learns gradually through sampled updates                |
| ✅ **Converges to Optimal?**  | Yes (if fully evaluated)                            | Yes                                                    | Yes (if exploration and learning rate are sufficient)   |

---

## 📝 Summary

### 🟦 Policy Iteration
- Alternates between **Monte Carlo-based policy evaluation** and **greedy policy improvement**.
- Requires full environment model.
- Evaluates policies using simulated episodes.

### 🟨 Value Iteration
- Combines evaluation and improvement in one Bellman update loop.
- Works best when the full environment model is available.
- Typically converges faster than policy iteration.

### 🟥 Q-learning
- A **model-free** method that learns from experience using temporal difference learning.
- Does not require environment model.
- Uses an ε-greedy strategy to balance exploration and exploitation.

---

## 📎 Notes

- All three methods will converge to the optimal policy in tabular environments under the right conditions.
- Q-learning is more scalable to real-world problems where the transition model is unknown.
- Value and policy iteration are useful for understanding and solving small MDPs exactly.

---

Happy Reinforcement Learning! 🚀
