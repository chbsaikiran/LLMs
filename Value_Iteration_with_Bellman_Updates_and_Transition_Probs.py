states = [1, 2]
actions = ['a1', 'a2']
gamma = 0.9
theta = 1e-4  # convergence threshold

# Transition probabilities and rewards
# Format: (state, action) -> list of (probability, next_state, reward)
transition_model = {
    (1, 'a1'): [(1.0, 1, 0)],                  # always goes to state 1
    (1, 'a2'): [(0.8, 2, 1), (0.2, 1, 0)],      # mostly goes to terminal state 2
    (2, 'a1'): [(1.0, 1, 0)],                  # goes to state 1
    (2, 'a2'): [(1.0, 2, 0)]                   # stays in terminal state
}

# Initialize value function
V = {s: 0.0 for s in states}

def value_iteration():
    iteration = 0
    while True:
        delta = 0
        print(f"\n📘 Iteration {iteration}")
        for state in states:
            v_old = V[state]
            action_values = []

            for action in actions:
                expected_value = 0.0
                for prob, next_state, reward in transition_model[(state, action)]:
                    expected_value += prob * (reward + gamma * V[next_state])
                action_values.append(expected_value)

            V[state] = max(action_values)  # Bellman Optimality Equation
            delta = max(delta, abs(v_old - V[state]))
            print(f"  V({state}) updated to {V[state]:.4f}")

        iteration += 1
        if delta < theta:
            print("\n✅ Converged Value Function!")
            break

    # Extract optimal policy
    policy = {}
    for state in states:
        best_action = None
        best_value = float('-inf')
        for action in actions:
            expected_value = 0.0
            for prob, next_state, reward in transition_model[(state, action)]:
                expected_value += prob * (reward + gamma * V[next_state])
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
        policy[state] = best_action

    print(f"\n🧠 Optimal Value Function: {V}")
    print(f"🏁 Optimal Policy: {policy}")

# Run value iteration
value_iteration()
