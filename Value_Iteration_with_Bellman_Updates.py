states = [1, 2]
actions = ['a1', 'a2']
gamma = 0.9
theta = 1e-4  # convergence threshold

# Environment: (state, action) -> (next_state, reward, is_terminal)
transition_table = {
    (1, 'a1'): (1, 0, False),
    (1, 'a2'): (2, 1, True),
    (2, 'a1'): (1, 0, False),
    (2, 'a2'): (2, 0, True)
}

# Initialize V(s) = 0
V = {s: 0.0 for s in states}

def value_iteration():
    iteration = 0
    while True:
        delta = 0  # max change in V
        print(f"\n📘 Iteration {iteration}")
        for state in states:
            v_old = V[state]
            q_values = []

            for action in actions:
                next_state, reward, is_terminal = transition_table[(state, action)]
                q = reward + gamma * V[next_state]
                q_values.append(q)

            V[state] = max(q_values)  # Bellman Optimality Update
            delta = max(delta, abs(v_old - V[state]))
            print(f"  V({state}) updated to {V[state]:.4f}")

        iteration += 1
        if delta < theta:
            print("\n✅ Converged Value Function!")
            break

    # Extract optimal policy from final V(s)
    policy = {}
    for state in states:
        best_action = None
        best_q = float('-inf')
        for action in actions:
            next_state, reward, is_terminal = transition_table[(state, action)]
            q = reward + gamma * V[next_state]
            if q > best_q:
                best_q = q
                best_action = action
        policy[state] = best_action

    print(f"\n🧠 Optimal Value Function: {V}")
    print(f"🏁 Optimal Policy: {policy}")

# Run value iteration
value_iteration()
