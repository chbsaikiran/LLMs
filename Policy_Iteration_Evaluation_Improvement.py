import random
from collections import defaultdict

# Define environment
states = [1, 2]
actions = ['a1', 'a2']
gamma = 0.9

# Environment: (state, action) → (next_state, reward, is_terminal)
transition_table = {
    (1, 'a1'): (1, 0, False),
    (1, 'a2'): (2, 1, True),
    (2, 'a1'): (1, 0, False),
    (2, 'a2'): (2, 0, True)
}

# Initialize policy randomly
policy = {
    1: random.choice(actions),
    2: random.choice(actions)
}

# Initialize value function
V = {s: 0.0 for s in states}

# Monte Carlo method: simulate an episode using current policy
def generate_episode(policy):
    episode = []
    state = 1
    while True:
        action = policy[state]
        next_state, reward, is_terminal = transition_table[(state, action)]
        episode.append((state, action, reward))
        if is_terminal:
            break
        state = next_state
    return episode

# Policy Evaluation using Monte Carlo
def policy_evaluation(policy, V, num_episodes=100):
    returns = defaultdict(list)
    
    for _ in range(num_episodes):
        episode = generate_episode(policy)
        
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G
            if state not in [x[0] for x in episode[0:t]]:  # first-visit MC
                returns[state].append(G)
                V[state] = sum(returns[state]) / len(returns[state])
    return V

# Policy Improvement: greedy w.r.t. Q(s,a)
def policy_improvement(V):
    policy_stable = True
    new_policy = {}
    
    for state in states:
        old_action = policy[state]
        action_values = {}
        for action in actions:
            next_state, reward, is_terminal = transition_table[(state, action)]
            action_values[action] = reward + gamma * V[next_state]
        best_action = max(action_values, key=action_values.get)
        new_policy[state] = best_action
        if best_action != old_action:
            policy_stable = False
    return new_policy, policy_stable

# Full Policy Iteration
def policy_iteration():
    global policy, V
    iteration = 0
    while True:
        print(f"\n📌 Iteration {iteration}")
        V = policy_evaluation(policy, V, num_episodes=100)
        print(f"V: {V}")
        policy, stable = policy_improvement(V)
        print(f"Policy: {policy}")
        if stable:
            print("\n✅ Policy converged!")
            break
        iteration += 1

# Run the full policy iteration loop
policy_iteration()
