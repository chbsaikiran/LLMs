import random
from collections import defaultdict

# Environment definition
states = [1, 2]
actions = ['a1', 'a2']
gamma = 0.9
alpha = 0.1  # learning rate
epsilon = 0.1  # exploration rate

# Transition table: (state, action) → (next_state, reward, is_terminal)
transition_table = {
    (1, 'a1'): (1, 0, False),
    (1, 'a2'): (2, 1, True),
    (2, 'a1'): (1, 0, False),
    (2, 'a2'): (2, 0, True)
}

# Initialize Q(s, a)
Q = defaultdict(lambda: 0.0)

# ε-greedy policy
def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)  # explore
    else:
        q_values = [Q[(state, a)] for a in actions]
        max_q = max(q_values)
        best_actions = [a for a in actions if Q[(state, a)] == max_q]
        return random.choice(best_actions)  # exploit

# Simulate one episode
def run_episode():
    state = 1
    total_reward = 0
    step = 0
    while True:
        action = choose_action(state)
        next_state, reward, is_terminal = transition_table[(state, action)]
        total_reward += reward

        # Q-learning update
        next_qs = [Q[(next_state, a)] for a in actions]
        max_next_q = max(next_qs)
        Q[(state, action)] += alpha * (reward + gamma * max_next_q - Q[(state, action)])

        state = next_state
        step += 1

        if is_terminal:
            break
    return total_reward, step

# Run Q-learning for many episodes
def train_q_learning(episodes=1000):
    for episode in range(episodes):
        total_reward, steps = run_episode()
        if episode % 100 == 0:
            print(f"Episode {episode}: reward={total_reward}, steps={steps}")

    # Extract policy
    learned_policy = {}
    for state in states:
        best_action = max(actions, key=lambda a: Q[(state, a)])
        learned_policy[state] = best_action

    print("\n🧠 Learned Q-values:")
    for key in sorted(Q):
        print(f"Q{key} = {Q[key]:.4f}")

    print("\n🏁 Learned Policy:")
    print(learned_policy)

# Train the agent
train_q_learning(episodes=1000)


#The agent starts with zero knowledge, explores randomly.
#
#Over time, Q(s, a) values converge to optimal estimates.
#
#The learned policy will match the one from value iteration if trained long enough.
#
#Model-free: No need to know transition probabilities.
#
#Off-policy: Learns optimal policy even while exploring.
#
#Robust: Works with stochastic environments too.
