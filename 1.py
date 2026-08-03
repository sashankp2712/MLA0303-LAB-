import numpy as np
import random

# Grid size
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Create grid
#  1  = Dirt
# -1  = Obstacle
#  0  = Empty
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Define dirt positions
dirt_positions = [(0,2), (1,4), (3,1), (4,3)]
for pos in dirt_positions:
    grid[pos] = 1

# Define obstacle positions
obstacles = [(1,1), (2,3), (3,3)]
for pos in obstacles:
    grid[pos] = -1

# Movement function
def move(state, action):
    r, c = state

    if action == 'UP':
        r = max(0, r-1)
    elif action == 'DOWN':
        r = min(GRID_SIZE-1, r+1)
    elif action == 'LEFT':
        c = max(0, c-1)
    elif action == 'RIGHT':
        c = min(GRID_SIZE-1, c+1)

    # If obstacle, stay in same place
    if grid[r, c] == -1:
        return state, -1

    reward = grid[r, c]
    grid[r, c] = 0  # clean dirt if present
    return (r, c), reward

# -----------------------------
# RANDOM POLICY
# -----------------------------
def random_policy():
    state = (0,0)
    total_reward = 0
    steps = 50

    for _ in range(steps):
        action = random.choice(ACTIONS)
        state, reward = move(state, action)
        total_reward += reward

    return total_reward

# -----------------------------
# GREEDY POLICY
# -----------------------------
def greedy_policy():
    state = (0,0)
    total_reward = 0
    steps = 50

    for _ in range(steps):
        best_action = None
        best_reward = -float('inf')

        for action in ACTIONS:
            next_state, reward = move(state, action)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        state, reward = move(state, best_action)
        total_reward += reward

    return total_reward

# -----------------------------
# VALUE ITERATION
# -----------------------------
def value_iteration(gamma=0.9, theta=0.001):
    V = np.zeros((GRID_SIZE, GRID_SIZE))

    while True:
        delta = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                v = V[r, c]
                values = []

                for action in ACTIONS:
                    (nr, nc), reward = move((r,c), action)
                    values.append(reward + gamma * V[nr, nc])

                V[r, c] = max(values)
                delta = max(delta, abs(v - V[r, c]))

        if delta < theta:
            break

    return V

# -----------------------------
# RUN SIMULATION
# -----------------------------
print("Random Policy Reward:", random_policy())
print("Greedy Policy Reward:", greedy_policy())

V = value_iteration()
print("\nOptimal Value Function from Value Iteration:")
print(V)
