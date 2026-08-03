import numpy as np

# Grid size
GRID_SIZE = 4
gamma = 0.9
theta = 0.001  # convergence threshold

# Rewards grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Define item location (+2)
item_location = (1, 2)

# Define goal location (+5)
goal_location = (3, 3)

# Define obstacles (-2 penalty)
obstacles = [(1, 1), (2, 2)]

# Policy: move RIGHT if possible, else DOWN
def policy(state):
    r, c = state
    if c < GRID_SIZE - 1:
        return "RIGHT"
    else:
        return "DOWN"

# Transition function
def step(state, action):
    r, c = state

    if action == "UP":
        r = max(0, r - 1)
    elif action == "DOWN":
        r = min(GRID_SIZE - 1, r + 1)
    elif action == "LEFT":
        c = max(0, c - 1)
    elif action == "RIGHT":
        c = min(GRID_SIZE - 1, c + 1)

    next_state = (r, c)

    # Obstacle handling
    if next_state in obstacles:
        return state, -2

    # Item reward
    if next_state == item_location:
        return next_state, 2

    # Goal reward
    if next_state == goal_location:
        return next_state, 5

    return next_state, 0

# Policy Evaluation
def policy_evaluation():
    V = np.zeros((GRID_SIZE, GRID_SIZE))

    while True:
        delta = 0

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                state = (r, c)

                # Skip goal (terminal state)
                if state == goal_location:
                    continue

                v = V[r, c]
                action = policy(state)
                next_state, reward = step(state, action)

                V[r, c] = reward + gamma * V[next_state]
                delta = max(delta, abs(v - V[r, c]))

        if delta < theta:
            break

    return V

# Run evaluation
value_function = policy_evaluation()

print("Value Function under given policy:\n")
print(np.round(value_function, 2))
