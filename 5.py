import numpy as np

# Grid size
ROWS = 5
COLS = 5

# Parameters
GAMMA = 0.9
THETA = 1e-4
STEP_COST = -1
PICKUP_REWARD = 50

# Define pickup locations
pickup_points = [(0, 4), (4, 4)]

# Actions
actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

# Initialize value function
V = np.zeros((ROWS, COLS))
policy = np.full((ROWS, COLS), " ")

# Check valid state
def is_valid(state):
    r, c = state
    return 0 <= r < ROWS and 0 <= c < COLS

# Get next state
def get_next_state(state, action):
    dr, dc = actions[action]
    next_state = (state[0] + dr, state[1] + dc)
    if is_valid(next_state):
        return next_state
    return state

# Value Iteration
def value_iteration():
    global V
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for r in range(ROWS):
            for c in range(COLS):
                
                if (r, c) in pickup_points:
                    new_V[r, c] = PICKUP_REWARD
                    continue
                
                best_value = float('-inf')
                
                for action in actions:
                    next_state = get_next_state((r, c), action)
                    reward = STEP_COST
                    value = reward + GAMMA * V[next_state]
                    best_value = max(best_value, value)
                
                new_V[r, c] = best_value
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        
        V = new_V
        
        if delta < THETA:
            break

# Extract Policy
def extract_policy():
    for r in range(ROWS):
        for c in range(COLS):
            
            if (r, c) in pickup_points:
                policy[r, c] = "P"
                continue
            
            best_action = None
            best_value = float('-inf')
            
            for action in actions:
                next_state = get_next_state((r, c), action)
                value = STEP_COST + GAMMA * V[next_state]
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            policy[r, c] = best_action

# Run
value_iteration()
extract_policy()

print("Optimal Value Function:\n")
print(np.round(V, 2))

print("\nOptimal Policy:\n")
print(policy)
