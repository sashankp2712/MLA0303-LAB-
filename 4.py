import numpy as np

# Grid size
ROWS = 5
COLS = 5

# Rewards
STEP_COST = -1
DELIVERY_REWARD = 100
GAMMA = 0.9
THETA = 1e-4  # Convergence threshold

# Define delivery points and obstacles
delivery_points = [(4, 4), (0, 4)]
obstacles = [(1, 1), (2, 2)]

# Actions: Up, Down, Left, Right
actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

# Initialize value function and policy
V = np.zeros((ROWS, COLS))
policy = np.random.choice(list(actions.keys()), size=(ROWS, COLS))

# Check valid state
def is_valid(state):
    r, c = state
    return (0 <= r < ROWS and 
            0 <= c < COLS and 
            state not in obstacles)

# Get next state
def get_next_state(state, action):
    dr, dc = actions[action]
    next_state = (state[0] + dr, state[1] + dc)
    if is_valid(next_state):
        return next_state
    return state

# Policy Evaluation
def policy_evaluation():
    global V
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for r in range(ROWS):
            for c in range(COLS):
                
                if (r, c) in delivery_points:
                    new_V[r, c] = DELIVERY_REWARD
                    continue
                
                if (r, c) in obstacles:
                    continue
                
                action = policy[r, c]
                next_state = get_next_state((r, c), action)
                reward = STEP_COST
                
                new_V[r, c] = reward + GAMMA * V[next_state]
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        
        V = new_V
        
        if delta < THETA:
            break

# Policy Improvement
def policy_improvement():
    global policy
    policy_stable = True
    
    for r in range(ROWS):
        for c in range(COLS):
            
            if (r, c) in delivery_points or (r, c) in obstacles:
                continue
            
            old_action = policy[r, c]
            best_action = None
            best_value = float('-inf')
            
            for action in actions:
                next_state = get_next_state((r, c), action)
                value = STEP_COST + GAMMA * V[next_state]
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            policy[r, c] = best_action
            
            if old_action != best_action:
                policy_stable = False
    
    return policy_stable

# Main Policy Iteration Loop
def policy_iteration():
    while True:
        policy_evaluation()
        if policy_improvement():
            break

policy_iteration()

print("Optimal Value Function:")
print(np.round(V, 2))

print("\nOptimal Policy:")
print(policy)
