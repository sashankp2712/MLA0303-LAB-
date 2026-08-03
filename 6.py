import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Number of ads
n_ads = 5

# True click probabilities (unknown to agent)
true_ctr = [0.05, 0.08, 0.03, 0.06, 0.10]

# Number of rounds
n_rounds = 10000


# -----------------------------
# 1️⃣ Epsilon-Greedy
# -----------------------------
def epsilon_greedy(epsilon=0.1):
    counts = np.zeros(n_ads)
    values = np.zeros(n_ads)
    total_rewards = []

    reward_sum = 0

    for t in range(n_rounds):
        if np.random.rand() < epsilon:
            action = np.random.randint(n_ads)
        else:
            action = np.argmax(values)

        reward = np.random.rand() < true_ctr[action]
        counts[action] += 1

        values[action] += (reward - values[action]) / counts[action]
        reward_sum += reward
        total_rewards.append(reward_sum)

    return total_rewards


# -----------------------------
# 2️⃣ UCB
# -----------------------------
def ucb():
    counts = np.zeros(n_ads)
    values = np.zeros(n_ads)
    total_rewards = []

    reward_sum = 0

    for t in range(n_rounds):
        if 0 in counts:
            action = np.argmin(counts)
        else:
            ucb_values = values + np.sqrt((2 * np.log(t)) / counts)
            action = np.argmax(ucb_values)

        reward = np.random.rand() < true_ctr[action]
        counts[action] += 1
        values[action] += (reward - values[action]) / counts[action]

        reward_sum += reward
        total_rewards.append(reward_sum)

    return total_rewards


# -----------------------------
# 3️⃣ Thompson Sampling
# -----------------------------
def thompson_sampling():
    alpha = np.ones(n_ads)
    beta = np.ones(n_ads)
    total_rewards = []

    reward_sum = 0

    for t in range(n_rounds):
        sampled_theta = np.random.beta(alpha, beta)
        action = np.argmax(sampled_theta)

        reward = np.random.rand() < true_ctr[action]

        if reward:
            alpha[action] += 1
        else:
            beta[action] += 1

        reward_sum += reward
        total_rewards.append(reward_sum)

    return total_rewards


# Run Algorithms
eg_rewards = epsilon_greedy(0.1)
ucb_rewards = ucb()
ts_rewards = thompson_sampling()

# Convert to CTR
eg_ctr = np.array(eg_rewards) / np.arange(1, n_rounds + 1)
ucb_ctr = np.array(ucb_rewards) / np.arange(1, n_rounds + 1)
ts_ctr = np.array(ts_rewards) / np.arange(1, n_rounds + 1)

# Plot Results
plt.plot(eg_ctr, label="Epsilon-Greedy")
plt.plot(ucb_ctr, label="UCB")
plt.plot(ts_ctr, label="Thompson Sampling")

plt.xlabel("Rounds")
plt.ylabel("Click Through Rate")
plt.title("Bandit Algorithm Comparison")
plt.legend()
plt.show()

print("Final CTRs:")
print("Epsilon-Greedy:", eg_ctr[-1])
print("UCB:", ucb_ctr[-1])
print("Thompson Sampling:", ts_ctr[-1])
