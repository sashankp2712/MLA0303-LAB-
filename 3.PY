import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Environment Setup
# -----------------------------
np.random.seed(42)

prices = np.array([10, 20, 30, 40, 50])
true_probabilities = np.array([0.30, 0.25, 0.18, 0.12, 0.08])
n_arms = len(prices)
n_rounds = 10000

def simulate_purchase(arm):
    return 1 if np.random.rand() < true_probabilities[arm] else 0


# -----------------------------
# 1️⃣ Epsilon-Greedy
# -----------------------------
def epsilon_greedy(epsilon=0.1):
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    revenue = []

    total = 0

    for t in range(n_rounds):
        if np.random.rand() < epsilon:
            arm = np.random.randint(n_arms)
        else:
            arm = np.argmax(values)

        reward = simulate_purchase(arm) * prices[arm]

        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]

        total += reward
        revenue.append(total)

    return revenue


# -----------------------------
# 2️⃣ UCB
# -----------------------------
def ucb():
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    revenue = []

    total = 0

    for t in range(n_rounds):
        if 0 in counts:
            arm = np.argmin(counts)
        else:
            confidence = np.sqrt((2 * np.log(t)) / counts)
            ucb_values = values + confidence
            arm = np.argmax(ucb_values)

        reward = simulate_purchase(arm) * prices[arm]

        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]

        total += reward
        revenue.append(total)

    return revenue


# -----------------------------
# 3️⃣ Thompson Sampling
# -----------------------------
def thompson_sampling():
    successes = np.zeros(n_arms)
    failures = np.zeros(n_arms)
    revenue = []

    total = 0

    for t in range(n_rounds):
        samples = [
            np.random.beta(successes[i] + 1, failures[i] + 1)
            for i in range(n_arms)
        ]
        arm = np.argmax(samples)

        purchase = simulate_purchase(arm)
        reward = purchase * prices[arm]

        if purchase == 1:
            successes[arm] += 1
        else:
            failures[arm] += 1

        total += reward
        revenue.append(total)

    return revenue


# -----------------------------
# Run Simulation
# -----------------------------
rev_eps = epsilon_greedy(epsilon=0.1)
rev_ucb = ucb()
rev_ts = thompson_sampling()

print("Final Revenue:")
print("Epsilon-Greedy:", rev_eps[-1])
print("UCB:", rev_ucb[-1])
print("Thompson Sampling:", rev_ts[-1])

# -----------------------------
# Plot Results
# -----------------------------
plt.plot(rev_eps, label="Epsilon-Greedy")
plt.plot(rev_ucb, label="UCB")
plt.plot(rev_ts, label="Thompson Sampling")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Revenue")
plt.title("Multi-Armed Bandit Pricing Comparison")
plt.legend()
plt.show()
