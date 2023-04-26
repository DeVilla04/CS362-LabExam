import numpy as np

# Define the reward and probability arrays
reward = [100 , 500 , 1000 , 5000, 10000 , 50000 , 100000 , 500000 , 1000000 , 5000000]
probability = [0.99 , 0.9, 0.8, 0.7, 0.6, 0.5 , 0.4 , 0.3 , 0.2 , 0.1]

assert(len(reward) == len(probability))

# Define the discount factor
gamma = 0.9

# Define the transition probabilities
p = np.zeros((len(reward)+1, 2, len(reward)+2))
for i in range(len(reward)):
    p[i, 0, i+1] = probability[i]
    p[i, 0, -1] = 1 - probability[i]
    p[i, 1, -1] = 1
p[-2, 0, -1] = 1
p[-1, 0, -1] = 1

# Define the value function and policy arrays
V = np.zeros(len(reward)+2)
policy = np.zeros(len(reward)+2, dtype=int)

# Perform value iteration
for i in range(1000):
    Q = np.zeros((len(reward)+2, 2))
    for s in range(len(reward)+1):
        for a in range(2):
            for s_next in range(len(reward)+2):
                Q[s, a] += p[s, a, s_next] * (reward[s-1] + gamma * V[s_next])
        policy[s] = np.argmax(Q[s, :])
        V[s] = np.max(Q[s, :])

# Print the optimal policy and value function
# print("Optimal policy:", policy)
print("Optimal value function:")
for i , v in enumerate(V):
	print(f"{i} -> {v}")


print(f"Maximum reward : {max(V)}")
