import numpy as np
from ql_gen import sim_ql_gen
import matplotlib.pyplot as plt

rounds = 1000
trials = 100
arms = 4
return_values = True
default_x0 = [0.1, 0.2, 0.5, 0.05, 0.5, 0.1]  # tau, lr_max, lr_sig, decay, sticky, q0
param_names = ['tau', 'lr_max', 'lr_sig', 'decay', 'sticky', 'q0']
bounds = {
    'tau': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    'lr_max': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'lr_sig': [0.1, 0.2, 0.5, 0.8, 1.0],
    'decay': [0.01, 0.02, 0.05, 0.08, 0.1],
    'sticky': [0.1, 0.2, 0.5, 0.8, 1.0],
    'q0': [0.01, 0.05, 0.1, 0.2, 0.3]
}

results = {}
for param in param_names:
    values = bounds[param]
    reward_rates = []
    for val in values:
        x0 = default_x0.copy()
        idx = param_names.index(param)
        x0[idx] = val
        a, r, q = sim_ql_gen(x0, (rounds, trials), arms, return_values)
        reward_rate = np.mean(r, axis=0)[trials-1]
        reward_rates.append(reward_rate)
    results[param] = {'values': values, 'reward_rates': reward_rates, 'max_reward': max(reward_rates)}

# Rank parameters by max_reward descending
ranked = sorted(results.items(), key=lambda x: x[1]['max_reward'], reverse=True)

print("Rank-ordered list of parameters by highest reward rate:")
for i, (param, data) in enumerate(ranked, 1):
    print(f"{i}. {param}: {data['max_reward']:.4f}")

# Plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, param in enumerate(param_names):
    ax = axes[i]
    data = results[param]
    ax.plot(data['values'], data['reward_rates'], marker='o')
    ax.set_xlabel(param)
    ax.set_ylabel('Reward Rate')
    ax.set_title(f'{param} vs Reward Rate')
    ax.grid(True)

plt.tight_layout()
# plt.savefig('parameter_sweep_plots.png')
plt.show()

# Also a bar plot for max rewards
max_rewards = [results[param]['max_reward'] for param in param_names]
plt.figure()
plt.bar(param_names, max_rewards)
plt.ylabel('Max Reward Rate')
plt.title('Max Reward Rate for Each Parameter')
# plt.savefig('max_reward_bar.png')
plt.show()