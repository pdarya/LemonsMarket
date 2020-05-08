import matplotlib. pyplot as plt
import numpy as np
import pandas as pd


def revert_dict(input_dict):
    return dict([(value, key) for key, value in input_dict.items()])

def get_action(qvals, player_lambda):
    """Choose action using softmax"""

    prob = 1 / (1 + np.exp(player_lambda * (qvals[1] - qvals[0])))
    current_probs = [prob, 1 - prob]
    action_id = np.random.choice(2, p=current_probs)
    return action_id


class SimpleModel(nn.Module):
    def __init__(self, n_features, hidden_size=16):
        super().__init__()
        self.linear_1 = nn.Linear(n_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, 1)

    def forward(self, features):
        output = nn.functional.relu(self.linear_1(features))
        output = nn.functional.relu(self.linear_2(output))
        output = self.linear_3(output)
        return output


def plot_qvalues(qvalues, size, eps=None, rewards=None):
    """
        Function to plot q-values (or normed q-values) for each round (or iteration).
        Probability of random action for each player in each round can be displayed.
    """
    plots = [
        ('seller', 'lemon'),
        ('seller', 'peach'),
        ('buyer', 'low_price'),
        ('buyer', 'high_price'),
    ]
    actions = {
        'seller': ['high price', 'low price'],
        'buyer': ['buy', 'do not buy'],
    }

    if eps or rewards:
        f, axarr = plt.subplots(2, 3, figsize=(18, 10))
    else:
        f, axarr = plt.subplots(2, 2, figsize=(18, 10))

    for idx, (player, obs) in enumerate(plots):
        for action_id in range(2):
            axarr[idx // 2, idx % 2].plot(
                range(size),
                np.array(qvalues[player][obs])[:, action_id],
                label=actions[player][action_id])
            axarr[idx // 2, idx % 2].legend(fontsize=12)
            axarr[idx // 2, idx % 2].set_title(f'{player} {obs}', fontsize=12)
            axarr[idx // 2, idx % 2].grid()

    if eps or rewards:
        for idx, player_type in enumerate(['seller', 'buyer']):
            if eps:
                axarr[idx, 2].plot(range(size), eps[player_type])
                axarr[idx, 2].set_title(f'epsilon for {player_type}', fontsize=12)
            elif rewards:
                axarr[idx, 2].plot(range(size), rewards[player_type])
                axarr[idx, 2].set_title(f'rewards for {player_type}', fontsize=12)
            axarr[idx, 2].grid()


def plot_trust_honesty(stats, size, rolling_num=20, returns_stats=True):
    trust = pd.Series((stats[:, 2] == 0).astype(int)).rolling(rolling_num).mean()
    honesty = pd.Series(np.logical_xor(stats[:, 0], stats[:, 1]).astype(int)).rolling(rolling_num).mean()
    lemon_honesty = (
        pd.Series(np.logical_and((stats[:, 0] == 0), (stats[:, 1] == 1))).rolling(rolling_num).sum() /
        pd.Series((stats[:, 0] == 0).astype(int)).rolling(rolling_num).sum()
    )

    f, axarr = plt.subplots(2, 1, figsize=(12, 8))
    axarr[0].plot(range(size), trust, color='blue')
    axarr[0].set_title('trust', fontsize=14)
    axarr[0].grid()

    axarr[1].plot(range(size), honesty, color='green')
    axarr[1].set_title('honesty', fontsize=14)
    axarr[1].grid()

    print('[INFO] %successfull deals:', np.sum(stats[:, 2] == 0) / len(stats))
    if returns_stats:
        return trust, honesty, lemon_honesty
