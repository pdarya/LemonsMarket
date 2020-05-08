from collections import defaultdict
from warnings import warn
import numpy as np
import torch
import torch.nn as nn
from config import *
from utils import get_action


class EpsGreedyAgent:
    """Class representing agent strategy"""
    def __init__(self, name, qval=defaultdict(lambda: np.zeros(2)),
                 epsilon=0.8, decay=0.9, min_epsilon=0.01, lr=0.1, save_history=False):
        self.player_type = name  # either seller or buyer
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.lr = lr
        self.qval = qval
        self.save_history = save_history
        self.history = defaultdict(list)

    def choose_action(self, obs):
        if np.random.uniform() <= self.epsilon:
            # exploration
            self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
            chosen_action = ID_TO_ACTION[self.player_type][np.random.choice(2)]
        else:
            chosen_action = ID_TO_ACTION[self.player_type][np.argmax(self.qval[obs])]

        if self.save_history:
            self.history['states'].append(obs)
            self.history['actions'].append(chosen_action)
        return chosen_action

    def update(self, obs, action, next_obs, reward):
        """One step update"""
        # obs for seller is lemon or peach
        # obs for buyer is high_price or low_price
        action_id = ACTION_TO_ID[self.player_type][action]
        self.qval[obs][action_id] = (
            self.qval[obs][action_id] +
            self.lr * (reward + np.max(self.qval[next_obs]) - self.qval[obs][action_id])
        )

    def update_policy(self, gamma=0.99):
        """Update based on one round history [Q(s,a) <- avg(Returns(s,a))]"""
        returns = list(self.history['rewards'])
        for idx in range(2, len(returns) + 1):
            returns[-idx] = returns[-idx] + gamma * returns[-idx + 1]

        returns_stats = defaultdict(list)
        for idx in range(len(self.history)):
            returns_stats[(self.history['states'][idx], self.history['actions'][idx])].append(returns[idx])

        for state in POSSIBLE_STATES[self.player_type]:
            for action in ACTION_TO_ID[self.player_type].keys():
                action_id = ACTION_TO_ID[self.player_type][action]
                if len(returns_stats[(state, action)]) > 0:
                    self.qval[state][action_id] = np.mean(returns_stats[(state, action)])
                else:
                    warn(f'There are no examples for {state}, {action} to compute new q-value.')

    def clear_history(self):
        self.history = defaultdict(list)

    def observe_reward(self, reward):
        self.history['rewards'].append(reward)


class BolzmanAgent(EpsGreedyAgent):
    def __init__(self, name, player_lambda, qval=defaultdict(lambda: np.zeros(2)),
                 lr=0.01, save_history=True):
        super().__init__(name=name, qval=qval, lr=lr, save_history=save_history)
        self.player_lambda = player_lambda
        self.probs = defaultdict(lambda: np.zeros(2))

    def choose_action(self, obs):
        action_id = get_action(self.qval[obs], self.player_lambda)
        self.probs[obs] = np.array(current_probs)
        chosen_action = ID_TO_ACTION[self.player_type][action_id]

        if self.save_history:
            self.history['states'].append(obs)
            self.history['actions'].append(chosen_action)
        return chosen_action


class DQNAgent:
    def __init__(self, name, model, player_lambda, gamma=0.9,
                 qval=defaultdict(lambda: np.zeros(2)), save_history=True):
        self.player_type = name
        self.qval = qval
        self.save_history = save_history
        self.history = defaultdict(list)
        self.model = model # torch model
        self.player_lambda = player_lambda
        self.rounds = 0
        self.gamma = gamma
        self.features = [[0] * len(FEATURE_TO_ID)]

    def choose_action(self, obs):
        features = []
        for action_id in range(2):
            features.append(self.features[-1] + [STATE_TO_ID[obs], action_id])
        self.qval[obs] = self.model(torch.tensor(features, dtype=torch.float32)).data.numpy().ravel()

        action_id = get_action(self.qval[obs], self.player_lambda)
        chosen_action = ID_TO_ACTION[self.player_type][action_id]

        if self.save_history:
            self.history['states'].append(obs)
            self.history['actions'].append(chosen_action)
            self.rounds += 1
        return chosen_action

    def clear_history(self):
        self.history = defaultdict(list)
        self.features = []

    def observe_reward(self, reward):
        self.history['rewards'].append(reward)

    def observe_game_result(self, result):
        # result = (car_type, price, deal)
        new_features = list(self.features[-1])
        new_features[FEATURE_TO_ID[result]] += 1
        self.features.append(new_features)

    def get_train_data(self):
        # compute targets
        returns = list(self.history['rewards'])
        for idx in range(2, len(returns) + 1):
            returns[-idx] = returns[-idx] + self.gamma * returns[-idx + 1]

        # compute features
        features = np.array(self.features) # TODO: implement
        states = list(map(lambda x: STATE_TO_ID[x], self.history['states']))
        actions = list(map(lambda x: ACTION_TO_ID[self.player_type][x], self.history['actions']))
        all_features = np.hstack((features[:-1], np.array(states).reshape(-1, 1), np.array(actions).reshape(-1, 1)))
        assert len(returns) == len(all_features), 'Features and target have different shape.'
        return all_features, np.array(returns)
