import numpy as np
from utils import revert_dict
from config import *


class StaticGame:
    """Game class for Lemon Market game."""

    def __init__(self):
        self.rewards = REWARDS
        self.probabilities = PROBABILITIES
        self.player_choices = {'seller': None, 'buyer': None}
        self.car_type = None;

    def reset(self):
        "Choose car type for a new game."
        self.car_type = ID_TO_CAR_TYPE[np.random.choice(len(ID_TO_CAR_TYPE), p=self.probabilities)]
        self.player_choices = {'seller': None, 'buyer': None}

    def step(self, player, choice):
        self.player_choices[player] = choice

    def reward(self, player=None):
        if not (self.player_choices['seller'] and self.player_choices['buyer']):
            raise ValueError('Not all choices have been made.')

        actions = [None] * 2
        for idx, player_type in enumerate(['seller', 'buyer']):
            actions[idx] = ACTION_TO_ID[player_type][self.player_choices[player_type]]
        reward = self.rewards[self.car_type][tuple(actions)]
        results = {
            'seller': reward[PLAYERS_TO_ID['seller']],
            'buyer': reward[PLAYERS_TO_ID['buyer']],
        }
        if player is None:
            return results
        else:
            return results[player]

    def observe(self, player=None):
        """Get players last action."""
        if player:
            return self.player_choices[player]
        return self.player_choices
