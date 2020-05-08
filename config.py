import numpy as np
from utils import revert_dict


REWARDS = {
    'lemon': np.array([[[10, -9], [0, 0]], [[0.5, 0.5], [0, 0]]]),
    'peach': np.array([[[1, 1], [0, 0]], [[-8.5, 10.5], [0, 0]]]),
}
PROBABILITIES = [0.25, 0.75]
POSSIBLE_STATES = {
    'seller': ('lemon', 'peach'),
    'buyer': ('high_price', 'low_price'),
}
STATE_TO_ID = {'lemon': 0, 'peach': 1, 'high_price': 0, 'low_price': 1}

ACTION_TO_ID = {
    'seller': {'high_price': 0, 'low_price': 1},
    'buyer': {'buy': 0, 'not_buy': 1},
}
ID_TO_ACTION = {
    'seller': {0: 'high_price', 1: 'low_price'},
    'buyer': {0: 'buy', 1: 'not_buy'},
}

PLAYERS_TO_ID = {
    'seller': 0,
    'buyer': 1,
}
ID_TO_PLAYER = revert_dict(PLAYERS_TO_ID)

CAR_TYPE_TO_ID = {
    'lemon': 0,
    'peach': 1,
}
ID_TO_CAR_TYPE = revert_dict(CAR_TYPE_TO_ID)

# features for nn training
def compute_features_mappings():
    feature_to_id = {}
    feature_to_name = {}

    counter = 0
    for car_type in CAR_TYPE_TO_ID.keys():
        for price in ACTION_TO_ID['seller'].keys():
            for deal in ACTION_TO_ID['buyer'].keys():
                feature_to_id[(car_type, price, deal)] = counter
                counter += 1
                feature_to_name[(car_type, price, deal)] = f'{car_type}-{price}-{deal}'

    return feature_to_id, feature_to_name, revert_dict(feature_to_id)

FEATURE_TO_ID, FEATURE_TO_NAME, ID_TO_FEATURE = compute_features_mappings()
