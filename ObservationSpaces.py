from CONSTANTS import MAP_X_DIM, MAP_Y_DIM
from Game import Enum, Game
from gymnasium import spaces
import numpy as np

from Units import Minion, UnitType

FULL_IMG_CHANNELS = 7
VEC_10_UNITS = 1 + 1 + 10 + 4 # Player + other player + 10 units + 4 turrets
VEC_10_ATTRIBUTES = 9

COMPOUND_1_UNIT_ATTRIBUTES = 50
COMPOUND_1_NUM_UNITS = 32
COMPOUND_1_STATE_ATTRIBUTES = 10

class ObservationMode(Enum):
    NONE = -1
    SIMPLE_IMG = 1
    FULL_IMG = 2
    VEC_10 = 3 # Vector in form [player, 10 closest non-player units, first enemy team player, all turrents]
    VEC_10_x2 = 4 # VEC_10 but with previous frame appended
    VEC_10_x2_MetaGoal = 5 # Incorporate "meta goal"
    COMPOUND_1 = 6 # First mockup of something more like alphastar used. Uses 1-hot for many values and sqrt scaling

    @staticmethod
    def supports_meta_goal(obs_mode: 'ObservationMode'):
        if obs_mode == ObservationMode.VEC_10_x2_MetaGoal:
            return True
        return False
    
class ObsModeParams:
    COMPOUND_1_COORD_IDX: int

def get_obs_space_for_obs_mode(obs_mode: ObservationMode):
    if obs_mode == ObservationMode.COMPOUND_1:
        return get_obs_space_compound_1()
    assert False, f"This function doesn't support the specified observation mode ({obs_mode})"

############### COMPOUND_1 ###############
def get_obs_space_compound_1():
    return spaces.Dict(
        {
            "state": spaces.Box(0, 1, shape=(COMPOUND_1_STATE_ATTRIBUTES,), dtype=np.float32),
            "units": spaces.Box(0, 1, shape=(COMPOUND_1_NUM_UNITS, COMPOUND_1_UNIT_ATTRIBUTES), dtype=np.float32),
        }
    )

# "one hot" type functions should take as input the representation, current index in the representation, and info to create the representation.
# The should return the index to start the next representation

def category_value_one_hot(unit_rep, current_idx, category_value: int, shift_amount, n_categories):
    # shift_amount is to account for categories that don't start at 0
    # n_categories is how many options there are in the category
    one_hot_index = current_idx + category_value - shift_amount
    unit_rep[one_hot_index] = 1
    return current_idx + n_categories # Currently 3 unit types

def sqrt_one_hot(unit_rep, current_idx, val, max_val, n_val):
    scaled_val = np.sqrt(val)
    one_hot_index = current_idx + int(np.clip(scaled_val * n_val // max_val, 0, n_val - 1))
    unit_rep[one_hot_index] = 1
    return current_idx + n_val

def coord_encode_binary(unit_rep, current_idx, coord_component):
    # Assumes coord_component has been scaled to the range of [0,1]
    x = int(coord_component * 256)
    unit_rep[current_idx] = x % 2
    unit_rep[current_idx + 1] = (x // 2) % 2
    unit_rep[current_idx + 2] = (x // 4) % 2
    unit_rep[current_idx + 3] = (x // 8) % 2
    unit_rep[current_idx + 4] = (x // 16) % 2
    unit_rep[current_idx + 5] = (x // 32) % 2
    unit_rep[current_idx + 6] = (x // 64) % 2
    unit_rep[current_idx + 7] = (x // 128) % 2
    return current_idx + 8

def add_one_value(unit_rep, current_idx, val):
    unit_rep[current_idx] = val
    return current_idx + 1

def get_numpy_for_unit_compound_1(unit_obs, unit: Minion, maxX, maxY, is_agent_controlled):
    global COMPOUND_1_COORD_IDX

    # is_agent_controlled is whether the unit in question is going be controlled by the agent observing the observation that's being created here
    idx = 0
    idx = add_one_value(unit_obs, idx, is_agent_controlled) # Controlled by agent
    idx = category_value_one_hot(unit_obs, idx, unit.unitType.value, 1, 3) # Unit Type
    idx = category_value_one_hot(unit_obs, idx, unit.team, 1, 2) # Unit Team
    idx = category_value_one_hot(unit_obs, idx, unit.state.value, 1, 5) # Unit State
    idx = sqrt_one_hot(unit_obs, idx, unit.stats.health, 5000, 10) # Health
    idx = add_one_value(unit_obs, idx, unit.stats.health / unit.stats.max_health) # Max health fraction
    idx = sqrt_one_hot(unit_obs, idx, unit.stats.armor, 300, 10) # Armor
    
    idx = add_one_value(unit_obs, idx, np.clip(unit.x / maxX, 0, 1)) # X coordinate
    idx = add_one_value(unit_obs, idx, np.clip(unit.y / maxY, 0, 1)) # Y coordinate
    idx = coord_encode_binary(unit_obs, idx, np.clip(unit.x / maxX, 0, 1)) # X coordinate binary
    idx = coord_encode_binary(unit_obs, idx, np.clip(unit.y / maxY, 0, 1)) # Y coordinate binary

    assert idx == COMPOUND_1_UNIT_ATTRIBUTES, "Wrong number of attributes set"
    return unit_obs

def get_state_compound_1(state_obs, game: Game, team):
    idx = 0
    idx = sqrt_one_hot(state_obs, idx, game.sim.sim_step, 1000, 8) # TODO: Once agent is training on full game, make sure the maximum here is high enough to distinguish the "late game"
    idx = category_value_one_hot(state_obs, idx, team, 1, 2) # Agent Team
    return state_obs

def get_numpy_compound_1(player, game: Game):
    units = game.sim.unitList[:COMPOUND_1_NUM_UNITS] #TODO: Make sure the most relevant units are included
    units_obs = np.zeros((COMPOUND_1_NUM_UNITS, COMPOUND_1_UNIT_ATTRIBUTES))
    for i, unit in enumerate(units):
        get_numpy_for_unit_compound_1(units_obs[i], unit, MAP_X_DIM, MAP_Y_DIM, unit.uid == player.uid)
    state_obs = np.zeros(COMPOUND_1_STATE_ATTRIBUTES)
    get_state_compound_1(state_obs, game, player.team)
    return {"state": state_obs, "units": units_obs}