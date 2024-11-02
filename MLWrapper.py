import time
from typing import Any, Mapping, Tuple, TypeVar
from stable_baselines3 import PPO
from sympy import N
from Game import *
import random

import gymnasium as gym
from gymnasium import spaces

from ObservationSpaces import FULL_IMG_CHANNELS, VEC_10_ATTRIBUTES, VEC_10_UNITS, ObservationMode, get_numpy_compound_1, get_obs_space_for_obs_mode

NUM_COMMANDS = 5
MAX_SIM_STEPS = 300

TypeVar_co = TypeVar("TypeVar_co", covariant=True)

class AgentRole(Enum):
    MAIN = 1
    ALT = 2

    def other_role(self):
        if self == AgentRole.MAIN:
            return AgentRole.ALT
        else:
            return AgentRole.MAIN

@dataclass
class LocationTarget:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    @staticmethod
    def get_random_target():
        return LocationTarget(random.random() * MAP_X_DIM, random.random() * MAP_Y_DIM)

DIST_CLIP = 75

def disp_to_target_normalized(source, target):
    # Vector displacement to target clipped to the box [-1, 1] x [-1, 1]
    return np.clip([target.x - source.x, target.y - source.y], -DIST_CLIP, DIST_CLIP) / DIST_CLIP


def dist_to_target_normalized(source, target, buffer=0):
    # displacement to target clipped to [0, 1]
    disp = [target.x - source.x, target.y - source.y]
    dist = np.sqrt(np.sum(np.power(disp, 2))) - buffer
    return np.clip(dist, 0, DIST_CLIP) / DIST_CLIP

class MetaGoalType(Enum):
    AVOID_DAMAGE = 0
    GET_GOLD = 1
    DAMAGE_UNIT = 2
    LOCATION_ONLY = 3


class MetaGoal:
    def __init__(self, player: Player, goal_type: MetaGoalType, target_unit: Optional[Minion] = None, location_target: Optional[LocationTarget] = None):
        self.player = player
        self.goal_type = goal_type
        self.target_unit = target_unit
        self.location_target = location_target
        if target_unit is not None:
            self.last_target_health = target_unit.stats.health
    
    def get_reward(self, steps_to_wait, new_gold, health_loss):
        step_wait_score = steps_to_wait ** 2 / (65 ** 2) / 200
        if health_loss > 0:
            health_score = -0.1 - 0.9 * health_loss # penalty for taking any damage at all
        else:
            health_score = 0.0
        
        gold_score = new_gold / 100

        if self.goal_type == MetaGoalType.DAMAGE_UNIT and self.target_unit is not None:
            damage_score = (self.last_target_health - self.target_unit.stats.health) / self.last_target_health
            self.last_target_health = self.last_target_health
        else:
            damage_score = 0.0

        if self.location_target is not None:
            dist = dist_to_target_normalized(self.player, self.location_target, buffer=15)
            location_score = -(dist ** 2)
        else:
            location_score = 0.0

        if self.goal_type == MetaGoalType.AVOID_DAMAGE:
            health_weight = 1.0
            gold_weight = 0.0
            damage_weight = 0.0
            location_weight = 0.1
        elif self.goal_type == MetaGoalType.GET_GOLD:
            health_weight = 0.0
            gold_weight = 1.0
            damage_weight = 0.0
            location_weight = 0.1
        elif self.goal_type == MetaGoalType.DAMAGE_UNIT:
            health_weight = 0.0
            gold_weight = 0.0
            damage_weight = 1.0
            location_weight = 0.1
        elif self.goal_type == MetaGoalType.LOCATION_ONLY:
            health_weight = 0.0
            gold_weight = 0.0
            damage_weight = 0.0
            location_weight = 1.0

        composite_reward = (
            step_wait_score 
            + gold_score * gold_weight
            + health_score * health_weight
            + damage_score * damage_weight
            + location_score * location_weight
            )
        return composite_reward
    
    def goal_done(self):
        if self.goal_type == MetaGoalType.DAMAGE_UNIT and not self.target_unit.active: # type: ignore
            return True
        return False

    def get_numpy_vec_10_x2_metagoal(self):
        goal = np.zeros((2, VEC_10_ATTRIBUTES, 1))
        # Goal type, [0, 0:4, 0]. One hot for this.
        goal[0, self.goal_type.value, 0] = 1 # the value works as an index
        # Displacement to target unit location, [0, 4:6, 0].
        if self.goal_type == MetaGoalType.DAMAGE_UNIT and self.target_unit is not None:
            goal[0, 4:6, 0] = disp_to_target_normalized(self.player, self.target_unit)
        # Displacement to location target [0, 6:8, 0]
        if self.location_target is not None:
            goal[0, 6:8, 0] = disp_to_target_normalized(self.player, self.location_target)
        return goal

@dataclass
class RewardTrackingProps: #properties that are tracked and used to compute rewards over time
    last_gold: int = 0
    last_health: float = 0.0
    cumulative_reward: float = 0.0
    last_cumulative_reward: float = 0.0
    last_reward: float = 0.0
    meta_goal: Optional[MetaGoal] = None

    def get_goal_type(self):
        if self.meta_goal is not None:
            return self.meta_goal.goal_type
        else:
            return None

class MLWrapper(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, x_dim, y_dim, render_mode=None, obs_mode=ObservationMode.SIMPLE_IMG, display_elem_tracking_only=False, max_sim_steps=MAX_SIM_STEPS):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.last_rep = None # Last representation that was returned

        self.map_x_bucket = MAP_X_DIM / x_dim
        self.map_y_bucket = MAP_Y_DIM / y_dim

        self.auto_gen_meta_goals = obs_mode == ObservationMode.VEC_10_x2_MetaGoal
        
        # Tracking data
        self.tracking = {AgentRole.MAIN: RewardTrackingProps(), AgentRole.ALT: RewardTrackingProps()}

        # Gym API
        self.obs_modes= {AgentRole.MAIN: obs_mode, AgentRole.ALT: None}
        self.observation_space = self.get_obs_space()
        self.action_space = spaces.Box(-1, 1, shape=(8,)) # NUM_COMMANDS + 2 coord inputs + steps to wait input
        self.render_mode = render_mode

        self.max_sim_steps = max_sim_steps
        self.display_elem_tracking_only = display_elem_tracking_only
        self.last_frames = {AgentRole.MAIN: [], AgentRole.ALT: []}
        self.reset() # Create game with any randomness
        self.action_tracking = []
    
    def get_obs_space(self):
        obs_mode = self.obs_modes[AgentRole.MAIN]
        if obs_mode == ObservationMode.SIMPLE_IMG: return spaces.Box(0, 255, (1, self.x_dim, self.y_dim), np.uint8)
        elif obs_mode == ObservationMode.FULL_IMG: return spaces.Box(0, 255, (FULL_IMG_CHANNELS, self.x_dim, self.y_dim), np.uint8)
        elif obs_mode == ObservationMode.VEC_10: return spaces.Box(-1, 1, (VEC_10_ATTRIBUTES, VEC_10_UNITS))
        elif obs_mode == ObservationMode.VEC_10_x2: return spaces.Box(-1, 1, (2, VEC_10_ATTRIBUTES, VEC_10_UNITS))
        elif obs_mode == ObservationMode.VEC_10_x2_MetaGoal: return spaces.Box(-1, 1, (2, VEC_10_ATTRIBUTES, VEC_10_UNITS + 1)) # Allocate one extra "unit space" to store the meta goal 
        elif obs_mode == ObservationMode.COMPOUND_1: return get_obs_space_for_obs_mode(obs_mode)
        elif obs_mode == ObservationMode.NONE: return spaces.Box(-1, 1)
        assert False, f"Invalid observation mode {obs_mode}"
    
    def step_game(self, command = None, alt_command = None, skip_render = False, delay = 0):
        self.game.step(command=command, alt_command=alt_command, skip_render=skip_render, delay=delay)
        #Anything that should be called per-step
        self.gen_random_meta_goals()

    def gen_random_meta_goals(self, for_init=False):  
        if self.auto_gen_meta_goals:
            for role in self.players:
                player = self.players[role]
                tracking = self.tracking[role]
                goal = tracking.meta_goal
                should_have_goal = ObservationMode.supports_meta_goal(self.obs_modes[role]) 
                if should_have_goal and (self.game.sim.sim_step % 200 == 0 or for_init or goal is None or goal.goal_done()):
                    goal_type = random.choice([MetaGoalType.AVOID_DAMAGE, MetaGoalType.GET_GOLD])#, MetaGoalType.DAMAGE_UNIT, MetaGoalType.LOCATION_ONLY])
                    location_target = LocationTarget.get_random_target()
                    if goal_type == MetaGoalType.AVOID_DAMAGE or goal_type == MetaGoalType.GET_GOLD or goal_type == MetaGoalType.LOCATION_ONLY:
                        goal = MetaGoal(player, goal_type, location_target=location_target)
                    elif goal_type == MetaGoalType.DAMAGE_UNIT:
                        units = [unit for unit in self.game.sim.unitList if unit.team != player.team and unit.can_take_damage()]
                        weights = [1 if unit.unitType == UnitType.MINION else 4 for unit in units]
                        target = random.choices(units, weights=weights)[0]
                        goal = MetaGoal(player, goal_type, target_unit=target, location_target=location_target)
                    else:
                        goal = None
                        assert False, "Tried to generate invalid goal type"
                    tracking.meta_goal = goal
            


    def internal_step(self, command_index: int, command_dx, command_dy, sim_steps_to_wait: int):
        # Get command. Arbitrary index -> command mapping
        command = self.get_command(command_index, command_dx, command_dy, self.game.player, self.obs_modes[AgentRole.MAIN])

        self.step_game(command)

        for i in range(sim_steps_to_wait):
            self.step_game()

        return self.get_numpy()

    def get_command(self, command_index, command_dx, command_dy, player, obs_mode):
        command = None
        if obs_mode == ObservationMode.COMPOUND_1:
            # Interpret as absolute coordinates
            command_x = command_dx
            command_y = command_dy
        else:
            # Interpret as relative coordinates
            command_x = player.x + command_dx
            command_y = player.y + command_dy
        if command_index == 0:
            pass
        if command_index == 1:
            command = InputCommand(InputCommandType.MOVE, x=command_x, y=command_y)
        if command_index == 2:
            InputCommand(commandType=InputCommandType.RECALL)
        if command_index == 3:
            command = InputCommand(InputCommandType.ATTACK, x=command_x, y=command_y)
        if command_index == 4:
            command = InputCommand(InputCommandType.Q_CAST, x=command_x, y=command_y)
        return command
    
    def deltas_to_bucket(self, dx, dy):
        # smaller buckets than the coords ones
        # Center around 128 and clip
        # TODO: if agents are ever advanced enough to do more precise timing, scale up displacements to allow more precision
        dx = round(dx + 128)
        dy = round(dy + 128)
        dx = np.clip(dx, 0, 255)
        dy = np.clip(dy, 0, 255)
        return dx, dy

    def coords_to_bucket(self, in_x, in_y):
        x = round(in_x / self.map_x_bucket)
        y = round(in_y / self.map_y_bucket)
        x = np.clip(x, 0, self.x_dim - 1)
        y = np.clip(y, 0, self.y_dim - 1)
        return x, y

    def get_observer_team(self, agent_role: AgentRole): #TODO: just use the AgentRole to simplify/replace this function and the one below
        return self.players[agent_role].team

    def get_observer_player_other_player(self, agent_role: AgentRole):
        return self.players[agent_role], self.players[agent_role.other_role()]

    def get_numpy_array_form_only(self, agent_role: AgentRole = AgentRole.MAIN, obs_mode=None):
         # The "array form only" is to account for the observation modes that don't give a single array (nothing is shown for these cases)
        x = self.get_numpy(agent_role, obs_mode)
        if not isinstance(x, np.ndarray):
            return np.zeros((1, 1))
        return x

    def get_numpy(self, agent_role: AgentRole = AgentRole.MAIN, obs_mode=None):
        if obs_mode is None: obs_mode = self.obs_modes[AgentRole.MAIN]
        if obs_mode == ObservationMode.SIMPLE_IMG: return self.get_numpy_simple_img(agent_role)
        elif obs_mode == ObservationMode.FULL_IMG: return self.get_numpy_full_img(agent_role)
        elif obs_mode == ObservationMode.VEC_10: return self.get_numpy_vec_10(agent_role)
        elif obs_mode == ObservationMode.VEC_10_x2: return self.get_numpy_vec_10_x2(agent_role)
        elif obs_mode == ObservationMode.VEC_10_x2_MetaGoal: return self.get_numpy_vec_10_x2_metagoal(agent_role)
        elif obs_mode == ObservationMode.COMPOUND_1: return get_numpy_compound_1(self.players[agent_role], self.game)
        elif obs_mode == ObservationMode.NONE: return np.array([0])
        assert False, f"Invalid observation mode {obs_mode}"
    
    def get_numpy_simple_img(self, agent_role):
        #Grayscale image representation
        rep = np.zeros((1, self.x_dim, self.y_dim), dtype=np.uint8) 
        
        # # Add in events that are displayed. TODO: Add cast events as well (add loop at end, since those are more important).
        # for i in self.game.display.display_elements:
        #     elem = self.game.display.display_elements[i]
        #     x, y, _, _ = elem.get_coord_and_delta()
        #     x, y = self.coords_to_bucket(x, y)
        #     rep[0, x, y] = 15

        team = self.get_observer_team(agent_role)
        # Add in units
        for unit in self.game.sim.unitList:
            same_team = unit.team == team
            x, y = self.coords_to_bucket(unit.x, unit.y)
            if unit.unitType == UnitType.PLAYER:
                val = int(unit.unitType.value * 10 + (110 if same_team else 200))
                rep[0, max(x-1, 0):x+1, max(y-1, 0):y+1] = val
            else:
                rep[0, x, y] = int(unit.unitType.value * 10 + (60 if same_team else 150))

        return rep
    
    def get_numpy_full_img(self, agent_role):
        num_channels = FULL_IMG_CHANNELS # Just the number of attributes added below
        rep = np.zeros((num_channels, self.x_dim, self.y_dim), dtype=np.uint8) 
        team = self.get_observer_team(agent_role)
        # Add in units
        for unit in self.game.sim.unitList:
            x, y = self.coords_to_bucket(unit.x, unit.y)
            # Basic attributes and state
            rep[0:5, x, y] = self.get_basic_unit_props(team, unit)
        
        # Add in events that are displayed. TODO: Add cast events as well.
        for i in self.game.display.display_elements:
            elem = self.game.display.display_elements[i]
            x, y, dx, dy = elem.get_coord_and_delta()
            x, y = self.coords_to_bucket(x, y)
            dx, dy = self.deltas_to_bucket(dx, dy)
            rep[5, x, y] = dx
            rep[6, x, y] = dy

        return rep

    def get_numpy_vec_10(self, agent_role):
        num_channels = VEC_10_ATTRIBUTES # Just the number of attributes added below
        rep = np.zeros((num_channels, VEC_10_UNITS))
        player, other_player = self.get_observer_player_other_player(agent_role)
        team = player.team
        minions, turrets = self.game.get_vec_k_units(player.x, player.y, 10)

        units_to_embed = [player] + [other_player] + turrets + minions # Put turrets and other player first because it's best if those are in static order
        # Add in units
        for i, unit in enumerate(units_to_embed):
            # Basic attributes and state
            rep[0:5, i] = self.get_basic_unit_props_non_img(team, unit)
            rep[5:7, i] = disp_to_target_normalized(player, unit)
            if unit.target is not None:
                dx, dy, _, _ = unit.get_dist_to_target()
                rep[7:9, i] = np.clip([dx, dy], -50, 50) / 50

        # note that rep[1, 5:7, 0] as it's set in the previous loop is essentially meaningless because it's the distance from the observing player to itself.
        # So, use it to hold other data
        rep[5, 0] = self.game.sim.sim_step / 2000
        rep[6, 0] = self.cumulative_rewards[agent_role.other_role()] / 2.0

        return rep


    def get_numpy_vec_10_x2(self, agent_role):
        return self.get_numpy_vec_10_xN(agent_role, n=2)

    def get_numpy_vec_10_x2_metagoal(self, agent_role):
        base_rep = self.get_numpy_vec_10_xN(agent_role, n=2)
        goal = self.tracking[agent_role].meta_goal
        if goal is None:
            goal_rep = np.zeros((2, VEC_10_ATTRIBUTES, 1))
        else:
            goal_rep = goal.get_numpy_vec_10_x2_metagoal()
        
        return np.concatenate([base_rep, goal_rep], axis=2)

    
    def get_numpy_vec_10_xN(self, agent_role, n):
        rep = self.get_numpy_vec_10(agent_role)
        last_frames = self.last_frames[agent_role]
        if len(last_frames) == 0:
            last_frames.extend([rep] * (n - 1))
        
        stacked_rep = np.stack(last_frames + [rep], axis=0)
        last_frames.pop(0)
        last_frames.append(rep)
        return stacked_rep

    
    def get_basic_unit_props(self, main_team, unit: Minion):
        return (
            unit.unitType.value,
            10 if unit.team == main_team else 20,
            unit.state.value,
            unit.aa_state.value,
            unit.stats.health / 50
        )
    
    def get_basic_unit_props_non_img(self, observer_team, unit: Minion):
        return (
            unit.unitType.value - 2,
            -1 if unit.team == observer_team else 1,
            (unit.state.value - 3)/2,
            unit.aa_state.value - 2,
            np.clip(unit.stats.health / unit.stats.max_health, 0, 1)
        )
    
    # Gymnasium API
    def step(self, action):
        self.action_tracking.append(action)
        command_index, dx, dy, steps_to_wait = self.parse_action_array(action, self.obs_modes[AgentRole.MAIN])
        dx += (np.random.random() - 0.5) * 0.5

        obs = self.internal_step(command_index, dx, dy, steps_to_wait)

        terminated, truncated, _, _ = self.terminated_or_truncated_info()
        reward = self.get_terminated_truncated_reward(terminated, truncated, self.game.player)

        if not (terminated or truncated):
            reward = self.get_rewards_and_update_tracking(AgentRole.MAIN, steps_to_wait)


        goal_type = self.tracking[AgentRole.MAIN].get_goal_type()
        return obs, reward, terminated, truncated, {"reward": reward, "meta_goal_type": goal_type}

    def terminated_or_truncated_info(self):
        sim_steps_exceeded = self.game.sim.sim_step > self.max_sim_steps
        player_inactive = len(self.inactive_player_teams()) > 0
        base_inactive = len(self.inactive_base_teams()) > 0
        return player_inactive or base_inactive, sim_steps_exceeded, player_inactive, base_inactive
    
    def inactive_teams(self):
        return self.inactive_base_teams().union(self.inactive_player_teams())

    def inactive_base_teams(self):
        teams = set()
        if not self.game.base_A.active:
            teams.add(TEAM_A)
        if not self.game.base_B.active:
            teams.add(TEAM_B)
        return teams
    
    def inactive_player_teams(self):
        teams = set()
        if not self.game.player.active:
            teams.add(self.game.player.team)
        if not self.game.alt_player.active:
            teams.add(self.game.alt_player.team)
        return teams

    def get_terminated_truncated_reward(self, terminated, truncated, player, use_penalty=True):
        reward = 0
        if terminated:
            inactive_player = self.inactive_player_teams()
            inactive_base = self.inactive_base_teams()
            if len(inactive_player) > 0: # at least one player died
                if player.team not in inactive_player:
                    reward += 5.0 # Player didn't die, so victory
                elif len(inactive_player) == 1:           
                    reward += -5.0 # Defeat
                # If neither condition, then both players died (pretty much a draw)
            if len(inactive_base) > 0: # at least one base died
                if player.team not in inactive_base:
                    reward += 10.0 # Player's base didn't die, so victory
                elif len(inactive_base) == 1:
                    reward += -10.0 # Defeat
                # If neither condition, then both bases died (pretty much a draw)

        if (terminated or truncated) and use_penalty:
            reward -= self.tracking[AgentRole.ALT].cumulative_reward

        return reward
    
    def get_rewards_and_update_tracking(self, role, steps_to_wait):
        player = self.players[role]
        tracking = self.tracking[role]
        is_active = player.active
        new_gold = player.gold - tracking.last_gold
        if tracking.last_health > 0:
            health_loss = (player.stats.health - tracking.last_health) / tracking.last_health # Use last health rather than max health. Might want to use both in the future
        else:
            health_loss = 0 # Otherwise, this should be reflected as a termination penalty

        if tracking.meta_goal is not None:
            reward = tracking.meta_goal.get_reward(steps_to_wait, new_gold, health_loss)
        else:
            reward = self.get_rewards_default(steps_to_wait, new_gold, is_active, health_loss)

        # Update tracking
        tracking.last_gold = player.gold
        tracking.last_health = player.stats.health
        tracking.last_reward = reward
        tracking.last_cumulative_reward = tracking.cumulative_reward
        tracking.cumulative_reward += reward

        return reward

    def set_meta_goal(self, role: AgentRole, meta_goal: MetaGoal):
        self.tracking[role].meta_goal = meta_goal

    @staticmethod
    def get_rewards_default(steps_to_wait, new_gold, is_active, health_loss):
        steps_to_wait_reward = 0 if not is_active else steps_to_wait ** 2 / (65 ** 2 * 100)
        return new_gold / 500 + steps_to_wait_reward - health_loss

    @staticmethod
    def scale_steps_to_wait(val):
        val = (val + 1) / 2 # put in range (0,1)
        val = np.clip(val, 0, 1) # ensure in range
        val = np.power(val, 0.5) # scale to bias higher number of steps
        return int(val * 50 + 5) # scale out to range (5, 65)

    def parse_action_array(self, action, obs_mode):
        command_index = int(np.argmax(action[:NUM_COMMANDS]))
        if obs_mode == ObservationMode.COMPOUND_1:
            #Absolute coordinates
            dx, dy = (action[NUM_COMMANDS:NUM_COMMANDS + 2] + 1) * 50
        else:
            # Relative coordinates
            dx, dy = action[NUM_COMMANDS:NUM_COMMANDS + 2] * 50
        steps_to_wait = self.scale_steps_to_wait(action[NUM_COMMANDS + 2])
        return command_index, dx, dy, steps_to_wait

    def reset(self, seed = None, options = None):
        random.seed(seed)
        self.cumulative_rewards = {AgentRole.MAIN: 0.0, AgentRole.ALT: 0.0}
        team_to_play = TEAM_A if random.random() > 0.5 else TEAM_B
        self.game = Game(team_to_play=team_to_play, display_elem_tracking_only=self.display_elem_tracking_only)
        self.last_rep = None
        self.players = {AgentRole.MAIN: self.game.player, AgentRole.ALT: self.game.alt_player}
        self.init_tracking()
        self.gen_random_meta_goals(for_init=True)
        
        return self.get_numpy(), {}
    
    def init_tracking(self):
        for role in self.players:
            player = self.players[role]
            tracking = self.tracking[role]
            tracking.last_gold = player.gold
            tracking.last_health = player.stats.health
            tracking.cumulative_reward = 0.0
            tracking.last_cumulative_reward = 0.0
            tracking.last_reward = 0.0
            tracking.meta_goal = None

    def render(self):
        self.game.renderState()
        return None

    def close(self):
        pygame.quit()
        pygame.display.quit()

class MatchWrapper(MLWrapper):
    def __init__(self, x_dim, y_dim, render_mode=None, base_obs_mode=ObservationMode.SIMPLE_IMG, display_elem_tracking_only=False, max_sim_steps=MAX_SIM_STEPS):
        super().__init__(x_dim, y_dim, render_mode,obs_mode=base_obs_mode, display_elem_tracking_only=display_elem_tracking_only, max_sim_steps=max_sim_steps)
        self.cumulative_rewards = {AgentRole.MAIN: 0.0, AgentRole.ALT: 0.0} # Track rewards for each player
        self.last_rewards = {AgentRole.MAIN: 0.0, AgentRole.ALT: 0.0} # Track rewards for each player
        self.last_gold = {AgentRole.MAIN: 0, AgentRole.ALT: 0}
        self.last_health = {AgentRole.MAIN: 0.0, AgentRole.ALT: 0.0}
        self.next_timesteps = {AgentRole.MAIN: 0, AgentRole.ALT: 0}
        self.models: dict[AgentRole, Optional[PPO]] = {AgentRole.MAIN: None, AgentRole.ALT: None}
        self.display_infos: dict[AgentRole, str] = {AgentRole.MAIN: "", AgentRole.ALT: ""}
        self.obs_modes: dict[AgentRole, ObservationMode] = {AgentRole.MAIN: ObservationMode.SIMPLE_IMG, AgentRole.ALT: ObservationMode.SIMPLE_IMG}
        self.last_actions: dict[AgentRole, Optional[InputCommand]] = {AgentRole.MAIN: None, AgentRole.ALT: None}
        self.show_obs = False

    def set_models(self, models):
        self.models = models

    def set_display_infos(self, info):
        self.display_infos = info
    
    def set_obs_modes(self, obs_modes):
        self.obs_modes = obs_modes

    def get_timestep_to_stop(self):
        return min([self.next_timesteps[a] for a in self.next_timesteps])

    def render_font_block(self, font, y_start, string_to_render):
        text = font.render(string_to_render, True, (0, 0, 0))
        self.game.display.screen.blit(text, (0, y_start))
        y_start += text.get_rect().height
        return y_start

    def extra_display(self, render=False):
        if render:
            y_start = 0
            font = pygame.font.Font(None, 16)
            y_start = self.render_font_block(font, y_start, f"Sim step={self.game.sim.sim_step}")
            if self.show_obs:
                Z = np.abs(np.mean(self.get_numpy_array_form_only(), axis=0))
                Z = 255*Z/Z.max()
                surf = pygame.surfarray.make_surface(Z)
                self.game.display.screen.blit(surf, (0, y_start))
                y_start += surf.get_rect().height
            top_player_info, bottom_player_info = self.map_items_to_main_alt(self.display_infos)
            y_start = self.render_font_block(font, y_start, f"{top_player_info} (top player) vs")
            y_start = self.render_font_block(font, y_start, f"{bottom_player_info} (bottom player)")
            # Last actions
            top_player_last_action, bottom_player_last_action = self.map_items_to_main_alt(self.last_actions)
            y_start = self.render_font_block(font, y_start, str(top_player_last_action))
            y_start = self.render_font_block(font, y_start, str(bottom_player_last_action))
            top_tracking, bottom_tracking = self.map_items_to_main_alt(self.tracking)
            top_points, bottom_points = top_tracking.cumulative_reward, bottom_tracking.cumulative_reward
            y_start = self.render_font_block(font, y_start, f"Points: top={top_points:.2f} bottom={bottom_points:.2f}")
            top_last_rew, bottom_last_rew = top_tracking.last_reward, bottom_tracking.last_reward
            y_start = self.render_font_block(font, y_start, f"Last rewards: top={top_last_rew:.3f} bottom={bottom_last_rew:.3f}")
            top_goal, bottom_goal = top_tracking.get_goal_type(), bottom_tracking.get_goal_type()
            y_start = self.render_font_block(font, y_start, f"Goals: top={top_goal} bottom={bottom_goal}")

            for tracking in [top_tracking, bottom_tracking]:
                goal = tracking.meta_goal
                if goal is not None:
                    if goal.location_target is not None:
                        self.game.display.put_X(goal.location_target.x, goal.location_target.y, 3, (255, 0, 0))
                        self.game.display.put_line(goal.player.x, goal.player.y, goal.location_target.x, goal.location_target.y, (255, 0, 0))
                    if goal.target_unit is not None:
                        self.game.display.put_X(goal.target_unit.x, goal.target_unit.y, 3, (255, 0, 0))
                        self.game.display.put_line(goal.player.x, goal.player.y, goal.target_unit.x, goal.target_unit.y, (255, 0, 0))
            pygame.display.update()
            pygame.time.delay(50)


    def match_step(self, render_all_steps=False):
        timestep_to_stop = self.get_timestep_to_stop()
        agents = [a for a in self.next_timesteps if self.next_timesteps[a] == timestep_to_stop]

        # Run the simulation until it's at the timestep where the next agent will take an action
        while self.game.sim.sim_step < timestep_to_stop:
            self.step_game(skip_render=not render_all_steps, delay=0)
            self.extra_display(render_all_steps)

        commands: dict[AgentRole, Optional[InputCommand]] = {AgentRole.MAIN: None, AgentRole.ALT: None}
        for agent in agents:
            obs_mode = self.obs_modes[agent]
            obs = self.get_numpy(agent, obs_mode=obs_mode)
            commands[agent], steps_to_wait = self.get_command_info_for_agent(agent, obs)
            self.get_rewards_and_update_tracking(agent, steps_to_wait)
        for agent in agents:
            if commands[agent] is not None:
                self.last_actions[agent] = commands[agent]
        
        self.step_game(commands[AgentRole.MAIN], commands[AgentRole.ALT], delay=0)
        self.extra_display(render_all_steps)
    
    def map_items_to_main_alt(self, info_dict: Mapping[AgentRole, TypeVar_co]) -> Tuple[TypeVar_co, TypeVar_co]:
        # MAIN/ALT roles just correspond to whether the agent/model is controlling self.game.player or self.game.alt_player
        # It doesn't specify which team (TEAM_A - top or TEAM_B - bottom) MAIN/ALT correspond to
        # This function is used to map display items for each role to the team that they are on
        if self.game.team_to_play == TEAM_A: return info_dict[AgentRole.MAIN], info_dict[AgentRole.ALT]
        else: return info_dict[AgentRole.ALT], info_dict[AgentRole.MAIN]
    
    def run_match(self, render=False, render_all_steps=False):
        done = False
        while not done:
            self.match_step(render_all_steps)
            if render: self.game.renderState()
            terminated, truncated, player_inactive, base_inactive = self.terminated_or_truncated_info()
            if terminated or truncated:
                done = True
        for agent in self.players:
            player = self.players[agent]
            reward = self.get_terminated_truncated_reward(terminated, truncated, player, False)
            self.cumulative_rewards[agent] += reward # Don't need to account for the "other player score" penalty, because both players' scores are going to be compared (comparing a and b is the same as comparing a - b and b - a)
        
        return terminated, truncated, player_inactive, base_inactive

    def get_command_info_for_agent(self, agent: AgentRole, obs):
        player = self.players[agent]
        model = self.models[agent]
        assert model is not None
        obs_mode = self.obs_modes[agent]
        action, _ = model.predict(obs, deterministic=True) # TODO: evaluate whether this should actually be deterministic or not
        command_index, dx, dy, steps_to_wait = self.parse_action_array(action, obs_mode)
        self.next_timesteps[agent] = self.game.sim.sim_step + steps_to_wait
        return self.get_command(command_index, dx, dy, player, obs_mode), steps_to_wait

class PVPWrapper(MLWrapper):
    def __init__(self, x_dim, y_dim, enemy_model=None, render_mode=None, display_elem_tracking_only=False, obs_mode=ObservationMode.SIMPLE_IMG):
        super().__init__(x_dim, y_dim, render_mode, display_elem_tracking_only=display_elem_tracking_only, obs_mode=obs_mode)
        self.enemy_model = enemy_model
         # TODO if/when more properties are added, factor the next_timestep/last_gold out into "non terminated reward calculation info" <- this is probably a good use of the info parameter supported by gym
        self.next_enemy_timestep = 0
        self.enemy_obs_mode = ObservationMode.SIMPLE_IMG
    
    def set_enemy_info(self, model, obs_mode=ObservationMode.SIMPLE_IMG):
        self.enemy_model = model
        self.enemy_obs_mode = obs_mode
    
    def internal_step(self, command_index: int, command_dx, command_dy, sim_steps_to_wait: int):
        command = self.get_command(command_index, command_dx, command_dy, self.game.player, self.obs_modes[AgentRole.MAIN])
        alt_command = self.generate_enemy_command()
        self.step_game(command, alt_command)

        for i in range(sim_steps_to_wait):
            alt_command = self.generate_enemy_command()
            self.step_game(alt_command=alt_command)

        return self.get_numpy()
    
    def generate_enemy_command(self):
        if self.enemy_model is None:
            return None
        if self.game.sim.sim_step == self.next_enemy_timestep:
            action = self.enemy_model.predict(self.get_numpy(AgentRole.ALT, obs_mode=self.enemy_obs_mode), deterministic=True)[0] # Should this really be deterministic=True?
            command_index, dx, dy, steps_to_wait = self.parse_action_array(action, self.enemy_obs_mode)
            self.next_enemy_timestep = self.game.sim.sim_step + steps_to_wait
            self.get_rewards_and_update_tracking(AgentRole.ALT, 0) # Probably don't need to incorporate steps_to_wait reward for enemy in this case. probably reduces some noise
            return self.get_command(command_index, dx, dy, self.game.alt_player, self.enemy_obs_mode)
        else:
            return None


def test_wrapper():
    wrapper = MLWrapper(10, 10)
    for i in range(20):
        x = wrapper.internal_step(0, 0.0, 0.0, 10)
        print(list(x))

def test_env():
    # Testing that also includes the gym API functionality
    from matplotlib import pyplot as plt
    env = MLWrapper(10, 10, obs_mode=ObservationMode.VEC_10)
    print(env.observation_space.shape)
    for i in range(10):
        act = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)

        print(act)
        print(obs)
        print(obs.shape) # type: ignore
        env.render()

class DummyModel:
    def __init__(self, env):
        self.action_space = env.action_space
    def predict(self, *args, **kwargs):
        return self.action_space.sample(), None
    def save(self, *args, **kwargs):
        pass

def test_pvp_env():
    # Testing that also includes the gym API functionality
    from matplotlib import pyplot as plt
    
    model = DummyModel(MLWrapper(1,1))


    env = PVPWrapper(2, 2, model, obs_mode=ObservationMode.COMPOUND_1, display_elem_tracking_only=False)
    env.set_enemy_info(model, ObservationMode.NONE)
    print(env.observation_space.shape)
    for i in range(1000):
        act = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)

        #plt.imshow(obs[0], interpolation='nearest')
        #plt.show()

        # print(act)
        # print(obs)
        # print(obs.shape)
        env.render()
        env.game.renderState()
        if term:
            print(f"terminated ({'alt won' if not env.game.alt_player.active else 'main won'}). ", end="")
        if trunc:
            print("truncated. ", end="")
        print(f"cumulative rewards: main={env.cumulative_rewards[AgentRole.MAIN]}, alt={env.cumulative_rewards[AgentRole.ALT]}")
        if term or trunc:
            break
    val = 0
    if env.game.player.active:
        val += 1
    if env.game.alt_player.active:
        val -= 1
    return val, env.tracking[AgentRole.MAIN].cumulative_reward - env.tracking[AgentRole.ALT].cumulative_reward

def test_match_env():
    # Testing that also includes the gym API functionality
    from matplotlib import pyplot as plt

    model = DummyModel(MLWrapper(1,1))

    env = MatchWrapper(2, 2, base_obs_mode=ObservationMode.VEC_10_x2_MetaGoal, display_elem_tracking_only=False)
    env.game.use_input_tracking = True
    env.set_models({AgentRole.MAIN: model, AgentRole.ALT: model})
    env.set_obs_modes({AgentRole.MAIN: ObservationMode.VEC_10_x2_MetaGoal, AgentRole.ALT: ObservationMode.NONE})

    term, trunc, _, _ = env.run_match(render_all_steps=True, render=True)
    if term:
        print(f"terminated ({'alt won' if not env.game.alt_player.active else 'main won'}). ", end="")
    if trunc:
        print("truncated. ", end="")
    print(f"cumulative rewards: main={env.cumulative_rewards[AgentRole.MAIN]}, alt={env.cumulative_rewards[AgentRole.ALT]}")

    print(f"Total steps: {env.game.sim.sim_step}. Active statuses: main={env.game.player.active} alt={env.game.alt_player.active}")

    val = 0
    if env.game.player.active:
        val += 1
    if env.game.alt_player.active:
        val -= 1
    return (
        val,
        env.cumulative_rewards[AgentRole.MAIN] - env.cumulative_rewards[AgentRole.ALT],
        env.game.player.gold - env.game.alt_player.gold,
        env.get_terminated_truncated_reward(term, trunc, env.game.player, False) - env.get_terminated_truncated_reward(term, trunc, env.game.alt_player, False)
    )

def test_wr():
    # Make sure that a random agent v.s. other random agent has roughly 50% win rate to ensure there aren't any biases in simulation
    N = 2000
    main_win = 0
    diff = 0
    gold_diff = 0
    term_trunc_diff = 0
    for i in range(N):
        a, b, c, d = test_match_env()
        main_win += a
        diff += b
        gold_diff += c
        term_trunc_diff += d
    print(f"{main_win}, {diff / N:.5f} {gold_diff / N:.3f} {term_trunc_diff / N:.5f}")


if __name__ == "__main__":
    #test_match_env()
    test_pvp_env()