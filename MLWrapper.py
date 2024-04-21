import time
from typing import Any
from stable_baselines3 import PPO
from Game import *
import random

import gymnasium as gym
from gymnasium import spaces

NUM_COMMANDS = 5
MAX_SIM_STEPS = 1000

FULL_IMG_CHANNELS = 7

VEC_10_UNITS = 1 + 1 + 10 + 4 # Player + other player + 10 units + 4 turrets
VEC_10_ATTRIBUTES = 7

class AgentRole(Enum):
    MAIN = 1
    ALT = 2

class ObservationMode(Enum):
    SIMPLE_IMAGE = 1
    FULL_IMAGE = 2
    VECTOR_10 = 3 # Vector in form [player, 10 closest non-player units, first enemy team player, all turrents]

class MLWrapper(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, x_dim, y_dim, sim_step_randomness_scale = 3, render_mode=None, obs_mode=ObservationMode.SIMPLE_IMAGE, display_elem_tracking_only=False, max_sim_steps=MAX_SIM_STEPS):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.sim_step_randomness_scale = sim_step_randomness_scale # Amount of randomness injected into the number of sim steps to wait for next action
        self.last_rep = None # Last representation that was returned

        self.map_x_bucket = MAP_X_DIM / x_dim
        self.map_y_bucket = MAP_Y_DIM / y_dim

        # Gym API
        self.obs_mode = obs_mode
        self.observation_space = self.get_obs_space()
        self.action_space = spaces.Box(-1, 1, shape=(8,)) # NUM_COMMANDS + 2 coord inputs + steps to wait input
        self.render_mode = render_mode

        self.max_sim_steps = max_sim_steps
        self.display_elem_tracking_only = display_elem_tracking_only
        self.reset() # Create game with any randomness

        # Tracking data
        self.last_gold = self.game.player.gold
        self.cumulative_rewards = {AgentRole.MAIN: 0.0}
        self.end_of_episode_penalty = 0.0 # Mostly for use in the PVP case
    
    def get_obs_space(self):
        if self.obs_mode == ObservationMode.SIMPLE_IMAGE: return spaces.Box(0, 255, (1, self.x_dim, self.y_dim), np.uint8)
        elif self.obs_mode == ObservationMode.FULL_IMAGE: return spaces.Box(0, 255, (FULL_IMG_CHANNELS, self.x_dim, self.y_dim), np.uint8)
        elif self.obs_mode == ObservationMode.VECTOR_10: return spaces.Box(-1, 1, (VEC_10_ATTRIBUTES, VEC_10_UNITS))
        assert False, f"Invalid observation mode {self.obs_mode}"
    
    def internal_step(self, command_index: int, command_dx, command_dy, sim_steps_to_wait: int):
        # Get command. Arbitrary index -> command mapping
        command = self.get_command(command_index, command_dx, command_dy, self.game.player)

        self.game.step(command)

        for i in range(sim_steps_to_wait + random.randint(-self.sim_step_randomness_scale, self.sim_step_randomness_scale)):
            self.game.step()

        return self.get_numpy()

    def get_command(self, command_index, command_dx, command_dy, player):
        command = None
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

    def get_observer_team(self, for_main):
        return self.get_observer_player_other_player(for_main)[0].team

    def get_observer_player_other_player(self, for_main):
        if for_main:
            return self.game.player, self.game.alt_player
        else:
            return self.game.alt_player, self.game.player

    def get_numpy(self, for_main = True, obs_mode=None):
        if obs_mode is None: obs_mode = self.obs_mode
        if obs_mode == ObservationMode.SIMPLE_IMAGE: return self.get_numpy_simple_img(for_main)
        elif obs_mode == ObservationMode.FULL_IMAGE: return self.get_numpy_full_img(for_main)
        elif obs_mode == ObservationMode.VECTOR_10: return self.get_numpy_vec_10(for_main)
        assert False, f"Invalid observation mode {obs_mode}"
    
    def get_numpy_simple_img(self, for_main):
        #Grayscale image representation
        rep = np.zeros((1, self.x_dim, self.y_dim), dtype=np.uint8) 
        
        # Add in events that are displayed. TODO: Add cast events as well (add loop at end, since those are more important).
        for i in self.game.display.display_elements:
            elem = self.game.display.display_elements[i]
            x, y, _, _ = elem.get_coord_and_delta()
            x, y = self.coords_to_bucket(x, y)
            rep[0, x, y] = 15

        team = self.get_observer_team(for_main)
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
    
    def get_basic_unit_props(self, main_team, unit: Minion):
        return (
            unit.unitType.value,
            10 if unit.team == main_team else 20,
            unit.state.value,
            unit.aa_state.value,
            unit.stats.health / 50
        )
    
    def get_basic_unit_props_non_img(self, main_team, unit: Minion):
        return (
            unit.unitType.value - 1,
            -1 if unit.team == main_team else 1,
            unit.state.value - 1,
            unit.aa_state.value - 1,
            np.clip(unit.stats.health / 3000, 0, 1)
        )
    
    def get_numpy_full_img(self, for_main):
        num_channels = FULL_IMG_CHANNELS # Just the number of attributes added below
        rep = np.zeros((num_channels, self.x_dim, self.y_dim), dtype=np.uint8) 
        team = self.get_observer_team(for_main)
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

    def get_numpy_vec_10(self, for_main):
        num_channels = FULL_IMG_CHANNELS # Just the number of attributes added below
        rep = np.zeros((num_channels, VEC_10_UNITS))
        player, other_player = self.get_observer_player_other_player(for_main)
        team = player.team
        minions, turrets = self.game.sim.get_vec_k_units(player.x, player.y, 10, team)

        # Add in units
        for i, unit in enumerate([player] + [other_player] + minions + turrets):
            # Basic attributes and state
            rep[0:5, i] = self.get_basic_unit_props_non_img(team, unit)
            if unit.target is not None:
                dx, dy, _, _ = unit.get_dist_to_target()
                rep[5:7, i] = np.clip([dx, dy], -50, 50) / 50

        return rep
    
    # Gymnasium API
    def step(self, action):
        command_index, dx, dy, steps_to_wait = self.parse_action_array(action)
        dx += (np.random.random() - 0.5) * 0.5

        obs = self.internal_step(command_index, dx, dy, steps_to_wait)

        terminated, truncated = self.terminated_or_truncated_info()
        reward = self.get_terminated_truncated_reward(terminated, truncated, self.game.player)

        if not (terminated or truncated):
            new_gold = self.game.player.gold - self.last_gold
            reward = self.get_rewards(steps_to_wait, new_gold) # Reinforce eliminating enemies and longer times between actions
            self.last_gold = self.game.player.gold
            self.cumulative_rewards[AgentRole.MAIN] += reward

        return obs, reward, terminated, truncated, {}

    def terminated_or_truncated_info(self):
        return (not self.game.nexus_A.active) or (not self.game.nexus_B.active), self.game.sim.sim_step > self.max_sim_steps

    def get_terminated_truncated_reward(self, terminated, truncated, player):
        reward = 0
        if not self.game.nexus_A.active: # TODO: refactor so the winning team is returned by terminated_or_truncated and passed/used here
            won = player.team == TEAM_B
        elif not self.game.nexus_B.active:
            won = player.team == TEAM_A
        if terminated:
            if won:
                reward = 200 - 5 * self.game.sim.sim_step / self.max_sim_steps # player on team B by default, so victory
            else:            
                reward = -200 # Defeat=

        if terminated or truncated:
            reward -= self.end_of_episode_penalty

        return reward

    def get_rewards(self, steps_to_wait, new_gold):
        return new_gold / 20 + steps_to_wait ** 2 / 900

    def parse_action_array(self, action):
        command_index = int(np.argmax(action[:NUM_COMMANDS]))
        dx, dy = action[NUM_COMMANDS:NUM_COMMANDS + 2] * 50
        steps_to_wait = int(np.clip(action[NUM_COMMANDS + 2] * 10 + 20, 10, 30))
        return command_index, dx, dy, steps_to_wait

    def reset(self, seed = None, options = None):
        random.seed(seed)
        team_to_play = TEAM_A if random.random() > 0.5 else TEAM_B
        self.game = Game(team_to_play=team_to_play, display_elem_tracking_only=self.display_elem_tracking_only)
        self.last_rep = None
        return self.get_numpy(), {}
    
    def render(self):
        self.game.renderState()
        print("Rendered")
        return None

    def close(self):
        pygame.quit()
        pygame.display.quit()

class MatchWrapper(MLWrapper):
    def __init__(self, x_dim, y_dim, sim_step_randomness_scale=3, render_mode=None, base_obs_mode=ObservationMode.SIMPLE_IMAGE, display_elem_tracking_only=False, max_sim_steps=MAX_SIM_STEPS):
        super().__init__(x_dim, y_dim, sim_step_randomness_scale, render_mode,obs_mode=base_obs_mode, display_elem_tracking_only=display_elem_tracking_only, max_sim_steps=max_sim_steps)
        self.cumulative_rewards = {AgentRole.MAIN: 0.0, AgentRole.ALT: 0.0} # Track rewards for each player
        self.next_timesteps = {AgentRole.MAIN: 0, AgentRole.ALT: 0}
        self.last_gold = {AgentRole.MAIN: 0, AgentRole.ALT: 0}
        self.last_sim_steps_to_wait = {AgentRole.MAIN: 0, AgentRole.ALT: 0}
        self.models: dict[AgentRole, Optional[PPO]] = {AgentRole.MAIN: None, AgentRole.ALT: None}
        self.display_infos: dict[AgentRole, str] = {AgentRole.MAIN: "", AgentRole.ALT: ""}
        self.obs_modes: dict[AgentRole, ObservationMode] = {AgentRole.MAIN: ObservationMode.SIMPLE_IMAGE, AgentRole.ALT: ObservationMode.SIMPLE_IMAGE}
        self.players = {AgentRole.MAIN: self.game.player, AgentRole.ALT: self.game.alt_player}
        self.last_actions = {AgentRole.MAIN: None, AgentRole.ALT: None}
        self.show_obs = False

    def set_models(self, models):
        self.models = models

    def set_display_infos(self, info):
        self.display_infos = info
    
    def set_obs_modes(self, obs_modes):
        self.obs_modes = obs_modes

    def get_timestep_to_stop(self):
        return min([self.next_timesteps[a] for a in self.next_timesteps])

    def extra_display(self, render=False):
        if render:
            y_start = 0
            if self.show_obs:
                Z = np.abs(np.mean(self.get_numpy(), axis=0))
                Z = 255*Z/Z.max()
                surf = pygame.surfarray.make_surface(Z)
                self.game.display.screen.blit(surf, (0, y_start))
                y_start += surf.get_rect().height
            font = pygame.font.Font(None, 16)
            top_player_info, bottom_player_info = self.map_items_to_main_alt(self.display_infos)
            text = font.render(f"{top_player_info} (top player) vs", True, (0, 0, 0))
            self.game.display.screen.blit(text, (0, y_start))
            y_start += text.get_rect().height
            text = font.render(f"{bottom_player_info} (bottom player)", True, (0, 0, 0))
            self.game.display.screen.blit(text, (0, y_start))
            y_start += text.get_rect().height
            # Last actions
            top_player_last_action, bottom_player_last_action = self.map_items_to_main_alt(self.last_actions)
            text = font.render(str(top_player_last_action), True, (0, 0, 0))
            self.game.display.screen.blit(text, (0, y_start))
            y_start += text.get_rect().height
            text = font.render(str(bottom_player_last_action), True, (0, 0, 0))
            self.game.display.screen.blit(text, (0, y_start))
            y_start += text.get_rect().height
            pygame.display.update()
            pygame.time.delay(50)


    def match_step(self, render_all_steps=False):
        timestep_to_stop = self.get_timestep_to_stop()
        agents = [a for a in self.next_timesteps if self.next_timesteps[a] == timestep_to_stop]

        # Run the simulation until it's at the timestep where the next agent will take an action
        
        while self.game.sim.sim_step < timestep_to_stop:
            self.game.step(skip_render=not render_all_steps, delay=0)
            self.extra_display(render_all_steps)


        commands: dict[AgentRole, Optional[InputCommand]] = {AgentRole.MAIN: None, AgentRole.ALT: None}
        for agent in agents:
            obs_mode = self.obs_modes[agent]
            obs = self.get_numpy(for_main=agent==AgentRole.MAIN, obs_mode=obs_mode)
            commands[agent], steps_to_wait = self.get_command_info_for_agent(agent, obs)
            new_gold = self.players[agent].gold - self.last_gold[agent]
            self.cumulative_rewards[agent] += self.get_rewards(steps_to_wait, new_gold)
            self.last_gold[agent] = self.players[agent].gold
            self.last_sim_steps_to_wait[agent] = steps_to_wait
        self.last_actions = commands
        
        self.game.step(commands[AgentRole.MAIN], commands[AgentRole.ALT], delay=0)
        self.extra_display(render_all_steps)
    
    def map_items_to_main_alt(self, info_dict):
        # MAIN/ALT roles just correspond to whether the agent/model is controlling self.game.player or self.game.alt_player
        # It doesn't specific which team (TEAM_A - top or TEAM_B - bottom) MAIN/ALT correspond to
        # This function is used to map display items for each role to the team that they are on
        if self.game.team_to_play == TEAM_A: return info_dict[AgentRole.MAIN], info_dict[AgentRole.ALT]
        else: return info_dict[AgentRole.ALT], info_dict[AgentRole.MAIN]
    
    def run_match(self, render=False, render_all_steps=False):
        done = False
        while not done:
            self.match_step(render_all_steps)
            if render: self.game.renderState()
            terminated, truncated = self.terminated_or_truncated_info()
            if terminated or truncated:
                done = True
        for agent in self.players:
            player = self.players[agent]
            reward = self.get_terminated_truncated_reward(terminated, truncated, player)
            self.cumulative_rewards[agent] += reward # Don't need to account for the "other player score" penalty, because both players' scores are going to be compared (comparing a and b is the same as comparing a - b and b - a)

            
    def get_command_info_for_agent(self, agent: AgentRole, obs):
        player = self.players[agent]
        model = self.models[agent]
        assert model is not None
        action, _ = model.predict(obs, deterministic=False) # TODO: evaluate whether this should actually be deterministic or not
        command_index, dx, dy, steps_to_wait = self.parse_action_array(action)
        self.next_timesteps[agent] = self.game.sim.sim_step + steps_to_wait
        return self.get_command(command_index, dx, dy, player), steps_to_wait

class PVPWrapper(MLWrapper):
    def __init__(self, x_dim, y_dim, enemy_model=None, sim_step_randomness_scale=3, render_mode=None, display_elem_tracking_only=False, obs_mode=ObservationMode.SIMPLE_IMAGE):
        super().__init__(x_dim, y_dim, sim_step_randomness_scale, render_mode, display_elem_tracking_only=display_elem_tracking_only, obs_mode=obs_mode)
        self.enemy_model = enemy_model
         # TODO if/when more properties are added, factor the next_timestep/last_gold out into "non terminated reward calculation info" <- this is probably a good use of the info parameter supported by gym
        self.next_enemy_timestep = 0
        self.enemy_last_gold = 0
        self.enemy_last_simsteps_to_wait = 0
        self.cumulative_rewards = {AgentRole.MAIN: 0.0, AgentRole.ALT: 0.0}
        self.enemy_obs_mode = ObservationMode.SIMPLE_IMAGE
    
    def set_enemy_info(self, model, obs_mode=ObservationMode.SIMPLE_IMAGE):
        self.enemy_model = model
        self.enemy_obs_mode = obs_mode
    
    def internal_step(self, command_index: int, command_dx, command_dy, sim_steps_to_wait: int):
        command = self.get_command(command_index, command_dx, command_dy, self.game.player)
        alt_command = self.generate_enemy_command()
        self.game.step(command, alt_command)

        for i in range(sim_steps_to_wait + random.randint(-self.sim_step_randomness_scale, self.sim_step_randomness_scale)):
            alt_command = self.generate_enemy_command()
            self.game.step(alt_command=alt_command)

        return self.get_numpy()
    
    def generate_enemy_command(self):
        if self.enemy_model is None:
            return
        if self.game.sim.sim_step == self.next_enemy_timestep:
            action = self.enemy_model.predict(self.get_numpy(for_main=False, obs_mode=self.enemy_obs_mode), deterministic=True)[0]
            command_index, dx, dy, steps_to_wait = self.parse_action_array(action)
            self.next_enemy_timestep = self.game.sim.sim_step + steps_to_wait
            new_gold = self.game.alt_player.gold - self.enemy_last_gold
            self.cumulative_rewards[AgentRole.ALT] += self.get_rewards(self.enemy_last_simsteps_to_wait, new_gold)
            self.enemy_last_gold = self.game.alt_player.gold
            self.enemy_last_simsteps_to_wait = steps_to_wait
            self.end_of_episode_penalty = self.cumulative_rewards[AgentRole.ALT]
            return self.get_command(command_index, dx, dy, self.game.alt_player)
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
    env = MLWrapper(10, 10, obs_mode=ObservationMode.VECTOR_10)
    print(env.observation_space.shape)
    for i in range(10):
        act = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)

        print(act)
        print(obs)
        print(obs.shape)
        env.render()


def test_pvp_env():
    # Testing that also includes the gym API functionality
    from matplotlib import pyplot as plt
    
    class DummyModel:
        def __init__(self):
            self.action_space = spaces.Box(-50, 50, shape=(8,))
        def predict(self, *args):
            return self.action_space.sample(), None
    env = PVPWrapper(2, 2, DummyModel())
    print(env.observation_space.shape)
    for i in range(10):
        act = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)

        #plt.imshow(obs[0], interpolation='nearest')
        #plt.show()

        print(act)
        print(obs)
        print(obs.shape)
        env.render()

if __name__ == "__main__":
    test_env()