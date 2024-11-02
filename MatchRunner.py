import random
from time import time
from turtle import up
from typing import Optional, Union
import numpy as np
from stable_baselines3 import PPO, A2C, TD3, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import pickle
import multiprocessing
import torch as th
import matplotlib.pyplot as plt

from CustomArchitecture import add_custom_feature_extractor_glo_rep_only
from MLWrapper import MAX_SIM_STEPS, AgentRole, DummyModel, MatchWrapper, PVPWrapper
from ObservationSpaces import ObservationMode
from Utils import get_enum_val_str

X_DIM = Y_DIM = 50

WINRATE_UPDATE_RATE = 0.01

def path_generator(algorithm: str, obs_mode: ObservationMode, extra_descriptor=None):
    if extra_descriptor is not None:
        extra_str = "_" + extra_descriptor
    else:
        extra_str = ""
    return (
        f"./agent_collections/{get_enum_val_str(obs_mode).lower()}_{algorithm.lower()}{extra_str}_collection",
        f"./saved_models/{get_enum_val_str(obs_mode).lower()}_{algorithm.lower()}{extra_str}_arena"
    )

SUPPORTED_MODELS = Union[PPO, SAC, A2C, TD3, DDPG]

def get_train_env(obs_mode=ObservationMode.SIMPLE_IMG, display_elem_tracking_only=False):
    env = PVPWrapper(X_DIM, Y_DIM, render_mode="human", display_elem_tracking_only=display_elem_tracking_only, obs_mode=obs_mode)
    #venv = DummyVecEnv([lambda: env for i in range(4)])
    return env

def get_eval_env(obs_mode=ObservationMode.SIMPLE_IMG, render=False, max_sim_steps=MAX_SIM_STEPS):
    env = MatchWrapper(X_DIM, Y_DIM, render_mode="human", base_obs_mode=obs_mode, display_elem_tracking_only=not render, max_sim_steps=max_sim_steps)
    return env

class Agent:
    def __init__(self, agent_id, model_file_base_path, algorithm = "PPO", obs_mode = ObservationMode.SIMPLE_IMG, model = None, policy_kwargs = None, model_kwargs = None):
        self.agent_id = agent_id
        self.model_file_base_path = model_file_base_path
        self.algorithm = algorithm
        self.win_rate = 0.5
        self.num_steps_trained = 0
        self.win_rate_history = [] # List of (<training steps>, <winrate at training steps>) tuples
        self.version = 0

        self.obs_mode = obs_mode
        #Ensure that there's always a model stored
        if model is None: model = self.make_model(policy_kwargs=policy_kwargs, model_kwargs=model_kwargs)

        model.save(self.get_model_path())
        self.model = model

    def __str__(self) -> str:
        obs_str = str(self.obs_mode).split(".")[1].split(":")[0]
        return f"Agent {self.agent_id} <vers={self.get_version()} alg={self.algorithm} obs_mode={obs_str} WR={self.win_rate:.2f}>"# train_steps={self.num_steps_trained}>"

    def get_alg(self):
        if self.algorithm == "PPO":
            return PPO
        elif self.algorithm == "SAC":
            return SAC
        elif self.algorithm == "A2C":
            return A2C
        elif self.algorithm == "TD3":
            return TD3
        elif self.algorithm == "DDPG":
            return DDPG
        return PPO

    def make_model(self, policy_kwargs=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = dict()
        if self.obs_mode == ObservationMode.COMPOUND_1:
            add_custom_feature_extractor_glo_rep_only(policy_kwargs)
            return self.get_alg()("MultiInputPolicy", get_eval_env(self.obs_mode), policy_kwargs=policy_kwargs, **model_kwargs)
        elif self.obs_mode in [ObservationMode.SIMPLE_IMG, ObservationMode.FULL_IMG]:
            return self.get_alg()("CnnPolicy", get_eval_env(self.obs_mode), policy_kwargs=policy_kwargs, **model_kwargs)
        else:
            return self.get_alg()("MlpPolicy", get_eval_env(self.obs_mode), policy_kwargs=policy_kwargs, **model_kwargs)

    def clear_model(self):
        model = self.get_model()
        self.model = None
        return model

    def get_model_path(self, ver=None):
        if ver is None:
            ver = self.version
        return self.model_file_base_path + f"-{self.algorithm}-{self.agent_id}-V{ver}"

    def get_model(self, env=None, ver=None):
        # Note: if the model is already loaded, specifying "ver" won't have an effect
        if self.model is None:
            self.model = self.get_alg().load(self.get_model_path(ver), env)
        if env is not None:
            self.model.set_env(env)
        return self.model

    def save_model(self, model: SUPPORTED_MODELS):
        model.save(self.get_model_path())

    def get_num_steps_trained(self):
        if not hasattr(self, "num_steps_trained"):
            self.num_steps_trained = 0
        return self.num_steps_trained

    def get_version(self):
        if not hasattr(self, "version"):
            self.version = None
        return self.version

    def new_version(self):
        # Increases the version of the model and starts saving with a new number
        model = self.get_model()
        if self.get_version() is None:
            self.version = 0
        else:
            assert self.version is not None
            self.version += 1
        self.save_model(model)

    def get_win_rate_history(self):
        if not hasattr(self, "win_rate_history"):
            self.win_rate_history = []
        return self.win_rate_history

    def get_obs_mode(self):
        if not hasattr(self, "obs_mode"):
            self.obs_mode = ObservationMode.SIMPLE_IMG
        return self.obs_mode

    def update_training_steps(self, num_steps_to_add):
        self.num_steps_trained = self.get_num_steps_trained() + num_steps_to_add

    def update_win_rate(self, win_rate):
        self.win_rate = win_rate
        history = self.get_win_rate_history()
        history.append((self.get_num_steps_trained(), win_rate))
    
    def show_win_rate_graph(self):
        cumulative_wr = dict()
        count = dict()
        points = []
        for steps, wr in self.win_rate_history:
            cumulative_wr[steps] = wr if steps not in cumulative_wr else cumulative_wr[steps] + wr
            count[steps] = 1 if steps not in count else count[steps] + 1
        
        for steps, _ in self.win_rate_history:
            points.append((steps, cumulative_wr[steps] / count[steps]))

        plt.plot(*zip(*points))
        plt.xlabel("Steps")
        plt.ylabel("Win Rate")
        plt.show()

        

class DummyAgent(Agent):
    def __init__(self) -> None:
        self.agent_id = "X"
        self.algorithm = "Random"
        self.win_rate = 0.5
        self.version = "X"

        self.obs_mode = ObservationMode.NONE

    def update_training_steps(self, num_steps_to_add):
        pass

    def update_win_rate(self, win_rate):
        pass

    def get_model(self, env=None, *args, **kwargs):
        assert env is not None, "Must pass env to dummy agent"
        return DummyModel(env)

class AgentCollection: # A collection of agents that will be played against each other
    @staticmethod
    def load(data_path):
        with open(data_path, "rb") as f:
            coll = pickle.load(f)
        coll.data_path = data_path
        # for agent in coll.agents:
        #     agent: Agent
        #     agent.model_file_path = "./saved_models" + agent.model_file_path[1:]
        return coll

    def __init__(self, data_path, saved_models_location):
        self.agents: list[Agent] = []
        self.next_id = 0
        self.model_train_verbosity: Optional[int] = None
        self.saved_models_location = saved_models_location
        self.data_path = data_path
        self.use_dummy_model = False # Whether to use a dummy model for the opponent

    def should_use_dummy_model(self):
        if not hasattr(self, "use_dummy_model"):
            return False
        return self.use_dummy_model

    def get_agent_by_id(self, agent_id):
        return next(agent for agent in self.agents if agent.agent_id == agent_id)

    def get_saved_models_location(self):
        if not hasattr(self, "saved_models_location"):
            self.saved_models_location = "./saved_models/"
        return self.saved_models_location

    def copy_agent(self, agent: Agent):
        self.create_agent(agent.algorithm, agent.obs_mode, agent.get_model())

    def create_agent(self, algorithm = "PPO", obs_mode = ObservationMode.SIMPLE_IMG, model = None, policy_kwargs = None, model_kwargs = None):
        base_file_path = f"{self.get_saved_models_location()}/pvp_agent_collection"
        self.agents.append(Agent(self.next_id, base_file_path, algorithm, obs_mode=obs_mode, model=model, policy_kwargs=policy_kwargs, model_kwargs=model_kwargs))
        self.next_id += 1
        return self.agents[-1]

    def new_version_all_agents(self):
        for agent in self.agents:
            agent.new_version()
        self.save()

    @staticmethod
    def get_agent_ranking(agent: Agent):
        if len(agent.win_rate_history) == 0:
            return agent.win_rate
        return agent.win_rate + max([x[1] for x in agent.win_rate_history])
    
    def subset_top_k(self, k):
        # discards all but the top k agents (doesn't delete their models)
        # An agent's ranking is determined by its current win rate plus its maximum win rate
        self.agent = self.get_top_k(k)
        self.save()

    def get_top_k(self, k):
        if k >= len(self.agents):
            return self.agents # We already have k or less
        rankings = [(agent, self.get_agent_ranking(agent)) for agent in self.agents]
        rankings = sorted(rankings, key=lambda x: x[1], reverse=True)

        return [ranked_agent[0] for ranked_agent in rankings][:k]

    def train_agents(self, num_opponents=1, steps_per_opponent=10_000, include_self=False, selection_method="closest", verbosity_level=2, top_k=None):
        if verbosity_level > 0: print("######### Training All Agents")
        if top_k is not None:
            agents = self.get_top_k(top_k)
        else:
            agents = self.agents
        for i, agent in enumerate(agents):
            if verbosity_level > 1: print(f"############# Training Agent {agent} ({i+1} / {len(self.agents)})")
            self.train_agent(agent, num_opponents, steps_per_opponent, include_self, selection_method, verbosity_level)

    def train_agent(self, agent: Agent, num_opponents=1, steps_per_opponent=10_000, include_self=False, selection_method="closest", verbosity_level=3):
        model: Optional[SUPPORTED_MODELS] = agent.get_model()
        opponents = self.get_opponents(agent, num_opponents, include_self, selection_method)
        for i, opponent in enumerate(opponents):
            if verbosity_level > 2: print(f"################# Training Agent {agent} on opponent {opponent} (Opponent {i+1}/{len(opponents)})")
            self.run_train_step(agent, model, opponent, steps=steps_per_opponent)
            if verbosity_level > 3: print(f"################# Saved model for Agent {agent}")

        agent.save_model(model)
        agent.update_training_steps(steps_per_opponent * num_opponents)

    def run_train_step(self, agent: Agent, model: SUPPORTED_MODELS, enemy: Agent, steps=10_000):
        # Runs matches for agent1 against agent2 for the specified number of steps and saves the model at the end of it
        env = get_train_env(obs_mode=agent.obs_mode, display_elem_tracking_only=True)
        env.set_enemy_info(enemy.get_model(env=env), enemy.obs_mode)
        assert model is not None, "probably set unsupported algorithm types"
        model.set_env(env)
        if self.model_train_verbosity is not None: model.verbose = self.model_train_verbosity
        model.learn(steps)

    def get_opponents(self, agent: Agent, num_opponents=1, include_self=False, selection_method="closest"):
        if self.should_use_dummy_model():
            return [DummyAgent()] * num_opponents
        candidates = self.agents if include_self else [other_agent for other_agent in self.agents if other_agent.agent_id != agent.agent_id]
        opponents = self.get_opponents_helper(agent, num_opponents % len(candidates), selection_method, candidates)
        if num_opponents >= len(candidates):
            opponents = opponents + candidates * (num_opponents // len(candidates))
        return opponents

    def get_opponents_helper(self, agent, num_opponents, selection_method, candidates):
        if selection_method == "closest":
            return self.get_opponents_closest(agent, num_opponents, candidates)
        elif selection_method == "random":
            return self.get_opponents_random(num_opponents, candidates)
        elif selection_method == "random weighted" or True:
            return self.get_opponents_random_weighted(agent, num_opponents, candidates)

    def get_opponents_random(self, num_opponents, candidate_agents):
        return random.sample(candidate_agents, k=num_opponents)

    def get_opponents_closest(self, agent, num_opponents, candidate_agents):
        win_rate_diffs = [
            (other_agent, abs(other_agent.win_rate - agent.win_rate))
            for other_agent in candidate_agents
        ]

        assert len(win_rate_diffs) > num_opponents, "Not enough agents in pool to choose specified number of opponents"

        opponents = [pair[0] for pair in sorted(win_rate_diffs, key=lambda pair: pair[1])[:num_opponents]]
        random.shuffle(opponents)
        return opponents

    def get_opponents_random_weighted(self, agent, num_opponents, candidate_agents):
        weights = [
            1 / (abs(other_agent.win_rate - agent.win_rate) + 0.3)
            for other_agent in candidate_agents
        ]

        return random.choices(candidate_agents, weights, k=num_opponents)

    def run_eval_matches(self, eval_matches_per_agent=1, selection_method="random", include_self=False, render=False, verbosity_level=2):
        if verbosity_level > 0: print(f"##### Running evaluation matches for all agents")
        for i, agent in enumerate(self.agents):
            sum_ratios = 0 # Get the cumulative sum of WR ratios and apply at the end to reduce noise
            num_wins = 0
            if verbosity_level > 1: print(f"######### Running matches for Agent {agent} (Agent {i+1}/{len(self.agents)})")
            opponents = self.get_opponents(agent, num_opponents=eval_matches_per_agent, include_self=include_self, selection_method=selection_method)
            for j, opponent in enumerate(opponents):
                if verbosity_level > 2: print(f"############# Running match for Agent {agent} against Agent {opponent} (match {j+1}/{eval_matches_per_agent})")
                is_win, ratio = self.run_eval_match(agent, opponent, render, render, verbosity_level=verbosity_level, update_win_rates=False)
                sum_ratios += ratio
                if is_win:
                    num_wins += 1
            scaled_rate = np.clip(WINRATE_UPDATE_RATE * len(opponents), 0, 1)
            new_wr = num_wins / len(opponents)
            agent.update_win_rate(agent.win_rate * (1 - scaled_rate) + scaled_rate * new_wr)

    @staticmethod
    def run_eval_match(agent1: Agent, agent2: Agent, render=False, render_all_steps=False, verbosity_level=4, update_win_rates=True, agent1_ver=None, agent2_ver=None):
        env = get_eval_env(render=render)
        model1, model2 = agent1.get_model(ver=agent1_ver), agent2.get_model(env=env, ver=agent2_ver)
        if agent1_ver is None: agent1_ver = agent1.get_version()
        if agent2_ver is None: agent2_ver = agent2.get_version()
        env.set_models({AgentRole.MAIN: model1, AgentRole.ALT: model2})
        env.set_obs_modes({AgentRole.MAIN: agent1.obs_mode, AgentRole.ALT: agent2.obs_mode})
        env.set_display_infos({AgentRole.MAIN: f"Agent {agent1.agent_id} ver={agent1_ver}", AgentRole.ALT: f"Agent {agent2.agent_id} ver={agent2_ver}"})
        #if render: env.show_obs = True
        terminated, truncated, player_dead, base_dead = env.run_match(render, render_all_steps)
        point_difference = env.cumulative_rewards[AgentRole.MAIN] - env.cumulative_rewards[AgentRole.ALT]
        if point_difference > 0:
            winner = agent1
            loser = agent2
        else:
            winner = agent2
            loser = agent1

        win_rate_ratio = ((loser.win_rate + 0.1) / (winner.win_rate + 0.1)) ** 2
        factor = WINRATE_UPDATE_RATE * win_rate_ratio
        factor = min(factor, 0.1)
        if update_win_rates:
            winner.update_win_rate(winner.win_rate * (1 - factor) + factor)
            loser.update_win_rate(loser.win_rate * (1 - factor))
        if verbosity_level > 3:
            if terminated:
                termination_type = f"terminated ({'Base dead ' if base_dead else ''} {'Player dead' if player_dead else ''})"
            elif truncated:
                termination_type = "truncated"
            else:
                termination_type = "other"
            print(f"################# Match done. Winner={winner}, Loser={loser}. \n\tPoint diff={abs(point_difference):.2f}, WR ratio={win_rate_ratio:.2f}, WR factor={factor:.3f}, Sim steps={env.game.sim.sim_step}, termination type={termination_type}")
        env.close()

        if point_difference <= 0:
            win_rate_ratio = -win_rate_ratio # adjust for loss

        return point_difference > 0, win_rate_ratio

    def save(self):
        models = {agent.agent_id: agent.clear_model() for agent in self.agents}
        with open(self.data_path, 'wb') as f:
            pickle.dump(self, f)
        for agent in self.agents:
            agent.model = models[agent.agent_id]

    def show_agents(self):
        print("Agents:")
        for agent in self.agents:
            print(agent)

def run_eval_round(path, matches_per_agent=3, render=False):
    coll: AgentCollection = AgentCollection.load(path)
    coll.run_eval_matches(eval_matches_per_agent=matches_per_agent, verbosity_level=4, render=render)
    coll.save()
    coll.show_agents()

def run_eval_match_for_id(path, agent_id, other_id=None):
    coll: AgentCollection= AgentCollection.load(path)
    agent = coll.get_agent_by_id(agent_id)
    if other_id is not None:
        opponent = coll.get_agent_by_id(other_id)
    else:
        opponent = coll.get_opponents(agent)[0]
    print(f"opponent: {opponent}")
    coll.run_eval_match(agent, opponent, render=True, render_all_steps=True)
    coll.show_agents()
    coll.save()

def collection_training_regimen(path, full_iters=15, agents_per_iter=10, steps_per_opponent=10_000, opponents_per_iter=1, multiproc=False, multiproc_batch=4, agent_filter_hook=None, eval_matches_per_agent=3, filter_schedule=None):

    coll: AgentCollection = AgentCollection.load(path)
    coll.model_train_verbosity = 0
    print(coll.data_path)
    for i in range(full_iters):
        if filter_schedule is not None:
            k = filter_schedule[:i+1][-1]
        else:
            k = None
        if not multiproc: training_iter_regular(agents_per_iter, steps_per_opponent, coll, i, agent_filter_hook, opponents_per_iter, top_k=k)
        else: training_iter_multiprocessing(coll, path, agents_per_iter, steps_per_opponent, multiproc_batch)
        coll.run_eval_matches(eval_matches_per_agent=eval_matches_per_agent,verbosity_level=1)
        coll.show_agents()
        coll.save()

def training_iter_regular(agents_per_iter, steps_per_opponent, coll: AgentCollection, i, agent_filter_hook, opponents_per_iter, top_k=None):
    print(f"training round {i}")
    t = time()
    if top_k is not None:
        agents = coll.get_top_k(top_k)
    else:
        agents = []
        for j in range(agents_per_iter):
            if agent_filter_hook is not None:
                pool = [a for a in coll.agents if agent_filter_hook(a)]
            else:
                pool = coll.agents
            if agents_per_iter >= len(coll.agents):
                agent = coll.agents[j % len(coll.agents)]
            else:
                agent = random.choices(pool, [a.win_rate + 0.1 for a in pool])[0]
            agents.append(agent)

    for i, agent in enumerate(agents):
        coll.train_agent(agent, num_opponents=opponents_per_iter, steps_per_opponent=steps_per_opponent, include_self=True, selection_method="random weighted", verbosity_level=3)
        print(f"trained agent ({i+1}/{len(agents)}) {agent}-{agent.algorithm}, total time = {time() - t:.0f}")
        t = time()


def one_agent_iter_multiprocessing(collection_path, agent_id, steps_per_opponent, start_time):
    coll: AgentCollection = AgentCollection.load(collection_path)
    agent = coll.get_agent_by_id(agent_id)
    coll.train_agent(agent, steps_per_opponent=steps_per_opponent, selection_method="random weighted")
    print(f"trained agent {agent}, total time = {time() - start_time}")

def training_iter_multiprocessing(coll: AgentCollection, coll_data_path, agents_per_iter, steps_per_opponent, batch):
    start_time = time()

    #agents = random.choices(coll.agents, [a.win_rate for a in coll.agents], k=agents_per_iter)
    if agents_per_iter > len(coll.agents):
        agents = coll.agents * (agents_per_iter // len(coll.agents)) + random.sample(coll.agents, k=agents_per_iter % len(coll.agents))
    else:
        agents = random.sample(coll.agents, k=agents_per_iter)
    num_batch = (agents_per_iter - 1) // batch + 1
    for i in range(num_batch):
        agent_batch = agents[i * batch: (i + 1) * batch]
        print(f"Running multiproc batch {i + 1} / {num_batch} for agents {[(agent.agent_id, agent.algorithm, agent.obs_mode) for agent in agent_batch]}")
        processes = [multiprocessing.Process(target=one_agent_iter_multiprocessing, args=(coll_data_path, agent.agent_id, steps_per_opponent, start_time)) for agent in agent_batch]

        for p in processes:
            p.start()
        for p in processes:
            p.join()


def try_vectorized_environment(path):
    # Haven't tried yet
    def set_up_environment():
        env = get_train_env(display_elem_tracking_only=True)
        coll = AgentCollection.load(path)
        agent1 = coll.agents[2]
        agent2 = coll.agents[5]
        env.set_enemy_info(agent2.get_model())
        return env
    t0 = time()
    print("running")
    env = SubprocVecEnv([lambda: set_up_environment() for _ in range(1)])
    coll: AgentCollection = AgentCollection.load(path)
    agent1 = coll.agents[2]
    model = agent1.get_model(env)
    model.learn(10_000)
    agent1.save_model(model)
    print(f"total time={time()-t0}")

def create_env(algorithm, obs_mode, num_agents, extra_descriptor=None, policy_kwargs=None, model_kwargs=None, use_dummy=False):
    collection_path, model_path = path_generator(algorithm, obs_mode, extra_descriptor=extra_descriptor)
    coll = AgentCollection(collection_path, model_path)
    for i in range(num_agents):
        coll.create_agent(algorithm, obs_mode=obs_mode, policy_kwargs=policy_kwargs, model_kwargs=model_kwargs)
    coll.use_dummy_model = use_dummy
    coll.save()
    return coll

def compare_versions(path1, path2, v1, v2, n_matches=50, render=False):
    coll1 = AgentCollection.load(path1)
    coll2 = AgentCollection.load(path2)
    v1_wins, v2_wins = 0, 0
    print(f"Comparing version {v1}-{path1} to version {v2}-{path2}")
    for _ in range(n_matches):
        agent1 = random.choice(coll1.agents)
        agent2 = random.choice(coll2.agents)
        agent1_won, _ = AgentCollection.run_eval_match(agent1, agent2, update_win_rates=False, agent1_ver=v1, agent2_ver=v2, render=render, render_all_steps=render)
        if agent1_won:
            v1_wins += 1
            print(f"{agent1} beats {agent2} (v1 wins)")
        else:
            v2_wins +=1
            print(f"{agent2} beats {agent1} (v2 wins)")
    print("Testing complete")
    print(f"Version {v1}-{path1} has {v1_wins}, version {v2}-{path2} has {v2_wins}")

def train_and_version(path, version_iters=3):
    print(f"Running training and versioning for {path}")
    for i in range(version_iters):
        coll: AgentCollection = AgentCollection.load(path)
        collection_training_regimen(path, 3, 50, steps_per_opponent=10_000, opponents_per_iter=1, eval_matches_per_agent=20)
        coll.new_version_all_agents()
        print(f"New version created for {path}")
    #collection_training_regimen(path, 3, 2, steps_per_opponent=10_000, opponents_per_iter=1, eval_matches_per_agent=20)

def see_architecture(path, agent_id):
    coll: AgentCollection= AgentCollection.load(path)
    agent = coll.get_agent_by_id(agent_id)
    model: PPO = agent.get_model() # type: ignore
    print(model.policy)

def dummy_training_phase_specs(subset_indices=None):
    specs = [
        ("PPO", ObservationMode.VEC_10),
        ("SAC", ObservationMode.VEC_10),
        ("TD3", ObservationMode.VEC_10),
        ("SAC", ObservationMode.SIMPLE_IMG),
    ]
    if subset_indices is not None:
        return [specs[i] for i in subset_indices]
    else:
        return specs

def dummy_training_phase_create_envs(subset_indices=None, policy_kwargs=None):
    for alg, obs_mode in dummy_training_phase_specs(subset_indices):
        coll = create_env(alg, obs_mode, 50, extra_descriptor="dummy", policy_kwargs=policy_kwargs)
        coll.use_dummy_model = True
        coll.save()

def dummy_training_phase_train_envs(subset_indices=None):
    colls = []
    for alg, obs_mode in dummy_training_phase_specs(subset_indices):
        path, _ = path_generator(alg, obs_mode, extra_descriptor="dummy")
        train_and_version(path, 1)

def show_wr_graph_for_agent(path, agent_id):
    coll: AgentCollection= AgentCollection.load(path)
    agent = coll.get_agent_by_id(agent_id)
    print(f"current WR: {agent.win_rate}")
    agent.show_win_rate_graph()

ARCH_128x3 = [128] * 3 + [64]
ARCH_256x4 = [256] * 4 + [64]
ARCH_512x5 = [512] * 5 + [64]

def arch_dict(arch, activation=None):
    return dict(net_arch=dict(pi=arch, vf=arch, qf=arch, activation_fn=activation))

if __name__ == "__main__":
    path = path_generator("PPO", ObservationMode.COMPOUND_1, extra_descriptor="128x3")[0]
    
    #other_path = path_generator("SAC", ObservationMode.VEC_10)[0]
    #create_env("PPO", ObservationMode.COMPOUND_1, 1, extra_descriptor="128x3-glo-only", policy_kwargs=arch_dict(ARCH_128x3, th.nn.Sigmoid), model_kwargs=dict(), use_dummy=True)
    #expand_env()
    #coll: AgentCollection = AgentCollection.load(path)
    #coll.new_version_all_agents()

    #dummy_training_phase_create_envs([0], policy_kwargs=dict(net_arch=dict(pi=[128, 128, 128, 64], vf=[128, 128, 128, 64], qf=[128, 128, 128, 64], activation_fn=th.nn.ReLU)))
    #dummy_training_phase_train_envs([0])
    #show_wr_graph_for_agent(path, 0)
    #collection_training_regimen(path, 10, 50, 10_000, 1, eval_matches_per_agent=50, filter_schedule=[25, 17, 12, 10, 5, 4, 3])
    #collection_training_regimen(path, 25, 1, 10_000, 1, eval_matches_per_agent=10)

    #path, _ = path_generator("SAC", ObservationMode.VEC_10, extra_descriptor="dummy")
    #collection_training_regimen(path, 5, 3, steps_per_opponent=2_000, opponents_per_iter=3, eval_matches_per_agent=2)
    run_eval_round(path, matches_per_agent=3, render=True)
    #compare_versions(path, other_path, 1, 0, 100)
    # run_eval_match_for_id(path, 0, None)
    #see_architecture(path, 0)
