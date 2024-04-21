import random
from time import time
from typing import Optional, Union
from numpy import searchsorted
from stable_baselines3 import PPO, A2C, TD3, SAC, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
import pickle
import multiprocessing

from MLWrapper import MAX_SIM_STEPS, AgentRole, MLWrapper, MatchWrapper, ObservationMode, PVPWrapper

X_DIM = Y_DIM = 50

WINRATE_UPDATE_RATE = 0.03

COLLECTION_PATH = "./agent-collections/agent-collection-data"
SAVED_MODELS_PATH = "./saved_models"

SMALL_AGENT_COLLECTION_PATH = "./agent-collections/small-agent-collection"
SMALL_AGENT_SAVED_MODELS_PATH = "./saved_models/small_model_arena"

FULL_IMG_SMALL_AGENT_COLLECTION_PATH = "./agent-collections/full-img-small-agent-collection"
FULL_IMG_SMALL_AGENT_SAVED_MODELS_PATH = "./saved_models/full_img_small_model_arena"


VEC_10_SMALL_AGENT_COLLECTION_PATH = "./agent-collections/vec-10-small-agent-collection"
VEC_10_SMALL_AGENT_SAVED_MODELS_PATH = "./saved_models/vec_10_small_agent_arena"

VEC_10_SAC_COLLECTION_PATH = "./agent-collections/vec-10-sac-collection"
VEC_10_SAC_SAVED_MODELS_PATH = "./saved_models/vec_10_sac_arena"

VEC_10_A2C_COLLECTION_PATH = "./agent-collections/vec-10-a2c-collection"
VEC_10_A2C_SAVED_MODELS_PATH = "./saved_models/vec_10_a2c_arena"



SUPPORTED_MODELS = Union[PPO, SAC, A2C, TD3, DDPG]

def get_train_env(obs_mode=ObservationMode.SIMPLE_IMAGE, display_elem_tracking_only=False):
    return PVPWrapper(X_DIM, Y_DIM, render_mode="human", display_elem_tracking_only=display_elem_tracking_only, obs_mode=obs_mode)

def get_eval_env(obs_mode=ObservationMode.SIMPLE_IMAGE, render=False, max_sim_steps=MAX_SIM_STEPS):
    return MatchWrapper(X_DIM, Y_DIM, render_mode="human", base_obs_mode=obs_mode, display_elem_tracking_only=not render, max_sim_steps=max_sim_steps)

class Agent:
    def __init__(self, agent_id, model_file_path, algorithm = "PPO", obs_mode = ObservationMode.SIMPLE_IMAGE):
        self.agent_id = agent_id
        self.model_file_base_path = model_file_path
        self.algorithm = algorithm
        self.win_rate = 0.5 # Start at a lower rate to be able to compete with worse models
        self.num_steps_trained = 0
        self.win_rate_history = [] # List of (<training steps>, <winrate at training steps>) tuples
        self.version = 0

        self.obs_mode = obs_mode
        #Ensure that there's always a model stored
        model = self.make_model() # TODO: might want to add support for initialization off of an existing saved (or currently loaded) model that was trained elsewhere.
        
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
    
    def make_model(self):
        if self.get_obs_mode() in [ObservationMode.SIMPLE_IMAGE, ObservationMode.FULL_IMAGE]:
            return self.get_alg()("CnnPolicy", get_eval_env(self.get_obs_mode()))
        else:
            return self.get_alg()("MlpPolicy", get_eval_env(self.get_obs_mode()))
    
    def clear_model(self):
        model = self.get_model()
        self.model = None
        return model

    def get_model_path(self, ver=None):
        if ver is None:
            ver = self.get_version()
        elif ver == -1:
            ver = None # Want to have a way to specify the pre-versioning model, but None already means default to current version
        if hasattr(self, "model_file_path"):
            # convert old path to new format
            self.model_file_base_path = "-".join(self.model_file_path.split("-")[:-2])
        
        if ver is None:
            return self.model_file_base_path + f"-{self.algorithm}-{self.agent_id}"
        else:
            return self.model_file_base_path + f"-{self.algorithm}-{self.agent_id}-V{ver}"

    def get_model(self, env = None, ver=None):
        # Note: if the model is already loaded, specifying "ver" won't have an effect
        model = None
        if hasattr(self, "model"):
            model = self.model
        if model is None:
            model = self.get_alg().load(self.get_model_path(ver), env)
        elif env is not None:
            model.set_env(env)
        self.model = model
        return model
    
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
            self.obs_mode = ObservationMode.SIMPLE_IMAGE
        return self.obs_mode

    def update_training_steps(self, num_steps_to_add):
        self.num_steps_trained = self.get_num_steps_trained() + num_steps_to_add
    
    def update_win_rate(self, win_rate):
        self.win_rate = win_rate
        history = self.get_win_rate_history()
        history.append((self.get_num_steps_trained(), win_rate))
        

class AgentCollection: # A collection of agents that will be played against each other
    @staticmethod
    def load(data_path=COLLECTION_PATH):
        with open(data_path, "rb") as f:
            coll = pickle.load(f)
        coll.data_path = data_path
        # for agent in coll.agents:
        #     agent: Agent
        #     agent.model_file_path = "./saved_models" + agent.model_file_path[1:]
        return coll

    def __init__(self, saved_models_location="./saved_models/", data_path=COLLECTION_PATH):
        self.agents: list[Agent] = []
        self.next_id = 0
        self.model_train_verbosity: Optional[int] = None
        self.saved_models_location = saved_models_location
        self.data_path = data_path

    def get_agent_by_id(self, agent_id):
        return next(agent for agent in self.agents if agent.agent_id == agent_id)

    def get_saved_models_location(self):
        if not hasattr(self, "saved_models_location"):
            self.saved_models_location = "./saved_models/"
        return self.saved_models_location

    def create_agent(self, algorithm = "PPO", obs_mode=ObservationMode.SIMPLE_IMAGE):
        base_file_path = f"{self.get_saved_models_location()}/pvp-agent-collection"
        self.agents.append(Agent(self.next_id, base_file_path, algorithm, obs_mode=obs_mode))
        self.next_id += 1
        return self.agents[-1]

    def new_version_all_agents(self):
        for agent in self.agents:
            agent.new_version()
        self.save()

    def train_all_agents(self, num_opponents=1, steps_per_opponent=10_000, include_self=False, selection_method="closest", verbosity_level=2):
        if verbosity_level > 0: print("######### Training All Agents")
        for i, agent in enumerate(self.agents):
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

    def run_train_step(self, agent: Agent, model, enemy: Agent, steps=10_000):
        # Runs matches for agent1 against agent2 for the specified number of steps and saves the model at the end of it
        env = get_train_env(obs_mode=agent.obs_mode, display_elem_tracking_only=True)
        env.set_enemy_info(enemy.get_model(), enemy.obs_mode)
        assert model is not None, "probably set unsupported algorithm types"
        model.set_env(env)
        if self.model_train_verbosity is not None: model.verbose = self.model_train_verbosity
        model.learn(steps)
    
    def get_opponents(self, agent: Agent, num_opponents=1, include_self=False, selection_method="closest"):
        candidates = self.agents if include_self else [other_agent for other_agent in self.agents if other_agent.agent_id != agent.agent_id]
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
            1 / (2 * abs(other_agent.win_rate - agent.win_rate) + 0.1)
            for other_agent in candidate_agents
        ]

        return random.choices(candidate_agents, weights, k=num_opponents)

    def run_eval_matches(self, matches_per_agent=1, selection_method="random", render=False, verbosity_level=2):
        if verbosity_level > 0: print(f"##### Running evaluation matches for all agents")
        for i, agent in enumerate(self.agents):
            if verbosity_level > 1: print(f"######### Running matches for Agent {agent} (Agent {i+1}/{len(self.agents)})")
            opponents = self.get_opponents(agent, num_opponents=matches_per_agent, selection_method=selection_method)
            for j, opponent in enumerate(opponents):
                if verbosity_level > 2: print(f"############# Running match for Agent {agent} against Agent {opponent} (match {j+1}/{matches_per_agent})")
                self.run_eval_match(agent, opponent, render, verbosity_level=verbosity_level)

    @staticmethod
    def run_eval_match(agent1: Agent, agent2: Agent, render=False, render_all_steps=False, verbosity_level=4, update_win_rates=True, agent1_ver=None, agent2_ver=None):
        env = get_eval_env(render=render)
        model1, model2 = agent1.get_model(ver=agent1_ver), agent2.get_model(ver=agent2_ver)
        env.set_models({AgentRole.MAIN: model1, AgentRole.ALT: model2})
        env.set_obs_modes({AgentRole.MAIN: agent1.obs_mode, AgentRole.ALT: agent2.obs_mode})
        env.set_display_infos({AgentRole.MAIN: f"Agent {agent1.agent_id} ver={agent1_ver}", AgentRole.ALT: f"Agent {agent2.agent_id} ver={agent2_ver}"})
        #if render: env.show_obs = True
        env.run_match(render, render_all_steps)
        point_difference = env.cumulative_rewards[AgentRole.MAIN] - env.cumulative_rewards[AgentRole.ALT]
        if point_difference > 0:
            winner = agent1
            loser = agent2
        else:
            winner = agent2
            loser = agent1
        
        
        # TODO: include winrate difference in the factor here. Something like:
        # if winner.win_rate < loser.win_rate: factor = WINRATE_UPDATE_RATE * (1 + loser.win_rate - winner.win_rate) # And increase WINRATE_UPDATE_RATE
        # else: factor = WINRATE_UPDATE_RATE
        # OR, factor = WINRATE_UPDATE_RATE * ((loser.win_rate + 0.1) / (winner.win_rate + 0.1)) ** 2
        win_rate_ratio = ((loser.win_rate + 0.1) / (winner.win_rate + 0.1)) ** 2
        factor = WINRATE_UPDATE_RATE * win_rate_ratio
        factor = min(factor, 0.1)
        if update_win_rates:
            winner.update_win_rate(winner.win_rate * (1 - factor) + factor)
            loser.update_win_rate(loser.win_rate * (1 - factor))
        if verbosity_level > 3: print(f"################# Match done. Winner={winner}, Loser={loser}. Point diff={abs(point_difference):.2f}. WR ratio={win_rate_ratio:.2f}. WR factor={factor:.3f}. Sim steps={env.game.sim.sim_step}")
        env.close()
        
        return point_difference > 0
    
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

def create_new_agent(algorithm):
    coll: AgentCollection= AgentCollection.load()
    coll.create_agent(algorithm)
    coll.save()

def run_training_round(num_opponents=3, steps_per_opponent=10_000):
    coll: AgentCollection= AgentCollection.load()
    #coll.model_train_verbosity = 1
    coll.train_all_agents(num_opponents, steps_per_opponent, True)

def run_eval_round(path, matches_per_agent=3, render=False):
    coll: AgentCollection = AgentCollection.load(path)
    coll.run_eval_matches(matches_per_agent=matches_per_agent, verbosity_level=4, render=render)
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

def create_small_agent_arena():
    coll: AgentCollection = AgentCollection(saved_models_location=SMALL_AGENT_SAVED_MODELS_PATH)
    for i in range(10):
        coll.create_agent("PPO")
    for i in range(10):
        coll.create_agent("A2C")
    coll.save()

def collection_training_regimen(path, full_iters=15, agents_per_iter=10, steps_per_opponent=10_000, multiproc=False, multiproc_batch=4, agent_filter_hook=None, matches_per_agent=3):
    
    coll: AgentCollection = AgentCollection.load(path)
    coll.model_train_verbosity = 0
    for i in range(full_iters):
        if not multiproc: training_iter_regular(agents_per_iter, steps_per_opponent, coll, i, agent_filter_hook)
        else: training_iter_multiprocessing(coll, path, agents_per_iter, steps_per_opponent, multiproc_batch)
        coll.run_eval_matches(matches_per_agent=matches_per_agent,verbosity_level=4)
        coll.show_agents()
        coll.save()

def training_iter_regular(agents_per_iter, steps_per_opponent, coll: AgentCollection, i, agent_filter_hook):
    print(f"training round {i}")
    t = time()
    for j in range(agents_per_iter):
        if agent_filter_hook is not None:
            pool = [a for a in coll.agents if agent_filter_hook(a)]
        else:
            pool = coll.agents
        if agents_per_iter >= len(coll.agents):
            agent = coll.agents[j % len(coll.agents)]
        else:
            agent = random.choices(pool, [a.win_rate + 0.1 for a in pool])[0]
        coll.train_agent(agent, steps_per_opponent=steps_per_opponent, include_self=True, selection_method="random weighted", verbosity_level=3)
        print(f"trained agent {agent}-{agent.algorithm}, total time = {time() - t}")
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

def switch_out_agents():
    small_agent_path = "small-agent-collection"
    coll: AgentCollection = AgentCollection.load(small_agent_path)
    coll.agents = coll.agents[:10]
    coll.next_id = 10
    for i in range(10):
        coll.create_agent("PPO")
    coll.save()

def try_vectorized_environment(path):
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
    
def create_env():
    coll = AgentCollection(VEC_10_A2C_SAVED_MODELS_PATH, VEC_10_A2C_COLLECTION_PATH)
    for i in range(6):
        coll.create_agent("A2C", ObservationMode.VECTOR_10)
    coll.save()


def expand_env():
    coll = AgentCollection.load(VEC_10_SAC_COLLECTION_PATH)
    for i in range(3):
        coll.create_agent("SAC", ObservationMode.VECTOR_10)
    coll.save()

def compare_versions(path, v1, v2, n_matches=50, render=False):
    coll = AgentCollection.load(path)
    v1_wins, v2_wins = 0, 0
    print(f"Comparing version {v1} to version {v2} for {path}")
    for _ in range(n_matches):
        agent1 = random.choice(coll.agents)
        agent2 = random.choice(coll.agents)
        agent1_won = AgentCollection.run_eval_match(agent1, agent2, update_win_rates=False, agent1_ver=v1, agent2_ver=v2, render=render, render_all_steps=render)
        if agent1_won:
            v1_wins += 1
            print(f"{agent1} beats {agent2} (v1 wins)")
        else:
            v2_wins +=1
            print(f"{agent2} beats {agent1} (v2 wins)")
    print("Testing complete")
    print(f"Version {v1} has {v1_wins}, version {v2} has {v2_wins}")

if __name__ == "__main__":
    #create_env()
    #collection_training_regimen(VEC_10_SMALL_AGENT_COLLECTION_PATH, 10, 6, steps_per_opponent=20_000, multiproc=True, multiproc_batch=6)
    #expand_env()
    #collection_training_regimen(VEC_10_SAC_COLLECTION_PATH, 20, 6, steps_per_opponent=10_000, multiproc=True, multiproc_batch=3, matches_per_agent=2)
    # coll: AgentCollection = AgentCollection.load(VEC_10_A2C_COLLECTION_PATH)
    # coll.new_version_all_agents()
    #collection_training_regimen(VEC_10_A2C_COLLECTION_PATH, 5, 6, steps_per_opponent=50_000, matches_per_agent=3)
    #compare_versions(VEC_10_A2C_COLLECTION_PATH, -1, 0, 200)
    #run_eval_match_for_id(VEC_10_A2C_COLLECTION_PATH, 0, 5)
    run_eval_round(VEC_10_SMALL_AGENT_COLLECTION_PATH, matches_per_agent=3, render=True)
    #agent = Agent(101, "./saved_models/ppo-model", algorithm="PPO")