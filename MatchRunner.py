import random
from time import time
from typing import Optional, Union
from numpy import searchsorted
from stable_baselines3 import PPO, A2C, TD3, SAC, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
import pickle
import multiprocessing

from MLWrapper import AgentRole, MLWrapper, MatchWrapper, ObservationMode, PVPWrapper

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



SUPPORTED_MODELS = Union[PPO, SAC, A2C, TD3, DDPG]

def get_train_env(obs_mode=ObservationMode.SIMPLE_IMAGE, display_elem_tracking_only=False):
    return PVPWrapper(X_DIM, Y_DIM, render_mode="human", display_elem_tracking_only=display_elem_tracking_only, obs_mode=obs_mode)

def get_eval_env(obs_mode=ObservationMode.SIMPLE_IMAGE, render=False):
    return MatchWrapper(X_DIM, Y_DIM, render_mode="human", base_obs_mode=obs_mode, display_elem_tracking_only=not render)

class Agent:
    def __init__(self, agent_id, model_file_path, algorithm = "PPO", obs_mode = ObservationMode.SIMPLE_IMAGE):
        self.agent_id = agent_id
        self.model_file_path = model_file_path
        self.algorithm = algorithm
        self.win_rate = 0.5 # Start at a lower rate to be able to compete with worse models
        self.num_steps_trained = 0
        self.win_rate_history = [] # List of (<training steps>, <winrate at training steps>) tuples

        self.obs_mode = obs_mode
        #Ensure that there's always a model stored
        model = self.make_model() # TODO: might want to add support for initialization off of an existing saved (or currently loaded) model that was trained elsewhere.
        
        model.save(self.model_file_path)
        self.model = model
    
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

    def get_model(self, env = None): # TODO: figure out if it's okay to store the loaded model and therefore pickle it later
        model = None
        if hasattr(self, "model"):
            model = self.model
        if model is None:
            model = self.get_alg().load(self.model_file_path, env)
        elif env is not None:
            model.set_env(env)
        self.model = model
        return model
    
    def save_model(self, model: SUPPORTED_MODELS):
        model.save(self.model_file_path)
    
    def get_num_steps_trained(self):
        if not hasattr(self, "num_steps_trained"):
            self.num_steps_trained = 0
        return self.num_steps_trained
    
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
        file_path = f"./{self.get_saved_models_location()}/pvp-agent-collection-{algorithm}-{self.next_id}"
        self.agents.append(Agent(self.next_id, file_path, algorithm, obs_mode=obs_mode))
        self.next_id += 1
        return self.agents[-1]

    def train_all_agents(self, num_opponents=1, steps_per_opponent=10_000, include_self=False, selection_method="closest", verbosity_level=2):
        if verbosity_level > 0: print("######### Training All Agents")
        for i, agent in enumerate(self.agents):
            if verbosity_level > 1: print(f"############# Training Agent {agent.agent_id} (agent {i+1} / {len(self.agents)})")
            self.train_agent(agent, num_opponents, steps_per_opponent, include_self, selection_method, verbosity_level)

    def train_agent(self, agent: Agent, num_opponents=1, steps_per_opponent=10_000, include_self=False, selection_method="closest", verbosity_level=3):
        opponents = self.get_opponents(agent, num_opponents, include_self, selection_method)
        for i, opponent in enumerate(opponents):
            if verbosity_level > 2: print(f"################# Training Agent {agent.agent_id} on opponent {opponent.agent_id} (Opponent {i+1}/{len(opponents)})")
            self.run_train_step(agent, opponent, steps=steps_per_opponent)
            if verbosity_level > 3: print(f"################# Saved model for Agent {agent.agent_id}")

    def run_train_step(self, agent1: Agent, agent2: Agent, steps=10_000):
        # Runs matches for agent1 against agent2 for the specified number of steps and saves the model at the end of it
        env = get_train_env(obs_mode=agent1.obs_mode, display_elem_tracking_only=True)
        env.set_enemy_model(agent2.get_model(), agent2.obs_mode)
        model: Optional[SUPPORTED_MODELS] = agent1.get_model()
        assert model is not None, "probably set unsupported algorithm types"
        model.set_env(env)
        if self.model_train_verbosity is not None: model.verbose = self.model_train_verbosity
        model.learn(steps)
        agent1.save_model(model)
        agent1.update_training_steps(steps)
    
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
            if verbosity_level > 1: print(f"######### Running matches for Agent {agent.agent_id} (Agent {i+1}/{len(self.agents)})")
            opponents = self.get_opponents(agent, num_opponents=matches_per_agent, selection_method=selection_method)
            for j, opponent in enumerate(opponents):
                if verbosity_level > 2: print(f"############# Running match for Agent {agent.agent_id} against Agent {opponent.agent_id} (match {j+1}/{matches_per_agent})")
                self.run_eval_match(agent, opponent, render, verbosity_level=verbosity_level)

    def run_eval_match(self, agent1: Agent, agent2: Agent, render=False, render_all_steps=False, verbosity_level=4):
        env = get_eval_env(render=render)
        model1, model2 = agent1.get_model(), agent2.get_model()
        env.set_models({AgentRole.MAIN: model1, AgentRole.ALT: model2})
        env.set_obs_modes({AgentRole.MAIN: agent1.obs_mode, AgentRole.ALT: agent2.obs_mode})
        if render: env.show_obs = True
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
        winner.update_win_rate(winner.win_rate * (1 - factor) + factor)
        loser.update_win_rate(loser.win_rate * (1 - factor))
        env.close()
        if verbosity_level > 3: print(f"################# Finished Match. Winner={winner.agent_id}, Loser={loser.agent_id}. Point difference={abs(point_difference):.2f}. win_rate_ratio={win_rate_ratio:.2f}. Win rate factor={factor:.3f}")
    
    def save(self):
        models = {agent.agent_id: agent.clear_model() for agent in self.agents}
        with open(self.data_path, 'wb') as f:
            pickle.dump(self, f)
        for agent in self.agents:
            agent.model = models[agent.agent_id]
    
    def show_win_rates(self):
        print("Win rates:")
        for agent in self.agents:
            print(f"Agent {agent.agent_id}: {agent.win_rate}")

def create_new_agent(algorithm):
    coll: AgentCollection= AgentCollection.load()
    coll.create_agent(algorithm)
    coll.save()

def run_training_round(num_opponents=3, steps_per_opponent=10_000):
    coll: AgentCollection= AgentCollection.load()
    #coll.model_train_verbosity = 1
    coll.train_all_agents(num_opponents, steps_per_opponent, True)

def run_eval_round(path, matches_per_agent=3):
    coll: AgentCollection = AgentCollection.load(path)
    coll.run_eval_matches(matches_per_agent=matches_per_agent, verbosity_level=4)
    coll.save()
    coll.show_win_rates()

def run_eval_match_for_id(path, agent_id):
    coll: AgentCollection= AgentCollection.load(path)
    agent = coll.get_agent_by_id(agent_id)
    opponent = coll.get_opponents(agent)[0]
    print(f"opponent: {opponent.agent_id}")
    coll.run_eval_match(agent, opponent, render=True, render_all_steps=True)
    coll.show_win_rates()

def create_small_agent_arena():
    coll: AgentCollection = AgentCollection(saved_models_location=SMALL_AGENT_SAVED_MODELS_PATH)
    for i in range(10):
        coll.create_agent("PPO")
    for i in range(10):
        coll.create_agent("A2C")
    coll.save()

def collection_training_regimen(path, full_iters=15, agents_per_iter=10, steps_per_opponent=10_000, multiproc=False, multiproc_batch=4, agent_filter_hook=None):
    
    coll: AgentCollection = AgentCollection.load(path)
    coll.model_train_verbosity = 0
    for i in range(full_iters):
        if not multiproc: training_iter_regular(agents_per_iter, steps_per_opponent, coll, i, agent_filter_hook)
        else: training_iter_multiprocessing(coll, path, agents_per_iter, steps_per_opponent, multiproc_batch)
        coll.run_eval_matches(matches_per_agent=3,verbosity_level=4)
        coll.save()
        coll.show_win_rates()

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
        coll.train_agent(agent, steps_per_opponent=steps_per_opponent, selection_method="random weighted", verbosity_level=3)
        print(f"trained agent {agent.agent_id}-{agent.algorithm}, total time = {time() - t}")
        t = time()


def one_agent_iter_multiprocessing(collection_path, agent_id, steps_per_opponent, start_time):
    coll: AgentCollection = AgentCollection.load(collection_path)
    agent = coll.get_agent_by_id(agent_id)
    coll.train_agent(agent, steps_per_opponent=steps_per_opponent, selection_method="random weighted")
    print(f"trained agent {agent.agent_id}, total time = {time() - start_time}")

def training_iter_multiprocessing(coll: AgentCollection, coll_data_path, agents_per_iter, steps_per_opponent, batch):
    start_time = time()

    agents = random.choices(coll.agents, [a.win_rate for a in coll.agents], k=agents_per_iter)
    num_batch = (agents_per_iter - 1) // batch + 1
    for i in range(num_batch):
        agent_batch = agents[i * batch: (i + 1) * batch]
        print(f"Running multiproc batch {i + 1} / {num_batch} for agents {[(agent.agent_id, agent.algorithm) for agent in agent_batch]}")
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
        env.set_enemy_model(agent2.get_model())
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
    coll = AgentCollection(VEC_10_SMALL_AGENT_SAVED_MODELS_PATH, VEC_10_SMALL_AGENT_COLLECTION_PATH)
    for i in range(10):
        coll.create_agent("PPO", ObservationMode.VECTOR_10)
    coll.save()

if __name__ == "__main__":
    #create_env()
    collection_training_regimen(VEC_10_SMALL_AGENT_COLLECTION_PATH, 10, 7, steps_per_opponent=20_000, multiproc=True, multiproc_batch=7)
    #run_eval_match_for_id(FULL_IMG_SMALL_AGENT_COLLECTION_PATH, 1)
    #run_eval_round(FULL_IMG_SMALL_AGENT_COLLECTION_PATH, matches_per_agent=3)
    #agent = Agent(101, "./saved_models/ppo-model", algorithm="PPO")
    


    