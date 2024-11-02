from MLWrapper import *
from MatchRunner import get_eval_env
import time
import cProfile
from time import perf_counter

from ObservationSpaces import ObservationMode

# Bring in these constants to try and reduce inadvertent changes to the benchmark definition
STRUCTURE_STATS = lambda: BaseStats(200, 200, 50, 0, TURRET_SIZE, 50, 2.0, TURRET_AGGRO_DISTANCE, 300, 300)
MELEE_MINION_STATS = lambda: BaseStats(100, 100, 0, 8.0, MINION_SIZE, 5, 1.0, MINION_SIZE * 2, 20, 20)
RANGED_MINION_STATS = lambda: BaseStats(70, 70, 0, 8.0, MINION_SIZE * 0.8, 15, 1.0, MINION_AGGRO_DISTANCE, 20, 20)
PLAYER_STATS = lambda: BaseStats(500, 500, 100, 10.0, PLAYER_SIZE, 50, 1.0, MINION_AGGRO_DISTANCE * 1.0, 200, 100)
PLAYER_LEVEL_STATS = lambda: LevelStats(100, 10, 0.1, 10, 0.1)

def run_benchmark(obs_mode, N=10):
    class DummyModel:
        def __init__(self, env):
            self.action_space = env.action_space
        def predict(self, *args, **kwargs):
            return self.action_space.sample(), None

    # Run a non-timed match before timing
    env = get_eval_env(max_sim_steps=1000)
    model = DummyModel(env)
    env.set_obs_modes({AgentRole.MAIN: obs_mode, AgentRole.ALT: obs_mode})
    env.set_models({AgentRole.MAIN: model, AgentRole.ALT: model})
    env.run_match()

    accum = 0
    for i in range(N):
        env = get_eval_env(max_sim_steps=1000)
        env.set_obs_modes({AgentRole.MAIN: obs_mode, AgentRole.ALT: obs_mode})
        env.set_models({AgentRole.MAIN: model, AgentRole.ALT: model})
        t0 = perf_counter()
        env.run_match()
        accum += perf_counter() - t0
    t_1k = accum / N
    
    accum = 0
    for i in range(N):
        env = get_eval_env(max_sim_steps=2000)
        env.set_obs_modes({AgentRole.MAIN: obs_mode, AgentRole.ALT: obs_mode})
        env.set_models({AgentRole.MAIN: model, AgentRole.ALT: model})
        t0 = perf_counter()
        env.run_match()
        accum += perf_counter() - t0
    t_2k = accum / N

    accum = 0
    for i in range(N):
        env = get_eval_env(max_sim_steps=4000)
        env.set_obs_modes({AgentRole.MAIN: obs_mode, AgentRole.ALT: obs_mode})
        env.set_models({AgentRole.MAIN: model, AgentRole.ALT: model})
        t0 = perf_counter()
        env.run_match()
        accum += perf_counter() - t0
    t_4k = accum / N

    accum = 0
    for i in range(N):
        env = get_eval_env(max_sim_steps=8000)
        env.set_obs_modes({AgentRole.MAIN: obs_mode, AgentRole.ALT: obs_mode})
        env.set_models({AgentRole.MAIN: model, AgentRole.ALT: model})
        t0 = perf_counter()
        env.run_match()
        accum += perf_counter() - t0
    t_8k = accum / N

    print(f"Results for match with obs mode = {obs_mode}: 1k sim steps took {t_1k:.2f}s. 2k sim steps took {t_2k:.2f}s. 4k sim steps took {t_4k:.2f}s. 8k sim steps took {t_8k:.2f}s.")

def run_all_benchmarks():
    for obs_mode in [ObservationMode.SIMPLE_IMG, ObservationMode.FULL_IMG, ObservationMode.VEC_10]:
        run_benchmark(obs_mode)

def get_digits_divide_modulo(N):
    t0 = perf_counter()
    for i in range(N):
        x = int(random.random() * 200)
        y = x % 2
        y = (x // 2) % 2
        y = (x // 4) % 2
        y = (x // 8) % 2
        y = (x // 16) % 2
        y = (x // 32) % 2

    total_time = perf_counter() - t0
    print(f"Testing divide modulo. Average time = {total_time / N:.2e}")


def get_digits_bitwise(N):
    t0 = perf_counter()
    for i in range(N):
        x = int(random.random() * 200)
        y = x & 2 == 0
        y = x & 4 == 0
        y = x & 8 == 0
        y = x & 16 == 0
        y = x & 32 == 0
        y = x & 64 == 0

    total_time = perf_counter() - t0
    print(f"Testing bitwise. Average time = {total_time / N:.2e}")

"""
Initial benchmark:
Results for match with obs mode = ObservationMode.SIMPLE_IMAGE: 1k sim steps took 0.31s. 2k sim steps took 0.81s. 4k sim steps took 2.21s. 8k sim steps took 8.04s.
Results for match with obs mode = ObservationMode.FULL_IMAGE: 1k sim steps took 0.33s. 2k sim steps took 0.79s. 4k sim steps took 2.14s. 8k sim steps took 5.73s.
Results for match with obs mode = ObservationMode.VECTOR_10: 1k sim steps took 0.31s. 2k sim steps took 0.77s. 4k sim steps took 2.08s. 8k sim steps took 5.46s.
"""

if __name__ == "__main__":
    #cProfile.run('run_benchmark(ObservationMode.SIMPLE_IMAGE, N=5)')
    get_digits_divide_modulo(10000)
    get_digits_bitwise(10000)
