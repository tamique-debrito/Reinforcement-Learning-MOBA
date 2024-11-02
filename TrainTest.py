from torch import device
from MLWrapper import MLWrapper, PVPWrapper
from stable_baselines3 import PPO, A2C, TD3, SAC

from ObservationSpaces import ObservationMode

def test_model(model, steps=100):
    print("Done learning")
    vec_env = model.get_env()
    assert vec_env is not None
    obs = vec_env.reset()
    assert obs is not None
    for i in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

def run_test(obs_move=ObservationMode.FULL_IMG):
    env = MLWrapper(50, 50, render_mode="human", obs_mode=obs_move)

    model = SAC("CnnPolicy", env, verbose=1, device="cuda:0")
    model.learn(total_timesteps=100_000)

    print("Done learning")
    test_model(model, 200)
    model.save("./saved_models/sac-model")
    
    return



def run_test_pvp():
    env1 = PVPWrapper(50, 50, render_mode="human",)
    env2 = PVPWrapper(50, 50, render_mode="human")

    model1 = A2C("CnnPolicy", env1, verbose=1)
    model2 = A2C("CnnPolicy", env2, verbose=1)
    
    env1.set_enemy_info(model2)
    env2.set_enemy_info(model1)
    
    for i in range(5):
        model1.learn(total_timesteps=10_000)
        model2.learn(total_timesteps=10_000)

    test_model(model1)
    
    model1.save("./saved_models/pvp-a2c-model1")
    
    model2.save("./saved_models/pvp-a2c-model2")
    
    return model1, model2



if __name__ == "__main__":
    #run_test("full image")
    model = PPO.load("./saved_models/ppo-model")
    model.set_env(MLWrapper(50, 50, render_mode="human", obs_mode=ObservationMode.FULL_IMG))

    test_model(model, 300)