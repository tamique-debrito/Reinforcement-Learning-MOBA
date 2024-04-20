import gym
from gym import spaces
import numpy as np

class Simple2DNavigationEnv(gym.Env):
    def __init__(self):
        self.width = 10  # Width of the 2D space
        self.height = 10  # Height of the 2D space
        self.goal_position = np.array([self.width - 1, self.height - 1])  # Goal position

        # Define action space: Box for displacement (continuous) and Discrete for reset action
        self.action_space = spaces.Tuple((
            spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),  # Displacement action space
            spaces.Discrete(2)  # Reset action space
        ))

        # Define observation space: agent's position (2D)
        self.observation_space = spaces.Box(low=0, high=self.width, shape=(2,), dtype=np.float32)

        # Initialize the agent's position
        self.agent_position = np.array([0, 0])
        
        # Initialize episode step count
        self.steps = 0

    def reset(self):
        # Reset agent's position to the starting location
        self.agent_position = np.array([0, 0])
        self.steps = 0
        return self.agent_position

    def step(self, action):
        # Extract displacement action and reset action from the action tuple
        displacement_action, reset_action = action

        # Apply the displacement action to the agent's position
        self.agent_position += displacement_action

        # Ensure the agent's position stays within the bounds of the environment
        self.agent_position = np.clip(self.agent_position, [0, 0], [self.width - 1, self.height - 1])

        # If the reset action is triggered (action index 1), reset the agent's position to the starting location
        if reset_action == 1:
            self.agent_position = np.array([0, 0])

        # Calculate the reward based on the distance to the goal
        distance_to_goal = np.linalg.norm(self.agent_position - self.goal_position)
        reward = -distance_to_goal

        # Increment the step count
        self.steps += 1

        # Define termination conditions: reaching the goal or exceeding a maximum number of steps
        done = np.linalg.norm(self.agent_position - self.goal_position) < 1.0 or self.steps >= 100

        # Return observation, reward, done, info
        return self.agent_position, reward, done, {}

    def render(self, mode='human'):
        # Print the agent's position
        print(f"Agent's position: {self.agent_position}")