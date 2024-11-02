from enum import Enum
import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces


from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ObservationSpaces import COMPOUND_1_NUM_UNITS, COMPOUND_1_STATE_ATTRIBUTES, COMPOUND_1_UNIT_ATTRIBUTES


class CustomFeatureExtractorType(Enum):
    TRANSFORMER_1 = 1
    EMBED_AND_CNN = 2

class FeatureExtractorWithTransformer(BaseFeaturesExtractor):
    """
    A custom feature extractor that applies attention layers to the unit list
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        state_shape = observation_space.spaces["state"].shape
        assert state_shape is not None
        state_chann = state_shape[0]
        units_shape = observation_space.spaces["units"].shape
        assert units_shape is not None
        units_chann = units_shape[1]

        # self.state_nn = nn.Sequential(
        #     nn.Linear(state_chann, 16),
        #     nn.ReLU(),
        # )
         
        self.attn = nn.Sequential(
            nn.Linear(units_chann, 64),
            nn.TransformerEncoderLayer(64, 2),
            nn.TransformerEncoderLayer(64, 2),
        )
        self.proj = nn.Linear(state_chann + 64, features_dim)

    def forward(self, observations) -> th.Tensor:
        state = observations["state"]
        units = observations["units"]
        #state_ft = self.state_nn(state)
        units_ft = self.attn(units).mean(axis=1)
        concat = th.concat([state, units_ft], dim=1)
        return self.proj(concat)

class FeatureExtractorGlobalRepOnly(BaseFeaturesExtractor):
    """
    A custom feature extractor that applies attention layers to the unit list
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        state_shape = observation_space.spaces["state"].shape
        assert state_shape is not None
        state_chann = state_shape[0]
        units_shape = observation_space.spaces["units"].shape
        assert units_shape is not None
        units_chann = units_shape[1]

        self.units_pre_embed = nn.Sequential(
            nn.Linear(units_chann, 128),
            nn.ReLU(),
        )

        self.global_pre_embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.global_post_embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.units_post_embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.final_proj = nn.Linear(state_chann + 128 + 128, features_dim) # Concatenate state vector, image representation, and global unit representation

    def get_units_post_embeds(self, units):
        units_pre_embed = self.units_pre_embed(units)
        global_pre_embed = self.global_pre_embed(units_pre_embed)
        global_pre_embed_mean = global_pre_embed.mean(dim=1)
        global_rep_post = self.global_post_embed(global_pre_embed_mean).unsqueeze(1)
        units_pre_embed = units_pre_embed + global_rep_post
        units_post_embed = self.units_post_embed(units_pre_embed)
        return units_post_embed, global_rep_post.squeeze(1)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        state = observations["state"]
        units = observations["units"]
        #state_ft = self.state_nn(state)
        units_post_embed, global_rep = self.get_units_post_embeds(units)

        concat = th.concat([state, units_post_embed.mean(dim=1), global_rep], dim=1)
        return self.final_proj(concat)

class FeatureExtractorGlobalRepAndCNN(BaseFeaturesExtractor):
    """
    A custom feature extractor that applies attention layers to the unit list
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        state_shape = observation_space.spaces["state"].shape
        assert state_shape is not None
        state_chann = state_shape[0]
        units_shape = observation_space.spaces["units"].shape
        assert units_shape is not None
        units_chann = units_shape[1]

        self.units_pre_embed = nn.Sequential(
            nn.Linear(units_chann, 128),
            nn.ReLU(),
        )

        self.global_pre_embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.global_post_embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.units_post_embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Will use 16x16 spatial map
        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2), # should be 7x7 here
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), #should be 4x4 here
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten()
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.zeros((1, 128, 16, 16))
            ).shape[1]
        
        self.final_proj = nn.Linear(state_chann + n_flatten + 128, features_dim) # Concatenate state vector, image representation, and global unit representation

    def get_units_post_embeds(self, units):
        units_pre_embeds = self.units_pre_embed(units)
        global_pre_embed = self.global_pre_embed(units_pre_embeds)
        global_pre_embed_mean = global_pre_embed.mean(dim=1)
        global_rep_post = self.global_post_embed(global_pre_embed_mean).unsqueeze(1)
        units_pre_embeds = units_pre_embeds + global_rep_post
        units_post_embed = self.units_post_embed(units_pre_embeds)
        return units_post_embed, global_rep_post.squeeze(1)
    
    def scatter_embeds(self, units_post_embed: th.Tensor, units: th.Tensor):
        batch_size = units_post_embed.shape[0]
        num_embeds = units_post_embed.shape[1]
        coords = units[:, :, 32:34]

        if True:
            indices = th.clip(coords * 15, 0, 15).round()
            indices = (indices * th.tensor([[[1, 16]]], device="cuda:0")).sum(dim=2).to(th.int64) # Convert to flattened index
            indices = indices.unsqueeze(1).expand((batch_size, 128, num_embeds)).detach() # (batch, embed, image_index)
            units_post_embed = units_post_embed.transpose(1, 2) # Put into (batch, embed, unit_index) order
            base_img = th.zeros((batch_size, 128, 16, 16), device=units_post_embed.device)
            flattened_img = base_img.reshape((batch_size, 128, 16 * 16))
            flattened_img = flattened_img.scatter_add(2, indices, units_post_embed)
            final_img = flattened_img.reshape((batch_size, 128, 16, 16))
        else:
            indices: np.ndarray = np.clip(coords.cpu().numpy() * 15, 0, 15).round()
            indices = indices.astype(np.int64)
            base_img = th.zeros((batch_size, 128, 16, 16), device=units_post_embed.device)
            for batch_idx in range(batch_size):
                for unit_idx, (img_idx1, img_idx2) in enumerate(indices[batch_idx]):
                    base_img[batch_idx, :, img_idx1, img_idx2] += units_post_embed[batch_idx, unit_idx]
            final_img = base_img

        return final_img

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        state = observations["state"]
        units = observations["units"]
        #state_ft = self.state_nn(state)
        units_post_embed, global_rep = self.get_units_post_embeds(units)
        pre_embed_img = self.scatter_embeds(units_post_embed, units)
        post_embed_img = self.cnn(pre_embed_img)

        concat = th.concat([state, post_embed_img, global_rep], dim=1)
        return self.final_proj(concat)

class TestEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.step_num = 0
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(0, 1, shape=(COMPOUND_1_STATE_ATTRIBUTES,), dtype=np.float32),
                "units": spaces.Box(0, 1, shape=(COMPOUND_1_NUM_UNITS, COMPOUND_1_UNIT_ATTRIBUTES), dtype=np.float32),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        return self.observation_space.sample()

    def reset(self, seed=None, options=None):
        self.step_num = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_num += 1
        return self._get_obs(), 0, self.step_num > 100, False, {}
    
    def render(self):
        print("'Render...'")

def add_custom_feature_extractor_glo_rep_only(policy_kwargs):
    policy_kwargs["features_extractor_class"] = FeatureExtractorGlobalRepOnly

policy_kwargs = dict(
    features_extractor_class=FeatureExtractorGlobalRepAndCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

if __name__ == "__main__":
    model = PPO("MultiInputPolicy", TestEnv(), policy_kwargs=policy_kwargs, verbose=1)
    model.learn(1000)