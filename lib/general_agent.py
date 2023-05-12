# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class GeneralAgent
#   @author: by Kangyao Huang
#   @created date: 24.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    This agent is for all dm_control suite domains and tasks.
"""

import numpy as np
import torch
from lib.agents.agent_ppo2 import AgentPPO2
import gymnasium as gym

class GeneralAgent(AgentPPO2):
    def __init__(self, env_name, cfg, logger, dtype, device, num_threads, training=True, checkpoint=0):
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training
        self.setup_env(env_name)

        super().__init__(self.cfg, self.env, self.logger, self.dtype, self.device, self.num_threads,
                         training=self.training, checkpoint=checkpoint)

    def setup_env(self, env_name):
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.training:
            self.env = gym.make(env_name)
        else:
            self.env = gym.make(env_name, render_mode='human')

    def display(self, num_episode=1, mean_action=True):
        env = self.env
        for _ in range(num_episode):
            state, _ = env.reset()
            if self.running_state is not None:
                state = self.running_state(state)
            reward_episode = 0

            for t in range(10000):
                state_var = torch.tensor(state).unsqueeze(0)
                with torch.no_grad():
                    if mean_action:
                        action = self.policy(state_var)[0][0].numpy()
                    else:
                        action = self.policy.select_action(state_var)[0].numpy()
                action = int(action) if self.policy.is_disc_action else action.astype(np.float64)
                observation, reward, terminated, truncated, info = env.step(action)

                next_state = observation

                reward_episode += reward
                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                mask = 0 if terminated else 1

                if terminated:
                    break
                state = next_state
