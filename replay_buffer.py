from typing import Union, Any, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer as ReplayBufferSB3
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

def softmax(x):
    x -= np.max(x)
    return np.exp(x)/np.exp(x).sum()

class ReplayBuffer(ReplayBufferSB3):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage,
                         handle_timeout_termination)
        self.empirical_count = np.zeros(self.buffer_size)
        self.probs = softmax(np.zeros(self.buffer_size))

    def sample_adaptive(self, batch_size: int, learning_starts: int, temperature: float, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        desired_count = batch_size/(learning_starts + self.size() + 1) * (self.size() + 1)
        diff = desired_count - self.empirical_count
        # if self.size() % 100 == 0:
        self.probs = softmax(diff[:self.size()] / batch_size * 1/temperature)

        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.choice(np.arange(0, upper_bound), p=self.probs, size=batch_size)
            self.empirical_count[batch_inds] += 1
            return self._get_samples(batch_inds, env=env)

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            # batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
            batch_inds = (np.random.choice(np.arange(1, self.buffer_size), p=self.probs) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.choice(np.arange(0, self.pos), p=self.probs, size=batch_size)

        self.empirical_count[batch_inds] += 1
        return self._get_samples(batch_inds, env=env)

