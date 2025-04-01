from collections import defaultdict

import numpy as np
import torch


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, hidden_dims=None):
        super().__init__()

        # Default to [256, 256] if hidden_dims is None
        if hidden_dims is None:
            hidden_dims = [256, 256]

        input_dim = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)

        # Build Q network
        q_layers = []

        # First layer
        q_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        q_layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            q_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            q_layers.append(nn.ReLU())

        # Output layer
        q_layers.append(nn.Linear(hidden_dims[-1], 1))

        self.q_net = nn.Sequential(*q_layers)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.q_net(x)


class Actor(nn.Module):
    def __init__(self, env, hidden_dims=None):
        super().__init__()

        # Default to [256, 256] if hidden_dims is None
        if hidden_dims is None:
            hidden_dims = [256, 256]

        input_dim = np.array(env.single_observation_space.shape).prod()
        output_dim = np.prod(env.single_action_space.shape)

        # Build policy network
        policy_layers = []

        # First layer
        policy_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        policy_layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            policy_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            policy_layers.append(nn.ReLU())

        # Output layer
        policy_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        policy_layers.append(nn.Tanh())

        self.policy_net = nn.Sequential(*policy_layers)

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.policy_net(x)
        return x * self.action_scale + self.action_bias

    def get_action(self, x, sample=True):
        # For compatibility with simulate function from PPO
        action = self.forward(x)
        # Note that we don't add sampling noise here during evaluation
        return action

def simulate(env, actor, eval_episodes, eval_steps=np.inf):
    logs = defaultdict(list)
    step = 0
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False

        while not done:

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                actions = actor.get_action(torch.Tensor(obs).to('cpu'), sample=False)
                actions = actions.cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = np.logical_or(terminateds, truncateds)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            logs_episode['rewards'].append(rewards)

            step += 1

            if step >= eval_steps:
                break
        if step >= eval_steps:
            break

        logs['returns'].append(np.sum(logs_episode['rewards']))
        try:
            logs['successes'].append(infos['is_success'])
        except:
            logs['successes'].append(False)

    return_avg = np.mean(logs['returns'])
    return_std = np.std(logs['returns'])
    success_avg = np.mean(logs['successes'])
    success_std = np.std(logs['successes'])
    return return_avg, return_std, success_avg, success_std
