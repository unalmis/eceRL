#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system("pip3 install swig > /dev/null 2>&1")
get_ipython().system("pip3 uninstall box2d-py -y > /dev/null 2>&1")
get_ipython().system("pip3 install box2d-py > /dev/null 2>&1")
get_ipython().system("pip3 install box2d box2d-kengz > /dev/null 2>&1")
get_ipython().system("apt install xvfb > /dev/null 2>&1")
get_ipython().system("pip3 install pyvirtualdisplay > /dev/null 2>&1")
get_ipython().system("pip3 install gym==0.25.0 > /dev/null 2>&1")


# In[ ]:


import gym
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from pyvirtualdisplay import Display
from IPython import display as disp
import copy
from typing import Tuple

get_ipython().run_line_magic("matplotlib", "inline")


# In[ ]:


# Replay buffer
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )


# In[ ]:


def init_flags():

    flags = {
        "env": "MountainCarContinuous",
        "seed": 0,  # random seed
        "start_timesteps": 5e3,  # total steps of free exploration phase
        "max_timesteps": 2e5,  # maximum length of time steps in training
        "expl_noise": 0.1,  # noise strength in exploration
        "batch_size": 512,
        "discount": 0.99,
        "tau": 0.005,  # rate of target update
        "policy_freq": 2,  # delayed policy update frequency in TD3,
        "N": 1,  # number of agents,
        "RR": 2,  # replay ratio,
        "T": np.inf,  # time steps between agent resets (every 8e4 for RR=1) ,
        "beta": 50,  # action selection coefficient
    }

    return flags


def collect_actions(theta, state):
    actions = []
    for theta_i in theta:
        action, entropy, mean, vari = theta_i.select_action(np.array(state))
        actions.append(torch.from_numpy(action))
    return actions


def main(policy_name="DDPG") -> list:
    """
    Input:
    policy_name: str, the method to implement
    Output:
    evaluations: list, the reward in every episodes
    Call DDPG/TD3 trainer and
    """
    args = init_flags()
    env = gym.make(args["env"])
    env.seed(args["seed"] + 100)
    env.action_space.seed(args["seed"])
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args["discount"],
        "tau": args["tau"],
    }
    if policy_name == "TD3":
        kwargs["policy_freq"] = args["policy_freq"]
        theta = [TD3(**kwargs) for _ in range(args["N"])]
    elif policy_name == "DDPG":
        policy = DDPG(**kwargs)

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    evaluations = []
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    k = 0

    for t in range(int(args["max_timesteps"])):

        episode_timesteps += 1

        # Select action randomly or according to policy
        entropy = 0
        mean = 0
        vari = 0
        if t < args["start_timesteps"]:
            action = env.action_space.sample()
        else:
            with torch.no_grad():

                actions = collect_actions(theta, state)
                # compute Q's, then apply softmax
                q_sa = torch.hstack(
                    [
                        theta[k].critic.Q1(torch.FloatTensor(state), action)
                        for action in actions
                    ]
                )
                # dim
                max_q_sa, _ = torch.max(q_sa, dim=0)
                alpha = args["beta"] / max_q_sa
                p_select = F.softmax(q_sa / alpha)

                if t == args["start_timesteps"]:
                    print(p_select)

                action = np.random.choice(
                    a=torch.hstack(actions).numpy(), p=p_select.numpy()
                )
                action = np.atleast_1d(action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args["start_timesteps"]:
            for j in range(args["RR"]):
                for theta_i in theta:
                    theta_i.train(replay_buffer, args["batch_size"])

            if (t % (args["T"] / args["N"])) == 0:
                print(k)
                # reset just actor or both?
                theta[k].actor.reset()
                theta[k].critic.reset()
                k = (k + 1) % args["N"]

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            evaluations.append(episode_reward)
            entropies = []
            actions_l = []
            means = []
            varis = []
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    return evaluations


# In[ ]:


# Reference Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Construct the actor/critic network for TD3
class Actor_TD3(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor_TD3, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 2 * action_dim)
        self.max_action = max_action
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)

        mean = self.max_action * torch.tanh(a[:, : self.action_dim])
        std = nn.functional.softplus(a[:, self.action_dim :]) + 1e-9

        return mean, std

    def reset(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class Critic_TD3(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic_TD3, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], -1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def reset(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


# In[ ]:


class TD3(object):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount=0.99,
        tau=0.005,
        policy_freq=2,
        temperature=0.01,
    ):

        self.actor = Actor_TD3(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic_TD3(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        self.total_it = 0
        self.temperature = temperature

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        mean, std = self.actor(state)
        actor_dist = torch.distributions.Normal(mean, std)
        selected_action = actor_dist.rsample().clamp(-self.max_action, self.max_action)
        entropy = actor_dist.entropy()
        vari = torch.square(std)
        return (
            selected_action.data.numpy().flatten(),
            entropy.data.numpy(),
            mean.data.numpy(),
            vari.data.numpy(),
        )

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to the policy,
            target_mean, target_std = self.actor_target(next_state)
            target_actor_dist = torch.distributions.Normal(target_mean, target_std)
            next_action = target_actor_dist.rsample().clamp(
                -self.max_action, self.max_action
            )
            target_entropy = target_actor_dist.entropy()
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * (
                target_Q + (self.temperature * target_entropy)
            )
            target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            mean, std = self.actor(state)
            actor_dist = torch.distributions.Normal(mean, std)
            selected_action = actor_dist.rsample().clamp(
                -self.max_action, self.max_action
            )

            actor_loss = -(
                self.critic.Q1(state, selected_action)
                + (self.temperature * actor_dist.entropy())
            ).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                new_target_params = (
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
                target_param.data.copy_(new_target_params)

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                new_target_params = (
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
                target_param.data.copy_(new_target_params)


# In[ ]:


evaluation_td3 = main(policy_name="TD3")
