#!/usr/bin/env python
# coding: utf-8

# In[1]:


# @title Run to install MuJoCo and `dm_control`
import distutils.util
import os
import subprocess

if subprocess.run("nvidia-smi").returncode:
    raise RuntimeError(
        "Cannot communicate with GPU. "
        "Make sure you are using a GPU Colab runtime. "
        "Go to the Runtime menu and select Choose runtime type."
    )

print("Installing dm_control...")
get_ipython().system("pip install -q dm_control>=1.0.18")

# Configure dm_control to use the EGL rendering backend (requires GPU)
get_ipython().run_line_magic("env", "MUJOCO_GL=egl")

print("Checking that the dm_control installation succeeded...")
try:
    from dm_control import suite

    env = suite.load("cartpole", "swingup")
    pixels = env.physics.render()
except Exception as e:
    raise e from RuntimeError(
        "Something went wrong during installation. Check the shell output above "
        "for more information.\n"
        "If using a hosted Colab runtime, make sure you enable GPU acceleration "
        'by going to the Runtime menu and selecting "Choose runtime type".'
    )
else:
    del pixels, suite

get_ipython().system(
    'echo Installed dm_control $(pip show dm_control | grep -Po "(?<=Version: ).+")'
)


# In[2]:


# @title All `dm_control` imports required for this tutorial

# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation


# In[3]:


import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import copy
from typing import Tuple

get_ipython().run_line_magic("matplotlib", "inline")
import pickle


# In[4]:


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
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


# In[5]:


def init_flags():

    flags = {
        "env": "cheetah",
        "env_action": "run",
        "seed": 0,  # random seed
        "start_timesteps": 5e3,  # total steps of free exploration phase
        "max_timesteps": 6e5,  # maximum length of time steps in training
        "batch_size": 512,
        "discount": 0.99,
        "tau": 0.005,  # rate of target update
        "policy_freq": 2,  # delayed policy update frequency in TD3,
        "N": 2,  # number of agents,
        "RR": 1,  # replay ratio,
        "T": 4e5,  # time steps between agent resets ,
        "beta": 1,  # action selection coefficient,
        "actor_model_file": None,  # "actor_model.pt",
        "critic_model_file": None,  # "critic_model.pt",
        "temp_model_file": None,  # "temp_model.pt"
    }

    return flags


def collect_actions(theta, state):
    actions = []
    for theta_i in theta:
        action = theta_i.select_action(np.array(state))
        actions.append(torch.from_numpy(action))
    return actions


def get_timestep_info(timestep):
    ob = timestep.observation
    state = np.concatenate([*ob.values()]).ravel()
    return state, timestep.reward, timestep.last()


def main(policy_name="DDPG") -> list:
    """
    Input:
    policy_name: str, the method to implement
    Output:
    evaluations: list, the reward in every episodes
    Call DDPG/TD3 trainer and
    """
    args = init_flags()
    env = suite.load(
        args["env"], args["env_action"], task_kwargs={"random": args["seed"] + 100}
    )
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    action_spec = env.action_spec()
    ob_spec = env.observation_spec()

    state_dim = 0
    for _, val in ob_spec.items():
        state_dim += val.shape[0]
    state_dim

    action_dim = action_spec.shape[0]
    max_action = action_spec.maximum
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args["discount"],
        "tau": args["tau"],
        "actor_model_file": args["actor_model_file"],
        "critic_model_file": args["critic_model_file"],
        "temp_model_file": args["temp_model_file"],
    }
    if policy_name == "TD3":
        kwargs["policy_freq"] = args["policy_freq"]
        theta = [TD3(**kwargs) for _ in range(args["N"])]
    elif policy_name == "DDPG":
        policy = DDPG(**kwargs)

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    evaluations = []
    timestep = env.reset()
    state, _, _ = get_timestep_info(timestep)
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    k = 0
    max_episode_steps = 990

    for t in range(int(args["max_timesteps"])):

        episode_timesteps += 1

        if t % 1e5 == 0:
            theta[0].save_actor_model(filename="actor_model_2.pt")
            theta[0].save_critic_model(filename="critic_model_2.pt")
            theta[0].save_temp_model(filename="temp_model_2.pt")
            np.savetxt("reward_cheetah_run_N2_RR2.csv", evaluations, delimiter=",")

        # Select action randomly or according to policy
        entropy = 0
        mean = 0
        vari = 0
        if t < args["start_timesteps"]:
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum, size=action_spec.shape
            )
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

                if t % 1000 == 0:
                    print("temperature:", torch.exp(theta[0].log_alpha))

                rng = np.random.default_rng()
                action = rng.choice(
                    a=[np.array(tensor) for tensor in actions],
                    p=p_select.numpy(),
                    axis=0,
                )
                action = np.atleast_1d(action)

        # Perform action
        next_state, reward, done = get_timestep_info(env.step(action))
        done_bool = float(done) if episode_timesteps < max_episode_steps else 0

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
                # reset just actor or both? ask to confirm
                theta[k].actor.reset()
                theta[k].critic.reset()
                k = (k + 1) % args["N"]

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            evaluations.append(episode_reward)
            # Reset environment
            done = False
            timestep = env.reset()
            state, _, _ = get_timestep_info(timestep)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    return evaluations


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Construct the actor/critic network for TD3
class Actor_TD3(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):

        super(Actor_TD3, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 2 * action_dim)

        if isinstance(max_action, np.ndarray):
            max_action = torch.from_numpy(max_action)
        else:
            assert isinstance(max_action, list)
        max_action = torch.tensor(max_action)
        self.max_action = max_action.to(torch.float32)
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        state = state.float()
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        a = self.l3(a)

        mean = a[:, : self.action_dim]
        cov = nn.functional.softplus(a[:, self.action_dim :]) + 1e-9

        dist = torch.distributions.MultivariateNormal(
            mean, scale_tril=torch.diag_embed(cov)
        )
        return dist

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


# In[7]:


class TD3(object):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount=0.99,
        tau=0.005,
        policy_freq=2,
        init_temperature=1,
        actor_model_file=None,
        critic_model_file=None,
        temp_model_file=None,
    ):

        self.actor = Actor_TD3(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic_TD3(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if actor_model_file is not None:
            print("Loading pretrained model from", actor_model_file)
            self.load_pytorch_actor_model(actor_model_file)

        if critic_model_file is not None:
            print("Loading pretrained model from", critic_model_file)
            self.load_pytorch_critic_model(critic_model_file)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -1 * action_dim
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        if temp_model_file is not None:
            print("Loading pretrained model from", temp_model_file)
            self.load_pytorch_temp_model(temp_model_file)

        self.max_action = self.actor.max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        self.total_it = 0

        self.transform = torch.distributions.transforms.ComposeTransform(
            [
                torch.distributions.transforms.TanhTransform(),
                torch.distributions.transforms.AffineTransform(
                    loc=0, scale=self.max_action
                ),
            ]
        )

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        actor_dist = self.actor(state)
        selected_action = self.transform(actor_dist.rsample())
        return selected_action.data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        temperature = torch.exp(self.log_alpha)

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to the policy,
            target_actor_dist = self.actor_target(next_state)
            raw_next_action = target_actor_dist.rsample()
            next_action = self.transform(raw_next_action)
            target_entropy = (
                target_actor_dist.entropy()
                + self.transform.log_abs_det_jacobian(raw_next_action, next_action).sum(
                    -1
                )
            ).unsqueeze(-1)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * (
                target_Q + (temperature * target_entropy)
            )
            target_Q = target_Q.detach()
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # don't have to compute critic gradients
            for param in self.critic.parameters():
                param.requires_grad = False

            actor_dist = self.actor(state)
            raw_action = actor_dist.rsample()
            selected_action = self.transform(raw_action)
            actor_entropy = (
                actor_dist.entropy()
                + self.transform.log_abs_det_jacobian(raw_action, selected_action).sum(
                    -1
                )
            ).unsqueeze(-1)

            alpha_loss = (
                temperature * (actor_entropy - self.target_entropy).mean().detach()
            )
            actor_loss = -(
                self.critic.Q1(state, selected_action)
                + (temperature.detach() * actor_entropy)
            ).mean()

            alpha_actor_loss = alpha_loss + actor_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            self.log_alpha_optimizer.zero_grad()
            alpha_actor_loss.backward()
            self.actor_optimizer.step()
            self.log_alpha_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

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

    def save_actor_model(self, filename):
        # pytorch model object like an instance of Actor
        checkpoint = {
            "model_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def save_critic_model(self, filename):
        # pytorch model object like an instance of Actor
        checkpoint = {
            "model_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def save_temp_model(self, filename):
        # pytorch model object like an instance of Actor
        checkpoint = {
            "model_state_dict": self.log_alpha,
            "optimizer_state_dict": self.log_alpha_optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load_pytorch_actor_model(self, filename):
        # now to start up a model with the same trained parameters
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["model_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # model.eval()

    def load_pytorch_critic_model(self, filename):
        # now to start up a model with the same trained parameters
        checkpoint = torch.load(filename)
        self.critic.load_state_dict(checkpoint["model_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # model.eval()

    def load_pytorch_temp_model(self, filename):
        # now to start up a model with the same trained parameters
        checkpoint = torch.load(filename)
        self.log_alpha = checkpoint["model_state_dict"]
        self.log_alpha_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # model.eval()


# In[ ]:


evaluation_td3 = main(policy_name="TD3")
