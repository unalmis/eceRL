#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system("apt-get install x11-utils > /dev/null 2>&1")
get_ipython().system("pip install pyglet > /dev/null 2>&1")
get_ipython().system("apt-get install -y xvfb python-opengl > /dev/null 2>&1")

get_ipython().system("pip install gym pyvirtualdisplay > /dev/null 2>&1")


# In[ ]:


import os
import pdb
import sys
import copy
import json
import argparse
from datetime import datetime

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay


class DQN(nn.Module):

    def __init__(self, input, hidden, output):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def reset(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class QNetwork:

    def __init__(self, args, input, output, learning_rate):
        self.weights_path = "models/%s/%s" % (
            args["env"],
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        # Network architecture.
        self.hidden = 128
        self.model = DQN(input, self.hidden, output)

        # Loss and optimizer.
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if args["model_file"] is not None:
            print("Loading pretrained model from", args["model_file"])
            self.load_model_weights(args["model_file"])

    def save_model_weights(self, step, i):
        # Helper function to save your model / weights.
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.weights_path, f"model_{step}_{i}.h5"),
        )

    def load_model_weights(self, weight_file):
        # Helper function to load model weights.
        self.model.load_state_dict(torch.load(weight_file))


# In[ ]:


class Replay_Memory:
    def __init__(self, state_dim, action_dim, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) way to implement the memory is as a list of transitions.
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.states = torch.zeros((self.memory_size, state_dim))
        self.next_states = torch.zeros((self.memory_size, state_dim))
        self.actions = torch.zeros((self.memory_size, 1))
        self.rewards = torch.zeros((self.memory_size, 1))
        self.dones = torch.zeros((self.memory_size, 1))
        self.ptr = 0
        self.burned_in = False
        self.not_full_yet = True

    def append(self, states, actions, rewards, next_states, dones):
        self.states[self.ptr] = states
        self.actions[self.ptr, 0] = actions
        self.rewards[self.ptr, 0] = rewards
        self.next_states[self.ptr] = next_states
        self.dones[self.ptr, 0] = dones
        self.ptr += 1

        if self.ptr > self.burn_in:
            self.burned_in = True

        if self.ptr >= self.memory_size:
            self.ptr = 0
            self.not_full_yet = False

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        if self.not_full_yet:
            idxs = torch.from_numpy(np.random.choice(self.ptr, batch_size, False))
        else:
            idxs = torch.from_numpy(
                np.random.choice(self.memory_size, batch_size, False)
            )

        states = self.states[idxs]
        next_states = self.next_states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]
        return states, actions, rewards, next_states, dones


# In[ ]:


class DQN_Agent:
    def __init__(self, args, env):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        # Inputs
        self.args = args
        self.env = env
        self.environment_name = args["env"]
        self.render = self.args["render"]
        self.epsilon = args["epsilon"]
        self.network_update_freq = args["network_update_freq"]
        self.log_freq = args["log_freq"]
        self.test_freq = args["test_freq"]
        self.save_freq = args["save_freq"]
        self.learning_rate = args["learning_rate"]

        # Other Classes
        self.q_network = QNetwork(
            args,
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            self.learning_rate,
        )
        self.target_q_network = QNetwork(
            args,
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            self.learning_rate,
        )
        self.batch = list(range(32))

        # Save hyperparameters
        self.logdir = "logs/%s/%s" % (
            self.environment_name,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        with open(self.logdir + "/hyperparameters.json", "w") as outfile:
            json.dump((self.args), outfile, indent=4)

    def epsilon_greedy_policy(self, q_values, epsilon):
        # Creating epsilon greedy probabilities to sample from.

        # choose random action a fraction epsilon of the time
        # and a greedy action the rest of the time
        sample = np.random.rand()
        if sample < epsilon:
            return self.env.action_space.sample()
        else:
            return torch.argmax(q_values).item()

    def greedy_policy(self, q_values):
        return torch.argmax(q_values).item()

    def td_estimate(self, state, action):
        # pass through q_network to get Q values
        Q_values = self.q_network.model.forward(state)
        action = action.long()
        return Q_values.gather(1, action)

    def td_target(self, reward, next_state, done, discount_factor):
        # pass through target_q_network and take maximum over Q values
        Q_values = self.target_q_network.model.forward(next_state)
        max_Q_values, _ = torch.max(Q_values, dim=1, keepdim=True)

        # compute td_target
        return reward + discount_factor * (1 - done) * max_Q_values

    def train_dqn(self, memory, discount_factor):
        # Sample from the replay buffer.
        state, action, rewards, next_state, done = memory.sample_batch(batch_size=32)

        # Optimization step.
        # For reference, we used F.smooth_l1_loss as our loss function.
        self.q_network.optim.zero_grad()
        loss = F.smooth_l1_loss(
            self.td_estimate(state, action),
            self.td_target(rewards, next_state, done, discount_factor),
        )
        loss.backward()
        self.q_network.optim.step()

        return loss

    def hard_update(self):
        self.target_q_network.model.load_state_dict(self.q_network.model.state_dict())

    @classmethod
    def plots(cls, reward, td_error):
        """
        Plots:
        1) Avg Cummulative Test Reward over 20 Plots
        2) TD Error
        #"""
        reward, time = zip(*rewards)
        plt.figure(figsize=(8, 3))
        plt.subplot(121)
        plt.title("Cummulative Reward")
        plt.plot(time, reward)
        plt.xlabel("iterations")
        plt.ylabel("rewards")
        plt.legend()
        plt.ylim([0, None])

        loss, time = zip(*td_error)
        plt.subplot(122)
        plt.title("Loss")
        plt.plot(time, loss)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.show()

    def epsilon_decay(self, initial_eps=1.0, final_eps=0.05):
        if self.epsilon > final_eps:
            factor = (initial_eps - final_eps) / 10000
            self.epsilon -= factor


# In[ ]:


def init_flags():

    flags = {
        "env": "CartPole-v0",  # Change to "MountainCar-v0" when needed.
        "render": False,
        "train": 1,
        "frameskip": 1,
        "network_update_freq": 10,
        "log_freq": 5,
        "test_freq": 20,
        "save_freq": 500,
        "learning_rate": 5e-4,
        "memory_size": 50000,
        "epsilon": 0.5,
        "model_file": None,
        "N": 2,  # number of agents,
        "RR": 1,  # replay ratio,
        "T": 8e4,  # time steps between agent resets ,
        "beta": 0.01,  # action selection coefficient
    }

    return flags


def get_action(theta, k, epsilon, beta, state, model):

    actions = []
    for agent in theta:
        q_value_i = agent.q_network.model.forward((state.reshape(1, -1)))
        action = agent.epsilon_greedy_policy(q_value_i, epsilon)
        actions.append(action)

    # now we choose the max q value over our ensemble of agents
    q_sa = torch.hstack(
        [
            theta[k].q_network.model.forward((state.reshape(1, -1)))[:, a]
            for a in actions
        ]
    )
    max_q_sa, _ = torch.max(q_sa, dim=0)
    alpha = beta / max_q_sa
    p_select = F.softmax(q_sa / alpha)

    action = np.random.choice(a=actions, p=p_select.numpy())

    return action, p_select


def test_(
    theta,
    env,
    beta,
    T,
    k,
    N,
    RR,
    memory,
    frameskip,
    discount_factor,
    t,
    model_file=None,
    episodes=100,
):
    # Evaluate the performance of your agent over 100 episodes, by calculating cumulative rewards for the 100 episodes.
    # Here you need to interact with the environment, irrespective of whether you are using a memory.
    cum_reward = []
    td_error = []
    for count in range(episodes):
        reward, error, _, _ = generate_episode(
            theta,
            mode="test",
            env=env,
            epsilon=0.05,
            beta=beta,
            T=T,
            k=k,
            N=N,
            RR=RR,
            memory=memory,
            frameskip=frameskip,
            discount_factor=discount_factor,
            t=t,
        )
        cum_reward.append(reward)
        td_error.append(error)
    cum_reward = torch.tensor(cum_reward)
    td_error = torch.tensor(td_error)
    print(
        "\nTest Rewards: {0} | TD Error: {1:.4f}\n".format(
            torch.mean(cum_reward), torch.mean(td_error)
        )
    )
    return torch.mean(cum_reward), torch.mean(td_error)


def burn_in_memory(
    theta,
    epsilon,
    beta,
    env,
    T,
    k,
    N,
    RR,
    memory,
    discount_factor,
    t,
    mode="train",
    frameskip=1,
):
    # Initialize your replay memory with a burn_in number of episodes / transitions.
    while not memory.burned_in:
        _, _, t, _ = generate_episode(
            theta,
            mode="burn_in",
            env=env,
            epsilon=epsilon,
            beta=beta,
            T=T,
            k=k,
            N=N,
            RR=RR,
            memory=memory,
            frameskip=frameskip,
            discount_factor=discount_factor,
            t=t,
        )
    print("Burn Complete!")

    return t


def generate_episode(
    theta,
    epsilon,
    beta,
    env,
    T,
    k,
    N,
    RR,
    memory,
    discount_factor,
    t,
    mode="train",
    frameskip=1,
):
    """
    Collects one rollout from the policy in an environment.
    """
    done = False
    state = torch.from_numpy(env.reset())
    rewards = 0

    td_error = []
    while not done:
        with torch.no_grad():
            action, p_select = get_action(theta, k, epsilon, beta, state, mode)
        i = 0
        while (i < frameskip) and not done:
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state)
            rewards += reward
            i += 1

        if mode == "train":
            if t % 1000 == 0:
                print(p_select)

        if mode in ["train", "burn_in"]:
            memory.append(state, action, reward, next_state, done)
        if not done:
            state = copy.deepcopy(next_state.detach())

        # Train the network.
        if mode == "train":
            t += 1

            for j in range(RR):
                for theta_i in theta:
                    theta_i.train_dqn(memory, discount_factor)

            if (t % (T / N)) == 0:
                print(k)
                theta[k].q_network.model.reset()
                theta[k].target_q_network.model.reset()
                k = (k + 1) % N

    if td_error == []:
        return rewards, [], t, k
    return rewards, torch.mean(torch.stack(td_error)), t, k


def main(render=False):
    args = init_flags()
    args["render"] = render

    if args["env"] == "CartPole-v0":
        env = gym.make(args["env"], render_mode="rgb_array")
        discount_factor = 0.99
        num_episodes = 1000
        max_timesteps = 2e5
    else:
        raise Exception("Unknown Environment")

    memory = Replay_Memory(
        env.observation_space.shape[0],
        env.action_space.n,
        memory_size=args["memory_size"],
    )

    theta = [DQN_Agent(args, env) for _ in range(args["N"])]

    # time step
    t = 0
    # number of episodes
    step = 0
    # current agent being reset
    k = 0

    burn_in_memory(
        theta,
        epsilon=args["epsilon"],
        beta=args["beta"],
        env=env,
        T=args["T"],
        k=k,
        N=args["N"],
        RR=args["RR"],
        memory=memory,
        mode="train",
        frameskip=args["frameskip"],
        discount_factor=discount_factor,
        t=t,
    )

    rewards = []
    td_error = []

    while t < max_timesteps:
        # Generate Episodes using Epsilon Greedy Policy and train the Q network.
        step += 1
        _, _, t, k = generate_episode(
            theta,
            mode="train",
            env=env,
            epsilon=args["epsilon"],
            beta=args["beta"],
            T=args["T"],
            k=k,
            N=args["N"],
            RR=args["RR"],
            memory=memory,
            frameskip=args["frameskip"],
            discount_factor=discount_factor,
            t=t,
        )

        # Test the network.
        if step % args["test_freq"] == 0:
            print("here")
            test_reward, test_error = test_(
                theta,
                env=env,
                beta=args["beta"],
                N=args["N"],
                T=args["T"],
                k=k,
                RR=args["RR"],
                memory=memory,
                discount_factor=discount_factor,
                t=t,
                frameskip=args["frameskip"],
                model_file=None,
                episodes=20,
            )
            rewards.append([test_reward, step])
            td_error.append([test_error, step])

        # Update the target network.
        if step % args["network_update_freq"] == 0:
            for theta_i in theta:
                theta_i.hard_update()

        # Logging.
        if step % args["log_freq"] == 0:
            print("Step: {0:05d}/{1:05d}".format(step, num_episodes))

        # Save the model
        if step % args["save_freq"] == 0:
            for i, theta_i in enumerate(theta):
                theta_i.q_network.save_model_weights(step, i)

        # step += 1
        for theta_i in theta:
            theta_i.epsilon_decay()

    return rewards, td_error


# In[ ]:


rewards, td_error = main()
DQN_Agent.plots(rewards, td_error)
