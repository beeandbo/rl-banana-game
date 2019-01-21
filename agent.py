import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from deep_q_model import DeepQModel
import random
from collections import deque
import random

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-5               # learning rate
UPDATE_EVERY = 4        # how often to update the network
EPSILON_START = 1       # Epsilon params used in eps-greedy policy
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, seed, eps=EPSILON_START):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.local_model = DeepQModel(state_size, action_size, seed).to(device)
        self.target_model = DeepQModel(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=LR)

        # Replay memory
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.eps = eps
        self.steps = 0
        self.last_action = None
        self.training = True

    def act(self, state, explore = True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            explore (boolean): If true, explores a random action with self.eps probability
        """
        self.steps += 1

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_model.eval()
        with torch.no_grad():
            action_values = self.local_model.forward(state)
        self.local_model.train()
        # Epsilon-greedy action selection
        if not explore or random.random() > self.eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        return action

    def vectorize_experiences(self, experiences):
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def save(self, filename):
        torch.save(self.local_model.state_dict(), filename)

    def load(self, filename):
        self.local_model.load_state_dict(torch.load(filename))

        # Sync target with local
        self.target_model.load_state_dict(torch.load(filename))

    def train_model(self, experiences):
        states, actions, rewards, next_states, dones = self.vectorize_experiences(experiences)

        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.local_model(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        for target_param, local_param in zip(self.target_model.parameters(), self.local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

    def reset_episode(self):
        self.eps = max(EPSILON_MIN, self.eps * EPSILON_DECAY)

    def get_epsilon(self):
        return(self.eps)

    def learn(self, experience):
        """Update agent with the results of an experience.

        Params
        ======
            experience (Experience): Results of taking an action in the environment
        """

        self.memory.append(experience)
        if self.steps % UPDATE_EVERY == 0 and len(self.memory) >= BATCH_SIZE:
            self.train_model(random.sample(self.memory, k=BATCH_SIZE))

        if experience.done:
            self.reset_episode()
