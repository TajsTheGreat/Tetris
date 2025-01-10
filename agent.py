import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque

H = 250

class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.01):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(*input_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, output_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.MSELoss() # if not working add reduction='sum'
        self.to(self.device)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent():
    theTerminalState = False
    
    def __init__(self, id, gamma, epsilon, lr, input_dim, output_dim, samplesize, epsilon_decay=5e-4, epsilon_min=0.01, batchMaxLength=100_000):
        self.id = id

        self.batchMaxLength = batchMaxLength
        self.batch = deque(maxlen = self.batchMaxLength)
        self.replay_buffer = ReplayBuffer(self.batchMaxLength)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.samplesize = samplesize
        self.index = 0

        self.model = Model(input_dim, output_dim, lr=lr)
        self.weightPath = f"Models/{self.id}.pt"

        self.state_mem = np.zeros((self.batchMaxLength, *input_dim), dtype=np.float32)
        self.new_state_mem = np.zeros((self.batchMaxLength, *input_dim), dtype=np.float32)
        self.action_mem = np.zeros(self.batchMaxLength, dtype=np.int32)
        self.reward_mem = np.zeros(self.batchMaxLength, dtype=np.float32)
        self.terminal_mem = np.zeros(self.batchMaxLength, dtype=np.bool)
    
    def act(self, obs):    
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 39)
        else:
            state = torch.tensor([obs]).to(self.model.device)
            actions = self.model.forward(state)
            return torch.argmax(actions).item()
    
    # this stores the experience in the batch
    def updateBatch(self, state, action, reward, new_state, done):
        i = self.index % self.batchMaxLength
        self.state_mem[i] = state
        self.new_state_mem[i] = new_state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.terminal_mem[i] = done

        self.index += 1
    
    # learns from the experience
    def experience(self):
        if self.index < self.samplesize:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        
        self.model.optimizer.zero_grad()
        
        max_size = min(self.batchMaxLength, self.index)
        batch = np.random.choice(max_size, self.samplesize, replace=False)

        batch_index = np.arange(self.samplesize, dtype=np.int32)

        state_batch = torch.tensor(self.state_mem[batch]).to(self.model.device)

        action_batch = self.action_mem[batch]

        new_state_batch = torch.tensor(self.state_mem[batch]).to(self.model.device)
        reward_batch = torch.tensor(self.state_mem[batch]).to(self.model.device)
        terminal_batch = torch.tensor(self.state_mem[batch]).to(self.model.device)

        q_values = self.model.forward(state_batch)[batch_index, action_batch]
        q_values_next = self.model.forward(new_state_batch)
        q_values_next[terminal_batch] = 0.0

        q_reward = reward_batch + self.gamma * torch.max(q_values_next, dim=1)[0]

        loss_funtion = self.model.loss_function(q_reward, q_values).to(self.model.device)
        loss_funtion.backward()
        self.model.optimizer.step()

    # this loads the weights of the model
    def loadWeights(self):
        try:
            checkpoint = torch.load(self.weightPath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            pass

    # this saves the weights of the model
    def saveWeights(self):
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
        }, self.weightPath)

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)