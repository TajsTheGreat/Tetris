import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque

H = 250

class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
loss_function = nn.MSELoss() # if not working add reduction='sum'

device = "cpu"

class Agent():
    theTerminalState = False
    
    def __init__(self, id):
        self.id = id

        self.batchMaxLength = 100_000
        self.batch = deque(maxlen = self.batchMaxLength)
        self.batch_size = 0
        self.replay_buffer = ReplayBuffer(self.batchMaxLength)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 10**-7 
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.update_rate = 1000
        self.update_counter = 0

        self.model = Model(13, 4).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.weightPath = f"Models/{self.id}.pt"

        self.state_memory = 
    
    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            state = torch.tensor(state).to(device)
            return torch.argmax(self.model(state)).item()

    def loadWeights(self):
        try:
            checkpoint = torch.load(self.weightPath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            pass

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
    
def train_func():
    pass

