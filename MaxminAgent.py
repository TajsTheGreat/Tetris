import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque

# amount of neurons in each hidden layer
H = 200

# the neural network
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.01):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(*input_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, H)
        self.fc5 = nn.Linear(H, output_dim)

        # Uses Adam for optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.MSELoss() # if not working add reduction='sum'
        self.to(self.device)
    
    # Uses ReLU activation function to output raw Q-values
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x

# the agent
class Agent():
    def __init__(self, id, gamma, epsilon, lr, input_dim, output_dim, samplesize, n_networks=3, epsilon_decay_factor=5e-5, epsilon_min=0.01, batchMaxLength=100_000):
        self.id = id

        # Replay memory
        self.batchMaxLength = batchMaxLength
        self.batch = deque(maxlen=self.batchMaxLength)

        # Parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.samplesize = samplesize
        self.index = 0
        self.n_networks = n_networks

        # Initialize multiple Q-networks
        self.models = [Model(input_dim, output_dim, lr=lr) for _ in range(self.n_networks)]

        # Memory buffers
        self.state_mem = np.zeros((self.batchMaxLength, *input_dim), dtype=np.float32)
        self.new_state_mem = np.zeros((self.batchMaxLength, *input_dim), dtype=np.float32)
        self.action_mem = np.zeros(self.batchMaxLength, dtype=np.int32)
        self.reward_mem = np.zeros(self.batchMaxLength, dtype=np.float32)
        self.terminal_mem = np.zeros(self.batchMaxLength, dtype=bool)

        self.step_update = 10
    
    def act(self, obs):    
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 39), False
        else:
            state = torch.tensor([obs], dtype=torch.float32).to(self.models[0].device)
            with torch.no_grad():
                actions = [model.forward(state) for model in self.models]
                avg_actions = torch.mean(torch.stack(actions), dim=0)
            chosen_action = torch.argmax(avg_actions).item()
            chosen_q_value = avg_actions[0][chosen_action].item()
            return chosen_action, chosen_q_value

    def updateBatch(self, state, action, reward, new_state, done):
        i = self.index % self.batchMaxLength
        self.state_mem[i] = state
        self.new_state_mem[i] = new_state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.terminal_mem[i] = bool(done)

        self.index += 1
    
    def experience(self):
        if self.index < self.samplesize:
            return
        if self.index % self.step_update != 0:
            return
        
        max_size = min(self.batchMaxLength, self.index)
        batch = np.random.choice(max_size, self.samplesize, replace=False)
        batch_index = np.arange(self.samplesize, dtype=np.int32)

        state_batch = torch.tensor(self.state_mem[batch]).to(self.models[0].device)
        action_batch = self.action_mem[batch]
        new_state_batch = torch.tensor(self.new_state_mem[batch]).to(self.models[0].device)
        reward_batch = torch.tensor(self.reward_mem[batch]).to(self.models[0].device)
        terminal_batch = torch.tensor(self.terminal_mem[batch], dtype=torch.bool).to(self.models[0].device)

        # Calculate Q-values
        model_to_update = random.choice(self.models)
        model_to_update.optimizer.zero_grad()

        q_values = model_to_update.forward(state_batch)[batch_index, action_batch]
        
        # Compute min of Q-values from all models
        q_values_next = torch.stack([model.forward(new_state_batch) for model in self.models])
        q_values_next_min = torch.min(q_values_next, dim=0)[0]
        q_values_next_min[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_values_next_min, dim=1)[0]

        loss = model_to_update.loss_function(q_target, q_values)
        loss.backward()
        model_to_update.optimizer.step()
        return loss.item()

    def updateEpsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay_factor if self.epsilon > self.epsilon_min else self.epsilon_min

    def evaluate(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.models[0].device)
        with torch.no_grad():
            actions = [model.forward(state) for model in self.models]
            avg_actions = torch.mean(torch.stack(actions), dim=0)
        return torch.argmax(avg_actions).item()

    # this loads the weights of the model
    def loadWeights(self):
        try:
            checkpoint = torch.load(self.weightPath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Error: No weights found")
            pass

    # this saves the weights of the model
    def saveWeights(self):
        torch.save({
        'model_state_dict': self.models.state_dict(),
        'optimizer_state_dict': self.models.optimizer.state_dict()
        }, self.weightPath)
