import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque

# the neural network
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.01):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(*input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, output_dim)

        # Uses Adam for optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.MSELoss() # if not working add reduction='sum'
        self.to(self.device)
    
    # This uses ReLu to maximize the network
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x

# the agent
class Agent():
    
    def __init__(self, id, gamma, epsilon, lr, input_dim, output_dim, samplesize, epsilon_decay_factor=5e-5, epsilon_min=0.01, batchMaxLength=100_000):
        self.id = id

        # creates replay memory
        self.batchMaxLength = batchMaxLength
        self.batch = deque(maxlen = self.batchMaxLength)

        # defines parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.samplesize = samplesize
        self.index = 0

        # initializes the neural network
        self.model = Model(input_dim, output_dim, lr=lr)
        self.weightPath = f"Models/{self.id}.pt"
        self.target_model = Model(input_dim, output_dim, lr=lr)
        self.sync_target_network()

        # tracks the agents current state, action, rewards, next state and terminal
        self.state_mem = np.zeros((self.batchMaxLength, *input_dim), dtype=np.float32)
        self.new_state_mem = np.zeros((self.batchMaxLength, *input_dim), dtype=np.float32)
        self.action_mem = np.zeros(self.batchMaxLength, dtype=np.int32)
        self.reward_mem = np.zeros(self.batchMaxLength, dtype=np.float32)
        self.terminal_mem = np.zeros(self.batchMaxLength, dtype=bool)

        # target model update interval
        self.target_update_interval = 1000

        # the number of steps for each update
        self.step_update = 10
    
    # selects an action from the current state
    def act(self, obs):    

        # decides wether to prioritize exploration or exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 39)
        else:
            state = torch.tensor([obs], dtype=torch.float32).to(self.model.device)
            # does not calculate gradients for the action, which is not needed
            with torch.no_grad(): 
                actions = self.model.forward(state)
            return torch.argmax(actions).item()
    
    # stores the experience in the batch
    def updateBatch(self, state, action, reward, new_state, done):
        i = self.index % self.batchMaxLength
        self.state_mem[i] = state
        self.new_state_mem[i] = new_state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.terminal_mem[i] = bool(done)

        self.index += 1
    
    # copies the weights of the model to the target model
    def sync_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # learns from the experience
    def experience(self):
        if self.index < self.samplesize:
            return

        if self.index % self.step_update != 0:
            return

        self.model.optimizer.zero_grad()

        # Sample random batch
        max_size = min(self.batchMaxLength, self.index)
        batch = np.random.choice(max_size, self.samplesize, replace=False)

        batch_index = np.arange(self.samplesize, dtype=np.int32)

        # Convert data to tensors
        state_batch = torch.tensor(self.state_mem[batch]).to(self.model.device)
        action_batch = self.action_mem[batch]
        new_state_batch = torch.tensor(self.new_state_mem[batch]).to(self.model.device)
        reward_batch = torch.tensor(self.reward_mem[batch]).to(self.model.device)
        terminal_batch = torch.tensor(self.terminal_mem[batch], dtype=torch.bool).to(self.model.device)

        # Compute Q-values for the selected actions
        q_values = self.model.forward(state_batch)[batch_index, action_batch]

        # Use the policy network to select the best actions
        next_actions = torch.argmax(self.model.forward(new_state_batch), dim=1)

        # Use the target network to evaluate the Q-value of the selected actions
        q_values_next = self.target_model.forward(new_state_batch)[batch_index, next_actions]
        q_values_next[terminal_batch] = 0.0

        # Compute the target Q-values
        q_target = reward_batch + self.gamma * q_values_next

        # Compute loss
        loss = self.model.loss_function(q_target, q_values)

        # Backpropagate loss
        loss.backward()
        self.model.optimizer.step()

        # Periodically update the target network
        if self.index % self.target_update_interval == 0:
            self.sync_target_network()

        return loss.item()
    
    def updateEpsilon(self):
        self.epsilon = (self.epsilon - self.epsilon_min) * self.epsilon_decay_factor + self.epsilon_min
    
    def evaluate(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.model.device)
        # does not calculate gradients for the action, which is not needed
        with torch.no_grad(): 
            actions = self.model.forward(state)
        return torch.argmax(actions).item()

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
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.model.optimizer.state_dict()
        }, self.weightPath)
