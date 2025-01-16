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
    
    def __init__(self, id, gamma, epsilon, lr, input_dim, output_dim, samplesize, epsilon_decay_factor=5e-5, epsilon_min=0.01, batchMaxLength=100_000, alpha=0.6, beta_start=0.4, beta_increment=1e-4):
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
        self.alpha = alpha

        self.beta = beta_start
        self.beta_increment = beta_increment

        # initializes the neural network
        self.model = Model(input_dim, output_dim, lr=lr)
        self.weightPath = f"Models/{self.id}.pt"

        # tracks the agents current state, action, rewards, next state and terminal
        self.state_mem = np.zeros((self.batchMaxLength, *input_dim), dtype=np.float32)
        self.new_state_mem = np.zeros((self.batchMaxLength, *input_dim), dtype=np.float32)
        self.action_mem = np.zeros(self.batchMaxLength, dtype=np.int32)
        self.reward_mem = np.zeros(self.batchMaxLength, dtype=np.float32)
        self.terminal_mem = np.zeros(self.batchMaxLength, dtype=bool)
        self.priorities = np.zeros((batchMaxLength,), dtype=np.float32)


        # experience modulus thing
        self.step_update = 10
    
    # selects an action from the current state
    def act(self, obs):    
        # decides wether to prioritize exploration or exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 39), False
        else:
            state = torch.tensor([obs], dtype=torch.float32).to(self.model.device)
            with torch.no_grad():
                actions = self.model.forward(state)
            chosen_action = torch.argmax(actions).item()
            # Extracts q-values from the model for plotting
            chosen_q_value = actions[0, chosen_action].item()
            return chosen_action, chosen_q_value
    
    # stores the experience in the batch
    def updateBatch(self, state, action, reward, new_state, done):
        i = self.index % self.batchMaxLength
        state_tensor = torch.tensor(state, dtype=torch.float32)
        new_state_tensor = torch.tensor(new_state, dtype=torch.float32)

        # calculates the error (uses torch.no_grad() because speedy)
        with torch.no_grad():
            target = reward + self.gamma * torch.max(self.model(new_state_tensor)) * (1 - done)
            current = self.model(state_tensor)[action]
            error = abs(target - current).item()

        # Add to memory
        self.state_mem[i] = state
        self.new_state_mem[i] = new_state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.terminal_mem[i] = done
        self.priorities[i] = (error + 1e-5) ** self.alpha

        self.index += 1

    # learns from the experience
    def experience(self):
        if self.index < self.samplesize:
            return

        if self.index % self.step_update != 0:
            return

        self.model.optimizer.zero_grad()

        # Sample from buffer with the priorities
        if self.index >= self.batchMaxLength:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.index]

        probabilities = priorities / priorities.sum()
        indices = np.random.choice(self.index if self.index <= self.batchMaxLength else self.batchMaxLength, self.samplesize, p=probabilities)

        weights = (self.index * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        states = self.state_mem[indices]
        next_states = self.new_state_mem[indices]
        actions = self.action_mem[indices]
        rewards = self.reward_mem[indices]
        dones = self.terminal_mem[indices]

        # Saves the data in a way pytorch can understand
        states = torch.tensor(states, dtype=torch.float32).to(self.model.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.model.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.model.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.model.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.model.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.model.device)

        # cal the q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_values_next = self.model(next_states).max(1)[0]
        q_values_next[dones] = 0.0
        q_target = rewards + self.gamma * q_values_next

        # Calculate loss
        errors = q_target - q_values
        loss = (weights * errors ** 2).mean()

        loss.backward()
        self.model.optimizer.step()

        # Update priorities and beta
        self.update_priorities(indices, errors.detach().cpu().numpy())
        self.beta = min(1.0, self.beta + self.beta_increment)

        return loss.item()
    
    def updateEpsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay_factor if self.epsilon > self.epsilon_min else self.epsilon_min
    
    def evaluate(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.model.device)
        with torch.no_grad():
            actions = self.model.forward(state)
        return torch.argmax(actions).item()
    
    def update_priorities(self, batch_indices, errors):
        for idx, error in zip(batch_indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

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
