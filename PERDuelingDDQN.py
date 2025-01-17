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
        self.A = nn.Linear(256, output_dim)
        self.V = nn.Linear(256, 1)

        # Uses Adam for optimization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.MSELoss() # if not working add reduction='sum'
        self.to(self.device)
    
    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        flat3 = F.relu(self.fc3(flat2))
        flat4 = F.relu(self.fc4(flat3))

        # The value of the state
        V = self.V(flat4)

        # The advantage of each action
        A = self.A(flat4)

        return V + A - torch.mean(A, dim=1, keepdim=True)

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

        # initializes the policy network and the target network
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
        self.priorities = np.zeros((self.batchMaxLength,), dtype=np.float32)

        # target model update interval
        self.target_update_interval = 1000

        # the number of steps for each update
        self.step_update = 10
    
    # selects an action from the current state
    def act(self, obs):    

        # decides wether to prioritize exploration or exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 39), False
        else:
            state = torch.tensor([obs], dtype=torch.float32).to(self.model.device)
            # does not calculate gradients for the action, which is not needed
            with torch.no_grad(): 
                actions = self.model.forward(state)
            chosen_action = torch.argmax(actions).item()
            # Extracts q-values from the model for plotting
            chosen_q_value = actions[0, chosen_action].item() 
            return chosen_action, chosen_q_value
    
    # stores the experience in the batch
    def updateBatch(self, state, action, reward, new_state, done):
        i = self.index % self.batchMaxLength
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        new_state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)

        # calculates the error (uses torch.no_grad() because speedy)
        with torch.no_grad():
            target = reward + self.gamma * torch.max(self.model(new_state_tensor)) * (1 - done)
            current = self.model(state_tensor)[0, action]
            error = abs(target - current).item()

        # Add to memory
        self.state_mem[i] = state
        self.new_state_mem[i] = new_state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.terminal_mem[i] = done
        self.priorities[i] = (error + 1e-5) ** self.alpha

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
        terminal = self.terminal_mem[indices]

        # Saves the data in a way pytorch can understand
        states = torch.tensor(states, dtype=torch.float32).to(self.model.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.model.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.model.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.model.device)
        terminal = torch.tensor(terminal, dtype=torch.bool).to(self.model.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.model.device)

        # Compute Q-values for the selected actions
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # For the DDQN update
        next_actions = self.model(next_states).argmax(1)
        q_values_next = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        q_values_next[terminal] = 0.0

        # Compute the target Q-values
        q_target = rewards + self.gamma * q_values_next

        # Calculate loss
        errors = q_target - q_values
        loss = (weights * errors.pow(2)).mean()

        # Update priorities and beta
        self.update_priorities(indices, (abs(errors) + 1e-5).detach().cpu().numpy())
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Backpropagate loss
        loss.backward()
        self.model.optimizer.step()

        # Periodically update the target network
        if self.index % self.target_update_interval == 0:
            self.sync_target_network()

        return loss.item()
    
    def updateEpsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay_factor if self.epsilon > self.epsilon_min else self.epsilon_min
    
    def evaluate(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.model.device)
        # does not calculate gradients for the action, which is not needed
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
