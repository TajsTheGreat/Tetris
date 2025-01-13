# File is used to tune the reward function of the tetris AI 
from agent import Agent
from selfmadetetrisAI import Board
import pygame

env = Board()
env.reset()
exit_program = False
env.render()

name = "reward_tuning"
lr = 0.001
gamma = 0.999
epsilon = 1
input_dim = 17
output_dim = 40
samplesize = 150

epsilon_decay = 1e-6
epsilon_min = 0.01
batchMaxLength = 10_000
next = True

theBrain = Agent(name, gamma, epsilon, lr, [input_dim], output_dim, samplesize, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, batchMaxLength=batchMaxLength)

while not exit_program:

    done = False

    while not done:

        if next:
            # Observes the current state of the environment
            obs = env.get_state()
            
            # Chooses an action based on the current state
            action = theBrain.act(obs)

            
            # Takes the action and observes the new state, reward, and whether the game is done
            obs_, reward, done = env.step(action)

            # Updates the batch with the new experience
            theBrain.updateBatch(obs, action, reward, obs_, done)

            # Learns from the batch of experiences and updates the model
            theBrain.experience()

            env.render()
            print(f"Reward: {reward}, height reward: {env.height_reward}, height low reward:{env.height_low_reward}")
            next = False
        
        # controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    exit_program = True
                    done = True
                if event.key == pygame.K_r:
                    env.rendering = not env.rendering
                if event.key == pygame.K_SPACE:
                    next = True

    env.reset()

env.close()