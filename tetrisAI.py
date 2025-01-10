from agent import Agent
from selfmadetetrisAI import Board
import pygame
from time import sleep

env = Board()
env.reset()
exit_program = False
env.render()

pause = False
name = input("Enter the name of the model you want to load: ")
lr = 0.0005
gamma = 0.99
epsilon = 1
input_dim = 17
output_dim = 40
samplesize = 150

# needs to use _ instead of : in the name
name = f"name_{name}, lr_{lr}, gamma_{gamma}, epsilon_{epsilon}, input_dim_{input_dim}, output_dim_{output_dim}, samplesize_{samplesize}"

theBrain = Agent(name, gamma, epsilon, lr, [input_dim], output_dim, samplesize)

while not exit_program:

    done = False

    while not done:

        
        # Observes the current state of the environment
        obs = env.get_state()
        
        # Chooses an action based on the current state
        action = theBrain.act(obs)

        env.render()
        
        # Takes the action and observes the new state, reward, and whether the game is done
        obs_, reward, done = env.step(action)

        # Updates the batch with the new experience
        theBrain.updateBatch(obs, action, reward, obs_, done)


        # Learns from the batch of experiences and updates the model
        theBrain.experience()

        if pause:
            sleep(0.5)
        
        # controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    exit_program = True
                    done = True
                if event.key == pygame.K_p:
                    pause = not pause

    print(f"Score: {env.game.score}, Batchsize: {theBrain.index}, Epsilon: {theBrain.epsilon}")
    env.reset()

theBrain.saveWeights()

env.close()