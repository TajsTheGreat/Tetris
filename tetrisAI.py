from agent import Agent
from selfmadetetrisAI import Board
import pygame
from time import sleep

env = Board()
env.reset()
exit_program = False
env.render()

active = True
counter = 0

theBrain = Agent("first", 0.99, 1, 0.001, [17], 40, 150)

while not exit_program:

    done = False

    while not done:

        # controls
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    exit_program = True
                    done = True
        
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

    print(f"Score: {env.game.score}, Batchsize: {theBrain.index}")
    counter += 1
    if counter % 50 == 0:
        sleep(0.5)
    env.reset()

env.close()