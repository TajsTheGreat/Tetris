from agent import Agent
from selfmadetetrisAI import Board
import pygame
from time import sleep

env = Board()
env.reset()
exit_program = False
env.render()
game_counter = 0
dic_20 = {}
dic_100 = {}

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
                if event.key == pygame.K_r:
                    env.rendering = not env.rendering

    
    game_counter += 1
    if env.game.score not in dic_20:
        dic_20[env.game.score] = 1
    else:
        dic_20[env.game.score] += 1

    if env.game.score not in dic_100:
        dic_100[env.game.score] = 1
    else:
        dic_100[env.game.score] += 1

    if game_counter % 20 == 0:
        print(f"20 last games:{dict(sorted(dic_20.items()))}")
        dic_20 = {}

    if game_counter % 100 == 0:
        print(f"100 last games:{dict(sorted(dic_100.items()))}, game number: {game_counter}, batch index: {theBrain.index}, epsilon: {theBrain.epsilon}")
        dic_100 = {}
    
    if game_counter % 500 == 0:
        for i in range(100):
            env.reset()
            done = False
            while not done:
                env.render()
                obs = env.get_state()
                action = theBrain.evaluate(obs)
                obs_, reward, done = env.step(action)
            if env.game.score not in dic_100:
                dic_100[env.game.score] = 1
            else:
                dic_100[env.game.score] += 1
        print(f"100 evaluated games:{dict(sorted(dic_100.items()))}")
        dic_100 = {}


    env.reset()

theBrain.saveWeights()

env.close()