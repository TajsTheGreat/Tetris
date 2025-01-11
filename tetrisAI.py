from agent import Agent
from selfmadetetrisAI import Board
import pygame
from time import sleep

from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row
output_file("runtime_data.html")  # Save plot as an HTML file
p = figure(title="Example Plot")

env = Board()
env.reset()
exit_program = False
env.render()
game_counter = 0
dic_20 = {}
dic_100 = {}
avg_moves = 0
num_pos_games = 0

avg_moves_y = []
avg_num_pos_games_y = []
avg_x = []

pause = False
name = input("Enter the name of the model you want to load: ")
lr = 0.001
gamma = 0.999
epsilon = 1
input_dim = 17
output_dim = 40
samplesize = 150

epsilon_decay = 5e-5
epsilon_min = 0.01
batchMaxLength = 10_000

# needs to use _ instead of : in the name
name = f"name_{name}, lr_{lr}, gamma_{gamma}, epsilon_{epsilon}, input_dim_{input_dim}, output_dim_{output_dim}, samplesize_{samplesize}, epsilon_decay_{epsilon_decay}, epsilon_min_{epsilon_min}, batchMaxLength_{batchMaxLength}"

theBrain = Agent(name, gamma, epsilon, lr, [input_dim], output_dim, samplesize, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, batchMaxLength=batchMaxLength)

while not exit_program:

    done = False
    moves = 0

    while not done:

        
        # Observes the current state of the environment
        obs = env.get_state()
        
        # Chooses an action based on the current state
        action = theBrain.act(obs)

        env.render()
        
        # Takes the action and observes the new state, reward, and whether the game is done
        obs_, reward, done = env.step(action)

        # Increments the number of moves
        moves += 1

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

    # saves the moves made in the game
    avg_moves += moves

    if env.game.score > 0:
        num_pos_games += 1
    
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
        print(f"100 last games:{dict(sorted(dic_100.items()))}, game number: {game_counter}, batch index: {theBrain.index}, epsilon: {theBrain.epsilon}, avg moves: {avg_moves/100}")
        avg_moves_y.append(avg_moves/100)
        avg_num_pos_games_y.append(num_pos_games)
        avg_x.append(game_counter / 100)
        
        # makes a plot of the average number of moves and the number of positive games
        avg_moves_fig = figure(title="Average number of moves", x_axis_label="Game number", y_axis_label="Average number of moves")
        avg_moves_fig.line(avg_x, avg_moves_y)
        
        avg_num_pos_games_fig = figure(title="Number of positive games", x_axis_label="Game number", y_axis_label="Number of positive games")
        avg_num_pos_games_fig.line(avg_x, avg_num_pos_games_y)
        
        # uncomment the line below to show the plot, but it will open a new tab in your browser for every 100 games
        # show(row(avg_moves_fig, avg_num_pos_games_fig))


        avg_moves = 0
        num_pos_games = 0
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