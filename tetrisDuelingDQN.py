from DuelingDQN import Agent
from selfmadetetrisAI import Board
import pygame
from time import sleep

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row
import pandas as pd

env = Board()
env.reset()
exit_program = False
env.render()
game_counter = 0
batch_counter = 0
dic_20 = {}
dic_100 = {}
avg_moves = 0
num_pos_games = 0
loss_value = 0
reward_value = 0
score_value = 0
q_value_value = 0

avg_moves_y = []
avg_num_pos_games_y = []
avg_rewards_y = []
avg_score_y = []
avg_losses_y = []
avg_q_values_y = []
avg_losses_x = []
avg_x = []

experiment_avg_moves = 0
experiment_num_pos_games = 0
experiment_reward_value = 0

experiment_avg_moves_y = []
experiment_avg_num_pos_games_y = []
experiment_avg_rewards_y = []
experiment_avg_x = []

pause = False
name_input = input("Enter the name of the model: ")
lr = 0.001
gamma = 0.97
epsilon = 1
input_dim = 18
output_dim = 40
samplesize = 500

epsilon_min = 0.025
epsilon_decay_factor = 1/(200_000)
batchMaxLength = 100_000

height_reward_low = 0
bumpiness_reward = 0
hole_reward = 0
score_reward = 0
move_reward = 0
move_100_counter = 0

# needs to use _ instead of : in the name
name = f"name_{name_input}, lr_{lr}, gamma_{gamma}, epsilon_{epsilon}, input_dim_{input_dim}, output_dim_{output_dim}, samplesize_{samplesize}, epsilon_min_{epsilon_min}, batchMaxLength_{batchMaxLength}"

theBrain = Agent(name, gamma, epsilon, lr, [input_dim], output_dim, samplesize, epsilon_decay_factor=epsilon_decay_factor, epsilon_min=epsilon_min, batchMaxLength=batchMaxLength)

output_file(f"Models/{name_input}_runtime_data.html")  # Save plot as an HTML file

while not exit_program:

    done = False
    moves = 0
    experiment_avg_moves += 1
    experiment_num_pos_games_subtract = 0

    while not done:
        
        # Observes the current state of the environment
        obs = env.get_state()
        
        # Chooses an action based on the current state
        action = theBrain.act(obs)

        env.render()
        
        # Takes the action and observes the new state, reward, and whether the game is done
        obs_, reward, done = env.step(action[0])

        # Increments the number of moves
        moves += 1
        batch_counter += 1

        # if done:
        #     reward = reward - (1000/(moves/20)) 

        reward = reward + moves
        
        height_reward_low += env.height_low_reward
        bumpiness_reward += env.bumpiness
        hole_reward += env.hole_opening_reward
        score_reward += env.test_score
        move_reward += moves
        move_100_counter += 1

        # Updates the batch with the new experience
        theBrain.updateBatch(obs, action[0], reward, obs_, done)


        # Learns from the batch of experiences and updates the model
        result = theBrain.experience()
        if result is not None:
            loss_value += result
        
        reward_value += reward
        experiment_reward_value += reward

        if action[1] is not False:
            q_value_value += action[1]

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

        # updates the epsilon value
        theBrain.updateEpsilon()

        
        if batch_counter % 2500 == 0:
            if env.game.score > 0:
                experiment_num_pos_games += moves
            experiment_avg_moves_y.append(2500/experiment_avg_moves)
            experiment_avg_num_pos_games_y.append(experiment_num_pos_games/2500)
            experiment_avg_rewards_y.append(experiment_reward_value/2500)
            experiment_avg_x.append(batch_counter / 2500)

            experiment_avg_moves = 0
            experiment_num_pos_games = 0
            experiment_reward_value = 0
            experiment_num_pos_games_subtract = moves

            pd.DataFrame({"avg_moves": experiment_avg_moves_y, "num_pos_games": experiment_avg_num_pos_games_y, "reward": experiment_avg_rewards_y, "x": experiment_avg_x}).to_csv(f"Models/{name_input}_experiment_data.csv")

    # saves the moves made in the game
    avg_moves += moves
    score_value += env.game.score

    if env.game.score > 0:
        num_pos_games += 1
        experiment_num_pos_games += moves - experiment_num_pos_games_subtract
    
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
        print(f"height reward low: {height_reward_low/move_100_counter}, bumpiness reward: {bumpiness_reward/move_100_counter}, hole reward: {hole_reward/move_100_counter}, score reward: {score_reward/move_100_counter}, move reward: {move_reward/move_100_counter}")
        height_reward_low = 0
        bumpiness_reward = 0
        hole_reward = 0
        score_reward = 0
        move_reward = 0
        move_100_counter = 0

        avg_moves_y.append(avg_moves/100)
        avg_num_pos_games_y.append(num_pos_games)
        avg_x.append(game_counter / 100)
        avg_rewards_y.append(reward_value/100)
        avg_score_y.append(score_value/100)
        avg_q_values_y.append(q_value_value/100)
        if game_counter >= 500:
            avg_losses_y.append(loss_value/100)
            avg_losses_x.append(game_counter / 100)
        
        # makes a plot of the average number of moves and the number of positive games
        avg_moves_fig = figure(title="Average number of moves", x_axis_label="Game number", y_axis_label="Average number of moves")
        avg_moves_fig.line(avg_x, avg_moves_y)
        
        avg_num_pos_games_fig = figure(title="Number of positive games", x_axis_label="Game number", y_axis_label="Number of positive games")
        avg_num_pos_games_fig.line(avg_x, avg_num_pos_games_y)

        avg_rewards_fig = figure(title="Average reward", x_axis_label="Game number", y_axis_label="Average reward")
        avg_rewards_fig.line(avg_x, avg_rewards_y)

        avg_score_y_fig = figure(title="Average score", x_axis_label="Game number", y_axis_label="Average score")
        avg_score_y_fig.line(avg_x, avg_score_y)

        avg_q_values_y_fig = figure(title="Average Q-value", x_axis_label="Game number", y_axis_label="Average Q-value")
        avg_q_values_y_fig.line(avg_x, avg_q_values_y)

        if game_counter >= 500:
            avg_losses_fig = figure(title="Average loss", x_axis_label="Game number", y_axis_label="Average loss")
            avg_losses_fig.line(avg_losses_x, avg_losses_y)
            
            # uncomment the line below to show the plot, but it will open a new tab in your browser for every 100 games
            # show(row(avg_moves_fig, avg_num_pos_games_fig))
            save(row(avg_moves_fig, avg_num_pos_games_fig, avg_rewards_fig, avg_score_y_fig, avg_q_values_y_fig, avg_losses_fig))
        else:
            save(row(avg_moves_fig, avg_num_pos_games_fig, avg_rewards_fig, avg_score_y_fig, avg_q_values_y_fig))


        avg_moves = 0
        num_pos_games = 0
        loss_value = 0
        reward_value = 0
        score_value = 0
        q_value_value = 0
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