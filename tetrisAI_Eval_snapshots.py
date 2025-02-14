# This import statement must be changed to the agent you want to evaluate
from DuelingDDQN import Agent

from selfmadetetrisAI3 import Board
import pygame
import glob

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row
import pandas as pd

env = Board()
env.reset()
exit_program = False
env.render()
game_counter = 0
dic_100 = {}
avg_moves = 0
num_pos_games = 0

avg_moves_y = []
avg_num_pos_games_y = []
reward_y = []
avg_x = []
score = []

avg_moves_y2 = []
avg_num_pos_games_y2 = []

# with open("Models/PeterDQNFINALDATA_experiment_data.csv") as f:
#     DQN = pd.read_csv(f)

# with open("Models/PeterDDQNFINALDATA_experiment_data.csv") as f:
#     DDQN = pd.read_csv(f)

# with open("Models/PeterDUELINGDQNFINALDATA2_experiment_data.csv") as f:
#     DUELINGDQN = pd.read_csv(f)

# with open("Models/DDDQNtest3_experiment_data.csv") as f:
#     DDDQN = pd.read_csv(f)

name_input = input("Enter the name of the model you want to load: ")

# finds all the models with the name inputted
files = glob.glob(f'Models/*{name_input}*.pt')
for file in files:
    print(file)

number = int(input("Enter the number of the model you want to load: "))

# gets the parameters from the name of the model
name = files[number - 1]
name = name[7:-3]

values = name.split(", ")

lr = float(values[1].split("_")[1])
gamma = float(values[2].split("_")[1])
epsilon = float(values[3].split("_")[1])
input_dim = int(values[4].split("_")[2])
output_dim = int(values[5].split("_")[2])
samplesize = int(values[6].split("_")[1])

# the parameters does not really matter since the model is already trained
theBrain = Agent(name, gamma, epsilon, lr, [input_dim], output_dim, samplesize)

# to avoid errors make sure that the layers and neurons are the same as in the model
theBrain.loadWeights()

output_file(f"Models/{name_input}_runtime_eval_data.html")  # Save plot as an HTML file

snapshot = False
render = False
pause = False

while not exit_program:

    done = False
    moves = 0
    total_reward = 0

    while not done:

        # Observes the current state of the environment
        obs = env.get_state()
        
        # Chooses an action based on the current state
        action = theBrain.evaluate(obs)

        # Takes the action and observes the new state, reward, and whether the game is done
        obs_, reward, done = env.step(action)

        total_reward += reward

        if render:
            env.render()
        
        if pause:
            pygame.time.wait(100)

        # Increments the number of moves
        moves += 1
        
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
                    snapshot = True
                if event.key == pygame.K_t:
                    render = not render

    # saves the moves made in the game
    avg_moves += moves

    if env.game.score > 0:
        num_pos_games += 1
        avg_num_pos_games_y2.append(1)
    else:
        avg_num_pos_games_y2.append(0)
    
    game_counter += 1

    if env.game.score not in dic_100:
        dic_100[env.game.score] = 1
    else:
        dic_100[env.game.score] += 1

    if game_counter % 100 == 0:
        print(f"100 last games:{dict(sorted(dic_100.items()))}, game number: {game_counter}, avg moves: {avg_moves/100}")
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
        save(row(avg_moves_fig, avg_num_pos_games_fig))

        avg_moves = 0
        num_pos_games = 0
        dic_100 = {}

    if snapshot:
        env.render(snapshot)
        snapshot = False
    
    if game_counter > 10_000:
        exit_program = True
    
    score.append(env.game.score)
    avg_moves_y2.append(moves)
    reward_y.append(total_reward)

    env.reset()

env.close()

pd.DataFrame({"score": score, "avg_moves": avg_moves_y2, "num_positivs": avg_num_pos_games_y2, "reward": reward_y}).to_csv(f"Models/{name}_score_data.csv")