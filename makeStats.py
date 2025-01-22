from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row
import pandas as pd

with open("Models/PeterDQNFINALDATA_experiment_data.csv") as f:
    DQN = pd.read_csv(f)

with open("Models/PeterDDQNFINALDATA_experiment_data.csv") as f:
    DDQN = pd.read_csv(f)

with open("Models/PeterDUELINGDQNFINALDATA2_experiment_data.csv") as f:
    DUELINGDQN = pd.read_csv(f)

with open("Models/DDDQNtest3_experiment_data.csv") as f:
    DDDQN = pd.read_csv(f)

output_file("Models/Stats.html")

p = figure(title="Average number of positive games for every 2500 moves", x_axis_label='Datapoint for every 2500 moves', y_axis_label='Precentage of positive games', height=400)
p.line(DQN['x'], DQN['num_pos_games'], line_width=1.5, legend_label="DQN", color="blue")
p.line(DDQN['x'], DDQN['num_pos_games'], line_width=1.5, legend_label="DDQN", color="red")
p.line(DUELINGDQN['x'], DUELINGDQN['num_pos_games'], line_width=1.5, legend_label="Dueling DQN", color="green")
p.line(DDDQN['x'], DDDQN['num_pos_games'], line_width=1.5, legend_label="Dueling DDQN", color="orange")
p.legend.location = "bottom_right"

p2 = figure(title="Average number of moves per game for every 2500 games", x_axis_label='Datapoint for every 2500 moves', y_axis_label='Number of moves on average', height=400)
p2.line(DQN['x'], DQN['avg_moves'], line_width=1.5, legend_label="DQN", color="blue")
p2.line(DDQN['x'], DDQN['avg_moves'], line_width=1.5, legend_label="DDQN", color="red")
p2.line(DUELINGDQN['x'], DUELINGDQN['avg_moves'], line_width=1.5, legend_label="Dueling DQN", color="green")
p2.line(DDDQN['x'], DDDQN['avg_moves'], line_width=1.5, legend_label="Dueling DDQN", color="orange")
p2.legend.location = "top_left"

p3 = figure(title="Average reward per move for every 2500 moves", x_axis_label='Datapoint for every 2500 moves', y_axis_label='Average reward per move', height=400)
p3.line(DQN['x'], DQN['reward'], line_width=1.5, legend_label="DQN", color="blue")
p3.line(DDQN['x'], DDQN['reward'], line_width=1.5, legend_label="DDQN", color="red")
p3.line(DUELINGDQN['x'], DUELINGDQN['reward'], line_width=1.5, legend_label="Dueling DQN", color="green")
p3.line(DDDQN['x'], DDDQN['reward'], line_width=1.5, legend_label="Dueling DDQN", color="orange")
p3.legend.location = "top_left"

save(row(p, p2, p3))
