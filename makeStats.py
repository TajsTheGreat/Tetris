from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row
import pandas as pd

with open("Models/PeterDQNFINALDATA_experiment_data.csv") as f:
    DQN = pd.read_csv(f)

with open("Models/PeterDDQNFINALDATA_experiment_data.csv") as f:
    DDQN = pd.read_csv(f)

with open("Models/PeterDUELINGDQNFINALDATA_experiment_data.csv") as f:
    DUELINGDQN = pd.read_csv(f)

with open("Models/DDDQNtest2_experiment_data.csv") as f:
    DDDQN = pd.read_csv(f)

output_file("Tetris/Models/Stats.html")

print(DQN.columns)

p = figure(title="Tetris Dueling DDQN vs DDQN vs DQN vs Dueling DQN", x_axis_label='per 2500', y_axis_label='Average number of positive games')
p.line(DQN['x'], DQN['num_pos_games'], line_width=1.5, legend_label="DQN", color="blue")
p.line(DDQN['x'], DDQN['num_pos_games'], line_width=1.5, legend_label="DDQN", color="red")
p.line(DUELINGDQN['x'], DUELINGDQN['num_pos_games'], line_width=1.5, legend_label="Dueling DQN", color="green")
p.line(DDDQN['x'], DDDQN['num_pos_games'], line_width=1.5, legend_label="Dueling DDQN", color="orange")

p2 = figure(title="Tetris Dueling DDQN vs DDQN vs DQN vs Dueling DQN", x_axis_label='per 2500', y_axis_label='Average Moves')
p2.line(DQN['x'], DQN['avg_moves'], line_width=1.5, legend_label="DQN", color="blue")
p2.line(DDQN['x'], DDQN['avg_moves'], line_width=1.5, legend_label="DDQN", color="red")
p2.line(DUELINGDQN['x'], DUELINGDQN['avg_moves'], line_width=1.5, legend_label="Dueling DQN", color="green")
p2.line(DDDQN['x'], DDDQN['avg_moves'], line_width=1.5, legend_label="Dueling DDQN", color="orange")

p3 = figure(title="Tetris Dueling DDQN vs DDQN vs DQN vs Dueling DQN", x_axis_label='per 2500', y_axis_label='Average Reward')
p3.line(DQN['x'], DQN['reward'], line_width=1.5, legend_label="DQN", color="blue")
p3.line(DDQN['x'], DDQN['reward'], line_width=1.5, legend_label="DDQN", color="red")
p3.line(DUELINGDQN['x'], DUELINGDQN['reward'], line_width=1.5, legend_label="Dueling DQN", color="green")
p3.line(DDDQN['x'], DDDQN['reward'], line_width=1.5, legend_label="Dueling DDQN", color="orange")

save(row(p, p2, p3))