from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row
import pandas as pd
import numpy as np
import scipy.stats as stats

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

with open("Models/name_PeterDDQNFINALDATA, lr_0.0005, gamma_0.97, epsilon_1, input_dim_18, output_dim_40, samplesize_500, epsilon_min_0.01, batchMaxLength_100000_score_data.csv") as f:
    statlad = pd.read_csv(f)
print((sum(statlad.score))/len(statlad.score))
print((np.var(statlad.score)))
print((np.std(statlad.score)))
confidence_level = 0.95
degrees_freedom = len(statlad.score) - 1
sample_mean = np.mean(statlad.score)
sample_standard_error = stats.sem(statlad.score)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
print("Confidence interval for statlad.score:", confidence_interval)
# print((sum(statlad.avg_moves))/len(statlad.avg_moves))
# print((sum(statlad.num_positivs))/len(statlad.num_positivs))

with open("Models/name_PeterDQNFINALDATA, lr_0.0005, gamma_0.97, epsilon_1, input_dim_18, output_dim_40, samplesize_500, epsilon_min_0.01, batchMaxLength_100000_score_data.csv") as f:
    statlad2 = pd.read_csv(f)

print((sum(statlad2.score))/len(statlad2.score))
print((np.var(statlad2.score)))
print((np.std(statlad2.score)))
confidence_level = 0.95
degrees_freedom = len(statlad2.score) - 1
sample_mean = np.mean(statlad2.score)
sample_standard_error = stats.sem(statlad2.score)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
print("Confidence interval for statlad.score:", confidence_interval)
# print((sum(statlad2.avg_moves))/len(statlad2.avg_moves))
# print((sum(statlad2.num_positivs))/len(statlad2.num_positivs))

with open("Models/name_PeterDUELINGDQNFINALDATA2, lr_0.0005, gamma_0.97, epsilon_1, input_dim_18, output_dim_40, samplesize_500, epsilon_min_0.01, batchMaxLength_100000_score_data.csv") as f:
    statlad3 = pd.read_csv(f)

print((sum(statlad3.score))/len(statlad3.score))
print((np.var(statlad3.score)))
print((np.std(statlad3.score)))
confidence_level = 0.95
degrees_freedom = len(statlad3.score) - 1
sample_mean = np.mean(statlad3.score)
sample_standard_error = stats.sem(statlad3.score)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
print("Confidence interval for statlad.score:", confidence_interval)
# print((sum(statlad3.avg_moves))/len(statlad3.avg_moves))
# print((sum(statlad3.num_positivs))/len(statlad3.num_positivs))

with open("Models/name_DDDQNtest3, lr_0.0005, gamma_0.97, epsilon_1, input_dim_18, output_dim_40, samplesize_500, epsilon_min_0.01, batchMaxLength_100000_score_data.csv") as f:
    statlad4 = pd.read_csv(f)

print((sum(statlad4.score))/len(statlad4.score))
print((np.var(statlad4.score)))
print((np.std(statlad4.score)))
confidence_level = 0.95
degrees_freedom = len(statlad4.score) - 1
sample_mean = np.mean(statlad4.score)
sample_standard_error = stats.sem(statlad4.score)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
print("Confidence interval for statlad.score:", confidence_interval)
# print((sum(statlad4.avg_moves))/len(statlad4.avg_moves))
# print((sum(statlad4.num_positivs))/len(statlad4.num_positivs))
