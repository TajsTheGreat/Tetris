To run an AI you have to run the files called tetris and then AI or DQN or something else.
These files import the agent they have to use and the environment.

All the files with agent in them are different agents that can be imported.
The agents used for the rapport is agent.py(DQN), DoubleAgent.py(DoubleDQN), DuelingDQN.py and DuelingDDQN.py.
The two others are Prioritized Experience Replay(PER) and Maxmin Q-learning(MaxDQN).

There are 4 different environments the first one, selfmadetetris.py, is for playing manually. The three others
are for the AI's, there is 3 because we have experimented with different input dimensions and values. 
selfmadetetrisAI3.py is the final one and the one we evaluated to be most optimal for our chosen DQN's.

tetrisAI_Eval_snapshots.py is a file that tests the final product of the trained DQN's and records there scores,
average moves and number of positive games. This file is used to collect data for our report. 
makeStats.py is used to make graphs of the data of the 4 different DQN's training progress.