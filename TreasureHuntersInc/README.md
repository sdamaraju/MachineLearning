## Project - 4 : Treasure Hunters Inc.

A game where the player in a map has to reach the treasure overcoming the monsters and blockers in between the path.
Goal : Get the optimal path from source to destination.

## Input data and Output Information
Input is a map that consists of a Player, Monster, Blocker and Treasure in a 5 X 5 grid.

Output expected is an optimal path that helps the player to reach the treasure by avoiding the
monsters and the blockers.

## Policies used:
Base Implementation - Notebook : 738-HW4-200epochs and 738-HW-4

Greedy Epsilon - Notebook : 738-HW-4-RewardsPolicy-2

## Rules that apply

1. For each position we have 4 possible moves, 
UP, DOWN, LEFT, RIGHT, so each position in game board, 
player will have 4 possible ways to move.

2. There are 25 possible position on the game board.

## Rewards Policy
Rewards Policy

Initializing the 25 x 5 rewards table with possible transitions from transitions table and the game board.
The Policy : 
There are 25 possible position on the game board, 
we are going to define a rewards for each possible state in the grid.
Defining the rewards policy..
Whenever the player encounters a Monster, the reward is -10
Whenever the player encounters a Blocker, the reward is -5
Whenever the player encounters the Treasure, the reward is +10
Whenever the Player encounters a * , the reward is +1 so that the Player can move. 
Invalid moves from transition tables get a 0 reward.

## Steps followed in the project

1. Initialize transition,action and rewards tables
2. Build and Run the Reinforcement learning model on the map.
3. Build the QL table
4. Identify the optimal path from QLearning table execution.

# Observations

I observed that when we increase the epochs, better learning models are achieved.
The 200 epochs map is bit more complex than the normal one and still the model could achieve the
optimal path. The same map in 10 epochs run couldn't work efficiently.

Sources Used :  https://towardsdatascience.com/practical-reinforcement-learning-02-getting-started-with-q-learning-582f63e4acd9
