🕹️ Deep Q-Learning (DQL) Enhanced Pong Game AI
---
This project showcases a Deep Q-Learning (DQL) agent learning to play the classic Pong game. 
The agent utilizes deep neural networks to process game visuals and make real-time decisions, aiming to master the game through reinforcement learning.

📘 Project Description
---
This repository contains the code for a Pong game environment and a Deep Q-Learning (DQL) agent trained to play it. 
The agent learns by interacting with the environment—observing screen states, selecting actions (such as moving the paddle), and receiving rewards or penalties based on its performance.

Over time, the agent improves its strategy through reinforcement learning, aiming to maximize its score in this classic arcade setting.

📁 Project Structure
---
The project is organized into three main Python files:

DCQL_Pong.py – Game Environment
This module defines the Pong game using Pygame. It includes:

Implementation of core mechanics: paddle and ball movement, collision detection, and scoring.
A tailored reward system:
✅ Large reward for hitting the ball.
❌ Penalty for missing the ball.
🟡 Small reward if the opponent misses.
The environment is built to provide meaningful feedback to the agent, encouraging desired behavior during learning.

DQCL_Pong_Agent.py – DQL Agent
This file contains the core logic for the Deep Q-Learning agent, including:

A Convolutional Neural Network (CNN) for processing visual input from game frames.
Separate online and target networks to stabilize Q-value predictions.
An experience replay buffer to store gameplay experiences and sample them randomly for training, reducing temporal correlation between updates.
An epsilon-greedy strategy to balance:
Exploration: Trying new actions.
Exploitation: Leveraging known effective actions.
Carefully tuned hyperparameters:
Learning rate
Discount factor (gamma)
Epsilon decay schedule
Target network update frequency
DCQL_TrainAgent.py – Training Script
This is the main script that coordinates the training process:

Initializes the game environment and the DQL agent.
Processes raw game frames:
Grayscaling
Cropping
Resizing
Normalization
Runs the main training loop:
The agent plays the game, collects experience, and updates its model.
Displays progress updates and game scores in the console, allowing you to monitor learning performance in real time.
If you'd like, I can also add a visual training progress chart, demo video/GIF support, or instructions for running the project in Jupyter Notebook or Colab.

🌟 Features
---
🧠 Deep Q-Learning (DQL) Algorithm: Learns optimal strategies from raw pixel inputs using CNNs
🔁 Experience Replay: Improves stability by breaking temporal correlations
🎯 Target Network: Stabilizes Q-value targets, critical for deep RL
⚖️ Epsilon-Greedy Exploration: Balances trying new moves with exploiting known strategies
🔧 Customizable Parameters: Easy hyperparameter tuning in DQCL_Pong_Agent.py
🕹️ Pygame-based Pong Environment: Lightweight yet functional for training
🖼️ Robust Image Preprocessing: Converts frames into a suitable format for neural network input
