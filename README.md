🕹️ Deep Q-Learning (DQL) Enhanced Pong Game AI
---
This project showcases a Deep Q-Learning (DQL) agent learning to play the classic Pong game.</br>
The agent utilizes deep neural networks to process game visuals and make real-time decisions, aiming to master the game through reinforcement learning.

📘 Project Description
---
This repository contains the code for a Pong game environment and a Deep Q-Learning (DQL) agent trained to play it.</br> 
The agent learns by interacting with the environment—observing screen states, selecting actions (such as moving the paddle), and receiving rewards or penalties based on its performance.</br>
Over time, the agent improves its strategy through reinforcement learning, aiming to maximize its score in this classic arcade setting.

📁 Project Structure
---
The project is organized into three main Python files:</br>

DCQL_Pong.py – Game Environment</br>

This module defines the Pong game using Pygame. It includes:

Implementation of core mechanics: paddle and ball movement, collision detection, and scoring.</br>
A tailored reward system:</br>
✅ Large reward for hitting the ball.</br>
❌ Penalty for missing the ball.</br>
🟡 Small reward if the opponent misses.</br>
The environment is built to provide meaningful feedback to the agent, encouraging desired behavior during learning.

DQCL_Pong_Agent.py – DQL Agent</br>
This file contains the core logic for the Deep Q-Learning agent, including:</br>
A Convolutional Neural Network (CNN) for processing visual input from game frames.</br>
Separate online and target networks to stabilize Q-value predictions.</br>
An experience replay buffer to store gameplay experiences and sample them randomly for training, reducing temporal correlation between updates.</br>
An epsilon-greedy strategy to balance:</br>
Exploration: Trying new actions.</br>
Exploitation: Leveraging known effective actions.</br>
Carefully tuned hyperparameters:</br>
Learning rate</br>
Discount factor (gamma)</br>
Epsilon decay schedule</br>
Target network update frequency</br>
DCQL_TrainAgent.py – Training Script</br>
This is the main script that coordinates the training process:</br>

Initializes the game environment and the DQL agent.</br>
Processes raw game frames:</br>
Grayscaling</br>
Cropping</br>
Resizing</br>
Normalization</br>
Runs the main training loop:</br>
The agent plays the game, collects experience, and updates its model.</br>
Displays progress updates and game scores in the console, allowing you to monitor learning performance in real time.</br>
If you'd like, I can also add a visual training progress chart, demo video/GIF support, or instructions for running the project in Jupyter Notebook or Colab.

🌟 Features
---
🧠 Deep Q-Learning (DQL) Algorithm: Learns optimal strategies from raw pixel inputs using CNNs</br>
🔁 Experience Replay: Improves stability by breaking temporal correlations</br>
🎯 Target Network: Stabilizes Q-value targets, critical for deep RL</br>
⚖️ Epsilon-Greedy Exploration: Balances trying new moves with exploiting known strategies</br>
🔧 Customizable Parameters: Easy hyperparameter tuning in DQCL_Pong_Agent.py</br>
🕹️ Pygame-based Pong Environment: Lightweight yet functional for training</br>
🖼️ Robust Image Preprocessing: Converts frames into a suitable format for neural network input
