import random
import numpy as np

from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D
from collections import deque
import keras

NBACTIONS = 3  # Number of possible actions (e.g., up, down, stay)
IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4  # Number of past frames (to represent a state)

# Exploration and Learning Parameters
# OBSERVEPERIOD value DRAMATICALLY increased:
# Allows the agent to collect MUCH MORE random experience before training begins.
# This step is critical as negative scores persist.
OBSERVEPERIOD = 50000
GAMMA = 0.975        # Discount factor (determines how important future rewards are)
BATCH_SIZE = 64      # Size of the experience sample used for training

# Increased capacity of the experience replay memory
ExpReplay_CAPACITY = 100000
# Learning rate reduced: For more stable learning!
LEARNING_RATE = 0.00005

# Initial and final values for epsilon decay
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
# EPSILON_DECAY_STEPS value DRAMATICALLY increased:
# Ensures epsilon decreases MUCH SLOWER, the agent continues to explore for a long time.
EPSILON_DECAY_STEPS = 500000

# Target network update frequency reduced: Learning stability can be increased with more frequent updates
TARGET_UPDATE_FREQ = 500

class Agent:

    def __init__(self):
        self.model = self.createModel()  # Main neural network (online network)
        # Target neural network: A copy of the main network, its weights are updated slower.
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # Experience replay memory
        self.ExpReplay = deque(maxlen=ExpReplay_CAPACITY)
        self.steps = 0  # Total number of steps
        self.epsilon = INITIAL_EPSILON  # Exploration rate (100% exploration initially)

    def createModel(self):
        """Creates the Deep Convolutional Neural Network (DCNN) model."""
        model = Sequential()

        # Convolutional layers (for image processing)
        model.add(Conv2D(32, kernel_size=4, strides=(2,2), input_shape=(IMGHEIGHT,IMGWIDTH,IMGHISTORY), padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(64, kernel_size=4, strides=(2,2), padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(64, kernel_size=3, strides=(1,1), padding="same"))
        model.add(Activation("relu"))

        model.add(Flatten())  # Flatten the output of convolutional layers

        # Fully connected layers (to estimate Q-values)
        model.add(Dense(512))
        model.add(Activation("relu"))

        # Output layer (one Q-value for each action)
        model.add(Dense(units=NBACTIONS, activation="linear")) # Q-values should be linear

        # Compile the model - Learning rate explicitly specified for Adam optimizer
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(loss="mse", optimizer=optimizer)

        return model

    def FindBestAct(self, s):
        """Finds the best action according to the epsilon-greedy strategy."""
        # If in exploration phase or randomness rate is less than epsilon
        if random.random() < self.epsilon or self.steps < OBSERVEPERIOD:
            return random.randint(0, NBACTIONS - 1)  # Choose a random action
        else:
            # Exploitation phase: Choose the best action based on the model's prediction
            qvalue = self.model.predict(s, verbose=0)[0]
            bestA = np.argmax(qvalue)  # Choose the action with the highest Q-value
            return bestA

    def CaptureSample(self, sample):
        """Adds a sample (transition) to the experience replay memory."""
        self.ExpReplay.append(sample) # Automatically removes the oldest element due to maxlen

        self.steps += 1  # Increment the step counter

        # Epsilon decay strategy (starts after OBSERVEPERIOD)
        if self.steps > OBSERVEPERIOD:
            self.epsilon = FINAL_EPSILON + (INITIAL_EPSILON - FINAL_EPSILON) * \
                           np.exp(-(self.steps - OBSERVEPERIOD) / EPSILON_DECAY_STEPS)
            self.epsilon = max(self.epsilon, FINAL_EPSILON) # Ensure epsilon never falls below the minimum value

        # Copy target network weights from the main network at a specific frequency
        if self.steps % TARGET_UPDATE_FREQ == 0 and self.steps > OBSERVEPERIOD:
            self.target_model.set_weights(self.model.get_weights())


    def Process(self):
        """Trains the model by sampling from the experience replay memory."""
        # Start training after enough experience is collected AND ensure there are enough samples for a minibatch
        if self.steps > OBSERVEPERIOD and len(self.ExpReplay) >= BATCH_SIZE:
            minibatch = random.sample(self.ExpReplay, BATCH_SIZE)

            inputs = np.zeros((BATCH_SIZE, IMGHEIGHT, IMGWIDTH, IMGHISTORY))
            targets = np.zeros((BATCH_SIZE, NBACTIONS))

            for i in range(BATCH_SIZE):
                state_t, action_t, reward_t, state_t1 = minibatch[i]

                inputs[i] = state_t[0]

                current_q_values = self.model.predict(state_t, verbose=0)[0]
                targets[i] = current_q_values

                if state_t1 is None: # If there is no next state (terminal state)
                    targets[i, action_t] = reward_t
                else:
                    # target_model used for next state Q-values!
                    next_q_values = self.target_model.predict(state_t1, verbose=0)[0]
                    targets[i, action_t] = reward_t + GAMMA * np.max(next_q_values)

            self.model.fit(inputs, targets, batch_size=BATCH_SIZE, epochs=1, verbose=0)
