import DCQL_Pong
import DQCL_Pong_Agent

import numpy as np
import random
import skimage
import warnings
import matplotlib.pyplot as plt  # Already imported, no need to rewrite

# Importing relevant dimensions from DQCL_Pong_Agent
from DQCL_Pong_Agent import IMGHEIGHT, IMGWIDTH, IMGHISTORY

warnings.filterwarnings("ignore")

TOTAL_TRAIN_TIME = 100000

# Dimensions are imported from DQCL_Pong_Agent, so no need to redefine them here
# However, they can remain here for consistency; the important thing is that they match the values in the Agent file
# IMGHEIGHT = 40
# IMGWIDTH = 40
# IMGHISTORY = 4

def ProcessGameImage(RawImage):
    """Converts the raw game image to grayscale, crops, resizes, and normalizes it."""
    GreyImage = skimage.color.rgb2gray(RawImage)  # Convert to grayscale
    CroppedImage = GreyImage[0:400, 0:400]  # Crop the image (likely to select the game area)
    ReducedImage = skimage.transform.resize(CroppedImage, (IMGHEIGHT, IMGWIDTH))  # Resize to target dimensions

    # NORMALIZATION CORRECTED: Bring pixel values from 0-255 range to 0-1 range
    # Models generally process inputs in the 0-1 range better.
    ReducedImage = ReducedImage / 255.0  # Previously /128, now /255.0

    # Rescale intensity is no longer needed here, as we directly mapped to 0-1 range with /255.0.
    # skimage.exposure.rescale_intensity(ReducedImage, out_range=(0,255))

    return ReducedImage


def TrainExperiment():
    """Executes the Deep Q-Learning training experiment."""
    TrainHistory = []  # List to store scores during training
    TheGame = DCQL_Pong.PongGame()  # Initialize the Pong game environment
    TheGame.InitialDisplay()  # Show the game's initial display
    TheAgent = DQCL_Pong_Agent.Agent()  # Initialize the DQCL agent

    BestAction = 0  # Default value for the first action

    # Play the first game and get the initial state
    [InitialScore, InitialScreenImage] = TheGame.PlayNextMove(BestAction)
    InitialGameImage = ProcessGameImage(InitialScreenImage)  # Process the image

    # Create the game state (past frames)
    # The initial state is created by stacking 4 identical images using np.stack
    GameState = np.stack((InitialGameImage, InitialGameImage, InitialGameImage, InitialGameImage), axis=2)
    # Reshape to the format expected by the model (batch_size, height, width, channels)
    GameState = GameState.reshape((1, GameState.shape[0], GameState.shape[1],
                                   GameState.shape[2]))  # Here shape[0] and shape[1] will be equal to IMGHEIGHT, IMGWIDTH

    # Main training loop
    for i in range(TOTAL_TRAIN_TIME):
        # Have the agent find the best action based on the current state (epsilon-greedy)
        BestAction = TheAgent.FindBestAct(GameState)

        # Play the next step in the game and get the reward and new screen image
        [ReturnScore, NewScreenImage] = TheGame.PlayNextMove(BestAction)

        # Process the new screen image
        NewGameImage = ProcessGameImage(NewScreenImage)
        # Reshape the new image to (batch_size, height, width, channels=1) format
        NewGameImage = NewGameImage.reshape(1, IMGHEIGHT, IMGWIDTH, 1)

        # Create the next state: combine the new image and the previous 3 images
        # We are concatenating along the channel dimension with axis=3
        NextState = np.append(NewGameImage, GameState[:, :, :, :3], axis=3)

        # Send the experience to the agent to capture
        TheAgent.CaptureSample((GameState, BestAction, ReturnScore, NextState))
        # Tell the agent to train the model
        TheAgent.Process()

        # Update the game state to the next state
        GameState = NextState

        # Print training progress and game score every 250 steps
        if i % 250 == 0:
            print(f"Train Time {i} Game Score {TheGame.GScore}")
            TrainHistory.append(TheGame.GScore)

# Start the training
TrainExperiment()
