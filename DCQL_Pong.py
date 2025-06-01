import random
import pygame
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D
from collections import deque

FPS = 60

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 420  # For the score or info area at the top
GAME_HEIGHT = 400  # Actual game area height

PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_BUFFER = 15

BALL_WIDTH = 20
BALL_HEIGHT = 20

# Define speeds in pixels/frame. Will be normalized with DeltaframeTime.
PADDLE_SPEED_PIXELS_PER_FRAME = 5
BALL_X_SPEED_PIXELS_PER_FRAME = 3
BALL_Y_SPEED_PIXELS_PER_FRAME = 3

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


def drawPaddle(switch, paddleYPos):
    """Draws a paddle at the specified position."""
    if switch == "left":
        paddle = pygame.Rect(PADDLE_BUFFER, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    elif switch == "right":
        paddle = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, paddle)


def drawBall(ballXPos, ballYPos):
    """Draws the ball at the specified position."""
    ball = pygame.Rect(ballXPos, ballYPos, BALL_WIDTH, BALL_HEIGHT)
    pygame.draw.rect(screen, WHITE, ball)


def updatePaddle(switch, action, paddleYPos, ballYPos, DeltaframeTime):
    """Updates the paddle's position."""
    # Normalize speed to be independent of FPS
    speed_factor = PADDLE_SPEED_PIXELS_PER_FRAME * (DeltaframeTime / (1000.0 / FPS))

    if switch == "left":
        if action == 1:  # Up
            paddleYPos -= speed_factor
        if action == 2:  # Down
            paddleYPos += speed_factor
        # If action is 0 (stop), the paddle does not move.

        # Boundary check
        if paddleYPos < 0:
            paddleYPos = 0
        if paddleYPos > GAME_HEIGHT - PADDLE_HEIGHT:
            paddleYPos = GAME_HEIGHT - PADDLE_HEIGHT

    elif switch == "right":  # Computer-controlled paddle (Simple AI)
        paddle_center = paddleYPos + PADDLE_HEIGHT / 2
        ball_center = ballYPos + BALL_HEIGHT / 2

        if paddle_center < ball_center:
            paddleYPos += speed_factor
        elif paddle_center > ball_center:
            paddleYPos -= speed_factor

        # Boundary check
        if paddleYPos < 0:
            paddleYPos = 0
        if paddleYPos > GAME_HEIGHT - PADDLE_HEIGHT:
            paddleYPos = GAME_HEIGHT - PADDLE_HEIGHT

    return paddleYPos


def updateBall(paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection, DeltaframeTime):
    """Updates the ball's position and direction."""
    # Normalize ball speed with DeltaframeTime
    ball_speed_x_factor = BALL_X_SPEED_PIXELS_PER_FRAME * (DeltaframeTime / (1000.0 / FPS))
    ball_speed_y_factor = BALL_Y_SPEED_PIXELS_PER_FRAME * (DeltaframeTime / (1000.0 / FPS))

    ballXPos += ballXDirection * ball_speed_x_factor
    ballYPos += ballYDirection * ball_speed_y_factor

    score = 0.0 # Default score is 0, changes only on events

    # Left paddle collision (agent's paddle) - Reward value further increased!
    if (ballXPos <= PADDLE_BUFFER + PADDLE_WIDTH and
            ballYPos + BALL_HEIGHT >= paddle1YPos and
            ballYPos <= (paddle1YPos + PADDLE_HEIGHT) and
            ballXDirection == -1):
        ballXDirection = 1  # Reverse direction
        score = 100.0  # MUCH HIGHER reward for the agent when hitting the ball (from previous 50.0 to 100.0)

    # Left wall (if agent misses the ball) - Penalty value kept the same
    elif ballXPos <= 0:
        ballXDirection = 1  # Send ball back (start new round)
        score = -10.0  # Large penalty
        # Reset ball to center
        ballXPos = WINDOW_WIDTH / 2
        ballYPos = random.randint(0, 9) * (GAME_HEIGHT - BALL_HEIGHT) / 9
        return [score, ballXPos, ballYPos, ballXDirection, ballYDirection]

    # Right paddle collision (AI's paddle) - Reward reduced to zero. To focus on agent's own hits.
    if (ballXPos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER - BALL_WIDTH and
            ballYPos + BALL_HEIGHT >= paddle2YPos and
            ballYPos <= paddle2YPos + PADDLE_HEIGHT and
            ballXDirection == 1):
        ballXDirection = -1  # Reverse direction
        score = 0.0 # No reward when hitting opponent's paddle (from previous 0.5 to 0.0)


    # Right wall (if AI misses the ball) - Reward value reduced. To focus on agent hitting its own ball.
    elif ballXPos >= WINDOW_WIDTH - BALL_WIDTH:
        ballXDirection = -1  # Send ball back (start new round)
        score = 5.0  # Lower reward for agent if AI misses the ball (from previous 10.0 to 5.0)
        # Reset ball to center
        ballXPos = WINDOW_WIDTH / 2
        ballYPos = random.randint(0, 9) * (GAME_HEIGHT - BALL_HEIGHT) / 9
        return [score, ballXPos, ballYPos, ballXDirection, ballYDirection]

    # Top wall collision
    if ballYPos <= 0:
        ballYPos = 0
        ballYDirection = 1

    # Bottom wall collision
    elif ballYPos >= GAME_HEIGHT - BALL_HEIGHT:
        ballYPos = GAME_HEIGHT - BALL_HEIGHT
        ballYDirection = -1

    return [score, ballXPos, ballYPos, ballXDirection, ballYDirection]


class PongGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('PONG DCQL ENV')
        self.paddle1YPos = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2  # Initial position of the left paddle
        self.paddle2YPos = GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2  # Initial position of the right paddle
        self.ballXPos = WINDOW_WIDTH / 2
        self.ballYPos = random.randint(0, 9) * (GAME_HEIGHT - BALL_HEIGHT) / 9 # Ball's Y position must be within GAME_HEIGHT

        self.clock = pygame.time.Clock()

        self.GScore = 0  # Overall score

        # Initial ball direction
        self.ballXDirection = random.sample([-1, 1], 1)[0]
        self.ballYDirection = random.sample([-1, 1], 1)[0]

    def InitialDisplay(self):
        """Draws the initial state of the game."""
        pygame.event.pump()  # Empty the event queue

        screen.fill(BLACK)  # Fill screen with black

        drawPaddle("left", self.paddle1YPos)
        drawPaddle("right", self.paddle2YPos)

        drawBall(self.ballXPos, self.ballYPos)

        pygame.display.flip()  # Update the screen

    def PlayNextMove(self, action):
        """Plays the next step of the game and returns the screen image."""
        # Get DeltaframeTime from the tick() method, which both limits FPS and provides elapsed time.
        DeltaframeTime = self.clock.tick(FPS)

        pygame.event.pump()  # Process events (necessary)

        score = 0  # Score for this step

        screen.fill(BLACK)  # Clear the screen

        # Update left paddle (controlled by agent)
        self.paddle1YPos = updatePaddle("left", action, self.paddle1YPos, self.ballYPos, DeltaframeTime)
        drawPaddle("left", self.paddle1YPos)

        # Update right paddle (controlled by computer)
        self.paddle2YPos = updatePaddle("right", 0, self.paddle2YPos, self.ballYPos, DeltaframeTime)
        drawPaddle("right", self.paddle2YPos)

        # Update the ball
        [score, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection] = updateBall(
            self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos,
            self.ballXDirection, self.ballYDirection, DeltaframeTime
        )

        drawBall(self.ballXPos, self.ballYPos)

        # Update overall score: simply add the current step's score
        self.GScore += score

        # Get screen image
        ScreenImage = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()  # Finally update and display the screen

        return [score, ScreenImage]

# Main block for testing purposes
if __name__ == '__main__':
    pg = PongGame()
    pg.InitialDisplay()
    # You can add a small loop for testing
    # for _ in range(300): # Run for about 5 seconds
    #     current_score, screen_img = pg.PlayNextMove(0) # Do nothing (don't move)
    #     print(f"Current Score: {current_score}, Game Score: {pg.GScore}")
    #     time.sleep(0.01) # To see it slower
    # pygame.quit()
