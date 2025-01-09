from agent import Agent
from selfmadetetrisRandom import Board
import pygame
import random

env = Board()
env.reset()
exit_program = False
env.render()

clock = pygame.time.Clock()
fps = 50
active = True

while not exit_program:
    
    # controls
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                first = random.randint(0, 9)
                second = random.randint(0, 3)
                print(f"Step: {first}, {second}")
                env.step([first, second])
                env.render()
                print("Step")
            if event.key == pygame.K_ESCAPE:
                env.reset()
                env.render()
                print("Reset")
    

    
    # Update the display (even if it's just a blank screen)
    pygame.display.flip()
    clock.tick(fps)

env.close()