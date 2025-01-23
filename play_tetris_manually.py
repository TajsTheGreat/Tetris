from selfmadetetris import Board
import pygame

env = Board()
env.reset()
exit_program = False
counter = 0

x = 0
down = False
space = False
reserve = False
rotate = False

clock = pygame.time.Clock()
fps = 50

while not exit_program:
    counter += 1
    if counter > 100000:
        counter = 0

    env.render(counter)

    # controls
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True  # Fixed typo: Should set exit_program, not 'done'
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                rotate = True
            if event.key == pygame.K_DOWN:
                down = True  # Changed from 'pressing_down' to 'down'
            if event.key == pygame.K_LEFT:
                x = -1
            if event.key == pygame.K_RIGHT:
                x = 1
            if event.key == pygame.K_SPACE:
                space = True
            if event.key == pygame.K_ESCAPE:
                env.reset()  # Added missing parentheses to call the reset method
            if event.key == pygame.K_x:
                reserve = True
            if event.key == pygame.K_q:
                exit_program = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                down = False  # Changed from 'pressing_down' to 'down'
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                x = 0  # Reset horizontal movement when key is released

    # Create action tuple
    action = (x, down, rotate, space, reserve)
    
    # Perform action
    env.step(action)

    # Reset action variables
    x = 0
    down = False
    rotate = False
    space = False
    reserve = False

    # Control frame rate
    clock.tick(fps)

env.close()
