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
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                rotate = True
            if event.key == pygame.K_DOWN:
                pressing_down = True
            if event.key == pygame.K_LEFT:
                x = -1
            if event.key == pygame.K_RIGHT:
                x = 1
            if event.key == pygame.K_SPACE:
                space = True
            if event.key == pygame.K_ESCAPE:
                env.reset
            if event.key == pygame.K_x:
                reserve = True
            if event.key == pygame.K_q:
                exit_program = True
        
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                pressing_down = False
    
    action = x, down, rotate, space, reserve
    action = env.step(action)
    clock.tick(fps)

env.close()