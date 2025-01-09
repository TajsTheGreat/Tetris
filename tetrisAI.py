from agent import Agent
from selfmadetetrisAI import Board
import pygame
import threading

env = Board()
env.reset()
exit_program = False
input_value = []
env.render()

clock = pygame.time.Clock()
fps = 50
input_text = ""
active = True

def get_input():
    global input_text
    global active
    input_text = input("Enter: ")
    active = False
    print("Input received")

input_thread = threading.Thread(target=get_input)
input_thread.start()

while not exit_program:


    if not active:
        active = True
        input_value.append(int(input_text))
        input_text = ""
        input_thread = threading.Thread(target=get_input)
        input_thread.start()
        if len(input_value) > 1:
            env.step(input_value)
            input_value = []
            env.render()
    
    # controls
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
    

    
    # Update the display (even if it's just a blank screen)
    pygame.display.flip()
    clock.tick(fps)

env.close()