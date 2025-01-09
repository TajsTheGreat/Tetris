import pygame
import random

# colors for each tetrimino 
colors = [
    (255, 255, 255), # white
    (0, 215, 0), # green
    (255, 0, 0), # red 
    (0, 0, 255), # blue
    (255, 120, 0), # orange
    (255, 255, 0), # yellow
    (255, 0, 255), # purple
    (102, 255, 255), # cyan
]

# class for a tetrimino
class Piece:
    x = 0
    y = 0

    # coordinates for each tetrimino
    pieces = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.pieces) - 1)
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.pieces[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.pieces[self.type])

# class for the game itself
class Tetris:
    def __init__(self, height, width):
        self.score = 0
        self.state = "start"
        self.field = []
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.piece= None
        self.next_piece = None
        self.reserved_piece = None
    
        self.height = height
        self.width = width
        self.heights = [0] * width
    
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(0)
            self.field.append(new_line)
        
    # Loads the next tetrimino
    def next_new_piece(self):
        self.next_piece = Piece(3, 0)

    # creates a new tetrimino at the top middle
    def new_piece(self):
        self.piece = self.next_piece
        self.next_new_piece()
    
    def reserve_piece(self):
        if self.reserve_piece is not None:
            temp = self.piece
            self.piece = self.reserved_piece
            self.reserved_piece = temp
        else:
            self.reserved_piece = self.piece
            self.new_piece()

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.piece.image():
                    if i + self.piece.y > self.height - 1 or \
                            j + self.piece.x > self.width - 1 or \
                            j + self.piece.x < 0 or \
                            self.field[i + self.piece.y][j + self.piece.x] > 0:
                        intersection = True
        return intersection

    # checks if lines are full and clears them if they are
    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for k in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[k][j] = self.field[k - 1][j]
        
        # score system based on actual tetris rules
        if lines == 1:
            self.score += 40
        elif lines == 2:
            self.score += 100
        elif lines == 3:
            self.score += 300
        elif lines == 4:
            self.score += 1200

    # fast drops pieces
    def go_space(self):
        while not self.intersects():
            self.piece.y += 1
        self.piece.y -= 1
        self.freeze()

    # makes pieces go down faster
    def go_down(self):
        self.piece.y += 1
        if self.intersects():
            self.piece.y -= 1
            self.freeze()
    
    # measures the height of the field
    def height_measure(self):
        for j in range(self.width):
            for i in range(self.height):
                if self.field[i][j] > 0:
                    self.heights[j] = self.height - i
                    break
        print(self.heights)

    # freezes the piece in place
    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.piece.image():
                    self.field[i + self.piece.y][j + self.piece.x] = self.piece.color
        self.break_lines()
        self.new_piece()
        if self.intersects():
            self.state = "gameover"   
        self.height_measure()

    # moves the piece left or right
    def go_side(self, dx):
        old_x = self.piece.x
        self.piece.x += dx
        if self.intersects():
            self.piece.x = old_x

    # rotates the piece
    def rotate(self):
        old_rotation = self.piece.rotation
        self.piece.rotate()
        if self.intersects():
            self.piece.rotation = old_rotation
    
    # steps the game
    def step(self, action):
        x, down, rotate, space, reserve = action
        if down:
            self.go_down()
        if x < 0 or x > 0:
            self.go_side(x)
        if rotate:
            self.rotate()
        if space:
            self.go_space()
        if reserve:
            self.reserve_piece()
    
    # resets the game
    def reset(self):
        self.score = 0
        self.state = "start"
        self.field = []
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.piece= None
        self.next_piece = None
        self.reserved_piece = None
    
        self.heights = [0] * self.width
    
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.field.append(new_line)
        
        self.new_piece()
        self.next_new_piece()
        self.height_measure()
        print("her")

class Board():
    # colors for each tetrimino 
    colors = [
        (255, 255, 255), # white
        (0, 215, 0), # green
        (255, 0, 0), # red 
        (0, 0, 255), # blue
        (255, 120, 0), # orange
        (255, 255, 0), # yellow
        (255, 0, 255), # purple
        (102, 255, 255), # cyan
    ]

    def __init__(self):
        pygame.init()
        # Set up game
        self.game = Tetris(20, 10)
        self.clock = pygame.time.Clock()

        # Define colors for screenfill and text fond
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)

        # screen size
        self.size = (400, 500)
        self.screen = pygame.display.set_mode(self.size)

        pygame.display.set_caption("Tetris")

        # Loop until the player hits the close button.
        self.done = False
        self.fps = 50
        self.counter = 0
        self.pressing_down = False

        # background color
        self.screen.fill(self.BLACK)
        self.pressing_down = False
    
    def reset(self):
        # Program status
        self.game_over = False
        self.won = False
        self.game.reset()
    
    def step(self, action):
        # step the game
        self.game.step(action)
        x, down, rotate, space, reserve = action
        self.pressing_down = down
        print("here")
        return 0, down, False, False, False

        
    
    def render(self, counter):
        if self.game.state == "start" and self.game.piece is None:
            self.game.new_piece()

        if counter % (50 // 2) == 0 or self.pressing_down: # if code fucks up, change back to (fps // game.level // 2) == 0:
            if self.game.state == "start":
                self.game.go_down()
        
        # background color
        self.screen.fill(self.BLACK)
            
        # Draw the grid
        for i in range(self.game.height):
            for j in range(self.game.width):
                pygame.draw.rect(self.screen, self.GRAY, [self.game.x + self.game.zoom * j, self.game.y + self.game.zoom * i, self.game.zoom, self.game.zoom], 1)
                if self.game.field[i][j] > 0:
                    pygame.draw.rect(self.screen, colors[self.game.field[i][j]],
                                    [self.game.x + self.game.zoom * j + 1, self.game.y + self.game.zoom * i + 1, self.game.zoom - 2, self.game.zoom - 1])

        # Draw the current piece
        if self.game.piece is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in self.game.piece.image():
                        pygame.draw.rect(self.screen, colors[self.game.piece.color],
                                        [self.game.x + self.game.zoom * (j + self.game.piece.x) + 1,
                                        self.game.y + self.game.zoom * (i + self.game.piece.y) + 1,
                                        self.game.zoom - 2, self.game.zoom - 2])
        
        # Draw the next piece
        if self.game.next_piece is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in self.game.next_piece.image():
                        pygame.draw.rect(self.screen, colors[self.game.next_piece.color],
                                        [70 + self.game.zoom * (j + 12) + 1,
                                        60 + self.game.zoom * (i + 2) + 1,
                                        self.game.zoom - 2, self.game.zoom - 2])
            pygame.draw.rect(self.screen, self.WHITE, [70 + self.game.zoom * 12, 60 + self.game.zoom * 2, self.game.zoom * 4, self.game.zoom * 4], 1)
        
        # Draw the reserved piece
        if self.game.reserved_piece is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in self.game.reserved_piece.image():
                        pygame.draw.rect(self.screen, colors[self.game.reserved_piece.color],
                                        [-30 + self.game.zoom * (j + 2) + 1,
                                        60 + self.game.zoom * (i + 2) + 1,
                                        self.game.zoom - 2, self.game.zoom - 2])
            pygame.draw.rect(self.screen, self.WHITE, [-30 + self.game.zoom * 2, 60 + self.game.zoom * 2, self.game.zoom * 4, self.game.zoom * 4], 1)

        font = pygame.font.SysFont('Comic Sans', 25, True, False)
        font1 = pygame.font.SysFont('Comic Sans', 65, True, False)
        text = font.render("Score: " + str(self.game.score), True, self.WHITE)
        next_text = font.render("Next:", True, self.WHITE)
        reserved_text = font.render("Reserved:", True, self.WHITE)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_restart = font1.render("Press ESC", True, (255, 215, 0))

        self.screen.blit(text, [0, 0])
        self.screen.blit(next_text, [315, 25])
        self.screen.blit(reserved_text, [0, 25])
        if self.game.state == "gameover":
            self.screen.blit(text_game_over, [20, 200])
            self.screen.blit(text_restart, [25, 265])
        pygame.display.flip()
    
    # Close the game
    def close(self):
        pygame.quit()
        


# # Initialize the game engine
# pygame.init()

# # Define colors for screenfill and text fond
# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
# GRAY = (128, 128, 128)

# # screen size
# size = (400, 500)
# screen = pygame.display.set_mode(size)

# pygame.display.set_caption("Tetris")

# # Loop until the player hits the close button.
# done = False
# clock = pygame.time.Clock()
# fps = 50
# game = Tetris(20, 10)
# counter = 0

# pressing_down = False

# while not done:
#     if game.piece is None:
#         game.new_piece()
#     counter += 1
#     if counter > 100000:
#         counter = 0

#     if counter % (fps // 2) == 0 or pressing_down: # if code fucks up, change back to (fps // game.level // 2) == 0:
#         if game.state == "start":
#             game.go_down()

#     # controls
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             done = True
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_UP:
#                 game.rotate()
#             if event.key == pygame.K_DOWN:
#                 pressing_down = True
#             if event.key == pygame.K_LEFT:
#                 game.go_side(-1)
#             if event.key == pygame.K_RIGHT:
#                 game.go_side(1)
#             if event.key == pygame.K_SPACE:
#                 game.go_space()
#             if event.key == pygame.K_ESCAPE:
#                 game.__init__(20, 10)
#             if event.key == pygame.K_x:
#                 game.reserve_piece()

#     if event.type == pygame.KEYUP:
#             if event.key == pygame.K_DOWN:
#                 pressing_down = False

#     # background color
#     screen.fill(BLACK)

#     # Draw the grid
#     for i in range(game.height):
#         for j in range(game.width):
#             pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
#             if game.field[i][j] > 0:
#                 pygame.draw.rect(screen, colors[game.field[i][j]],
#                                  [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])

#     # Draw the current piece
#     if game.piece is not None:
#         for i in range(4):
#             for j in range(4):
#                 p = i * 4 + j
#                 if p in game.piece.image():
#                     pygame.draw.rect(screen, colors[game.piece.color],
#                                      [game.x + game.zoom * (j + game.piece.x) + 1,
#                                       game.y + game.zoom * (i + game.piece.y) + 1,
#                                       game.zoom - 2, game.zoom - 2])
    
#     # Draw the next piece
#     if game.next_piece is not None:
#         for i in range(4):
#             for j in range(4):
#                 p = i * 4 + j
#                 if p in game.next_piece.image():
#                     pygame.draw.rect(screen, colors[game.next_piece.color],
#                                      [70 + game.zoom * (j + 12) + 1,
#                                       60 + game.zoom * (i + 2) + 1,
#                                       game.zoom - 2, game.zoom - 2])
#         pygame.draw.rect(screen, WHITE, [70 + game.zoom * 12, 60 + game.zoom * 2, game.zoom * 4, game.zoom * 4], 1)
    
#     # Draw the reserved piece
#     if game.reserved_piece is not None:
#         for i in range(4):
#             for j in range(4):
#                 p = i * 4 + j
#                 if p in game.reserved_piece.image():
#                     pygame.draw.rect(screen, colors[game.reserved_piece.color],
#                                      [-30 + game.zoom * (j + 2) + 1,
#                                       60 + game.zoom * (i + 2) + 1,
#                                       game.zoom - 2, game.zoom - 2])
#         pygame.draw.rect(screen, WHITE, [-30 + game.zoom * 2, 60 + game.zoom * 2, game.zoom * 4, game.zoom * 4], 1)

#     font = pygame.font.SysFont('Comic Sans', 25, True, False)
#     font1 = pygame.font.SysFont('Comic Sans', 65, True, False)
#     text = font.render("Score: " + str(game.score), True, WHITE)
#     next_text = font.render("Next:", True, WHITE)
#     reserved_text = font.render("Reserved:", True, WHITE)
#     text_game_over = font1.render("Game Over", True, (255, 125, 0))
#     text_restart = font1.render("Press ESC", True, (255, 215, 0))

#     screen.blit(text, [0, 0])
#     screen.blit(next_text, [315, 25])
#     screen.blit(reserved_text, [0, 25])
#     if game.state == "gameover":
#         screen.blit(text_game_over, [20, 200])
#         screen.blit(text_restart, [25, 265])

#     pygame.display.flip()
#     clock.tick(fps)

# pygame.quit()