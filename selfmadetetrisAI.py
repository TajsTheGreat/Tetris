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

    # returns the image of the piece, basically where it is in the grid
    def image(self):
        return self.pieces[self.type][self.rotation % len(self.pieces[self.type])]

    # rotates the piece one time
    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.pieces[self.type])
    
    # rotates the piece to a specific value
    def specific_rotate(self, value):
        self.rotation = value % len(self.pieces[self.type])

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
        self.lowest = 100
    
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
    
    # reserves a piece
    def reserve_piece(self):
        if self.reserve_piece is not None:
            temp = self.piece
            self.piece = self.reserved_piece
            self.reserved_piece = temp
        else:
            self.reserved_piece = self.piece
            self.new_piece()

    # checks if the piece intersects with the field
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
                        break
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

    # freezes the piece in place
    def freeze(self):
        count = 0
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.piece.image():
                    count += 1
                    self.field[i + self.piece.y][j + self.piece.x] = self.piece.color
                    if count == 4:
                        self.lowest = self.height - (self.piece.y + i)

        self.break_lines()
        self.new_piece()
        if self.intersects():
            self.state = "gameover" 
        self.height_measure()
        for i in range(len(self.heights)):
            if self.heights[i] > self.height - 4:
                self.state = "gameover"

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
    
    # moves the piece to the desired location and rotates it
    def place(self, firstvalue, secondvalue):
        self.piece.specific_rotate(firstvalue)
        while not (self.intersects() or self.piece.x == secondvalue):
            if 0 < (secondvalue - self.piece.x):
                self.piece.x += 1
            elif 0 > (secondvalue - self.piece.x):
                self.piece.x -= 1
        if self.intersects():
            if self.piece.x > 0:
                self.piece.x -= 1
            else:
                self.piece.x += 1
        self.go_space()
    
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
    
    # returns the current type of piece
    def get_piece(self):
        arr = []
        if self.piece is None:
            for i in range(7):
                arr.append(0)
            return arr
        for i in range(7):
            arr.append(1 if self.piece.type == i else 0)
        return arr

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
        self.game = Tetris(24, 10)
        self.clock = pygame.time.Clock()

        # Define colors for screenfill and text fond
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)

        # screen size
        self.size = (500, 550)
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

        # controls wheter the game is rendering or not
        self.rendering = True
    
    def reset(self):
        # Program status
        self.game_over = False
        self.won = False
        self.game.reset()
    
    def calculate_unreachable_holes(self, field, height, width):
        visited = [[False for _ in range(width)] for _ in range(height)]
        holes = 0

        for j in range(width):
            # Start from the top and mark reachable spaces
            for i in range(height):
                if field[i][j] > 0:
                    break
                visited[i][j] = True
    
            # Count unreachable holes in the column
            for i in range(height):
                if not visited[i][j] and field[i][j] == 0:
                    holes += 1

        return holes
    
    def step(self, value):
        if self.game.state == "gameover":
            print("Game over")
            temp = []
            for i in range(len(self.game.heights)):
                temp.append(self.game.heights[i])
            for i in range(7):
                temp.append(0)
            temp.append(self.calculate_unreachable_holes(self.game.field, self.game.height, self.game.width))
            return tuple(temp), 0, True
        


        # Calculate holes before placing the piece
        holes_before = self.calculate_unreachable_holes(self.game.field, self.game.height, self.game.width)

        # score and height total before the piece is placed
        score = self.game.score
        height_total = 0
        height_avg = sum(self.game.heights) / len(self.game.heights)
        height_var = 0
        min_height = self.game.height

        for i in range(len(self.game.heights)):
            height_total += self.game.heights[i]
            height_var += (self.game.heights[i] - height_avg) ** 2
            if self.game.heights[i] < min_height:
                min_height = self.game.heights[i]

        height_var = height_var / len(self.game.heights)
        

        # Place the piece
        firstvalue = int(str(value)[0]) if value > 9 else 0
        secondvalue = int(str(value)[1]) if value > 9 else value
        self.game.place(firstvalue, secondvalue)

        # Calculate holes after placing the piece
        holes_after = self.calculate_unreachable_holes(self.game.field, self.game.height, self.game.width)

        # Calculate rewards
        height_total2 = 0
        for i in range(len(self.game.heights)):
            height_total2 += self.game.heights[i]

        self.height_reward = 5 if height_total - height_total2 == -4 else (height_total - height_total2)
        

        self.height_low_reward = (min_height - self.game.lowest + 2) * 4

        score = self.game.score - score
        if score == 40:
            score = score * 2
        elif score == 100:
            score = score * 1.5
        elif score == 300:
            score = score / 0.75
        elif score == 1200:
            score = score / 3
        
        self.test_score = score*8

        self.bumpiness = -sum([abs(self.game.heights[i] - self.game.heights[i + 1]) for i in range(len(self.game.heights) - 1)])
       
        if holes_before - holes_after == 0:
            self.hole_opening_reward = 20
        else:
            self.hole_opening_reward = (holes_before - holes_after) * 60
        

        return self.get_state(), (score*8 + self.height_low_reward + self.bumpiness + self.hole_opening_reward), False if self.game.state == "start" else True
    
    def get_state(self):
        if self.game.state == "start" and self.game.piece is None:
            self.game.new_piece()
        temp = []
        for i in range(len(self.game.heights)):
            temp.append(self.game.heights[i])
        for i in range(7):
            temp.append(self.game.get_piece()[i])
        temp.append(self.calculate_unreachable_holes(self.game.field, self.game.height, self.game.width))
        # return the state of the game
        return tuple(temp)

    def render(self, snapshot=False):
        if self.game.state == "start" and self.game.piece is None:
            self.game.new_piece()
        
        # check if should render
        if not self.rendering:
            pygame.display.flip()
            return

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
        if self.game.state == "gameover" and not snapshot:
            self.screen.blit(text_game_over, [20, 200])
            self.screen.blit(text_restart, [25, 265])
        
        # Draw line of height limits
        pygame.draw.line(self.screen, self.RED, (self.game.x, self.game.y + 4 * self.game.zoom), (self.game.x + self.game.width * self.game.zoom, self.game.y + 4 * self.game.zoom), 1)

        pygame.display.flip()
    
    # Close the game
    def close(self):
        pygame.quit()
