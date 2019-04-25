import numpy as np
import sys
import pygame
import random
import operator
import colorsys
from brain import NeuralNetwork
from pygame.locals import *


# CONSTANTS:

FPS = 50
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

BLACK = (0,   0,   0)
WHITE = (255, 255, 255)
GREEN = (0, 255,   0)
RED = (255,   0,   0)
BLUE = (0,   0, 255)

am_of_colors = 24
l_colors = [(0, 0, 153), (0, 0, 230), (102, 102, 255), (179, 179, 255),
            (115, 0, 153), (172, 0, 230), (210, 77, 255), (236, 179, 255),
            (0, 102, 0), (0, 179, 0), (0, 255, 0), (179, 255, 179),
            (128, 128, 0), (204, 204, 0), (255, 255, 77), (255, 255, 179),
            (96, 64, 31), (154, 102, 50), (205, 153, 101), (224, 191, 159),
            (128, 0, 0), (204, 0, 0), (255, 26, 26), (255, 153, 153)]

LINE_1 = 3/9
LINE_2 = 4/9
LINE_3 = 5/9
LINE_4 = 6/9
LINE_WIDTH = 5

PLAYER_HEIGHT = SCREEN_HEIGHT-50

OBS_STEP = 5
OBS_SPACE = 150

# for players positions
left_track = int(((LINE_1+LINE_2)/2.0)*SCREEN_WIDTH),  PLAYER_HEIGHT
middle_track = int(((LINE_2+LINE_3)/2.0)*SCREEN_WIDTH),  PLAYER_HEIGHT
right_track = int(((LINE_3+LINE_4)/2.0)*SCREEN_WIDTH),  PLAYER_HEIGHT

l_player_pos = [left_track, middle_track, right_track]

# for obstacle start positions
left_track_obstacle = int(((LINE_1+LINE_2)/2.0)*SCREEN_WIDTH),  0
middle_track_obstacle = int(((LINE_2+LINE_3)/2.0)*SCREEN_WIDTH),  0
right_track_obstacle = int(((LINE_3+LINE_4)/2.0)*SCREEN_WIDTH),  0

l_start_obs = [left_track_obstacle, middle_track_obstacle, right_track_obstacle]


# HELPERS


def random_color():
    rgbl = [255, 0, 0]
    random.shuffle(rgbl)
    return tuple(rgbl)


def draw_bounds(screen):
    screen.fill(WHITE)
    pygame.draw.line(screen, BLACK, [LINE_1*SCREEN_WIDTH, 0],
                     [LINE_1*SCREEN_WIDTH, SCREEN_HEIGHT], LINE_WIDTH)
    pygame.draw.line(screen, BLACK, [LINE_2*SCREEN_WIDTH, 0],
                     [LINE_2*SCREEN_WIDTH, SCREEN_HEIGHT], LINE_WIDTH)
    pygame.draw.line(screen, BLACK, [LINE_3*SCREEN_WIDTH, 0],
                     [LINE_3*SCREEN_WIDTH, SCREEN_HEIGHT], LINE_WIDTH)
    pygame.draw.line(screen, BLACK, [LINE_4*SCREEN_WIDTH, 0],
                     [LINE_4*SCREEN_WIDTH, SCREEN_HEIGHT], LINE_WIDTH)

# GAME ELEMENTS


class Obstacle(object):
    size = 20

    def __init__(self, screen, position, step):
        self.color = BLACK
        self.position_x, self.position_y = position
        self.screen = screen
        self.step = step

    def position(self):
        return (self.position_x, self.position_y)

    def move(self):
        self.position_y += self.step
        if self.position_y > SCREEN_HEIGHT:
            return False
        else:
            return True

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position(), Obstacle.size)


class Obstacles(object):

    def __init__(self, screen, step, space):
        self.screen = screen
        self.step = step
        self.l_obstacles = []
        self.last = None
        self.space = space

    def restart(self):
        self.l_obstacles = []
        self.last = None

    # internal funct
    def add_obs(self):
        position = random.choice(l_start_obs)
        obstacle = Obstacle(self.screen, position, self.step)
        self.l_obstacles.append(obstacle)
        self.last = obstacle

    def add_obstacle(self):
        if self.last != None:
            if self.last.position_y > self.space:
                self.add_obs()
                return True

            else:
                return False
        else:
            self.add_obs()
            return True

    def move(self):
        l_rm_list = []
        for obs in self.l_obstacles:
            if not(obs.move()):
                l_rm_list.append(obs)
        for obs in l_rm_list:
            self.l_obstacles.remove(obs)

    def draw(self):
        for obs in self.l_obstacles:
            obs.draw()

    # Return dictionary with keys being tracks and values being nearest obstacles on those tracks
    def first_obstacles(self):
        temp = Obstacle(self.screen, (0, 0), self.step)
        left_obs = temp
        middle_obs = temp
        right_obs = temp

        for obs in self.l_obstacles:
            if obs.position_x == left_track[0]:
                if obs.position_y > left_obs.position_y:
                    left_obs = obs
            elif obs.position_x == middle_track[0]:
                if obs.position_y > middle_obs.position_y:
                    middle_obs = obs
            elif obs.position_x == right_track[0]:
                if obs.position_y > right_obs.position_y:
                    right_obs = obs

        dic_firstObs = {
            left_track: left_obs.position_y,
            middle_track: middle_obs.position_y,
            right_track: right_obs.position_y
        }
        return dic_firstObs


class Player(object):
    size = 20

    def __init__(self, screen):
        self.color = random_color()
        self.position = random.choice(l_player_pos)
        self.screen = screen
        self.points = 0
        self.NN = NeuralNetwork()

    def installBrain(self, NN):
        self.NN = NN

    def setColor(self, color):
        self.color = color

    def crossover(self, B):
        child = Player(self.screen)
        child.NN.crossover(B.NN)

        return child

    def mutation(self):
        self.NN.mutation()

    def move(self, X):

        if(self.NN == None):
            return False

        input = np.zeros((6, 1))

        if(self.position == left_track):
            input[0, 0] = 1
        if(self.position == middle_track):
            input[1, 0] = 1
        if(self.position == right_track):
            input[2, 0] = 1

        input *= 3

        input[3, 0] = X.get(left_track)/100
        input[4, 0] = X.get(middle_track)/100
        input[5, 0] = X.get(right_track)/100

        self.NN.forwardPropagation(input)

        if(self.NN.Y[0, 0] == 1 and self.NN.Y[1, 0] == 0):
            self.move_left()

        if(self.NN.Y[0, 0] == 0 and self.NN.Y[1, 0] == 1):
            self.move_right()

        return True

    def position(self):
        return position

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position, Player.size)

    def move_left(self):
        if self.position == left_track:
            return False
        else:
            if self.position == right_track:
                self.position = middle_track
            else:
                self.position = left_track
            return True

    def move_right(self):
        if self.position == right_track:
            return False
        else:
            if self.position == middle_track:
                self.position = right_track
            else:
                self.position = middle_track
            return True

    def reward(self):
        self.points += 1


class Generations(object):

    def __init__(self, amount, screen):
        self.amount = amount
        self.no_of_generation = 1
        self.screen = screen
        self.l_players = []
        self.l_loosers = []
        self.colors = l_colors[:]
        self.best_fit = 0

        while len(self.colors) < amount:
            self.colors.append((100, 100, 100))

        for i in range(0, self.amount):
            self.l_players.append(Player(self.screen))
        self.setColor()

    def setColor(self):
        for player, color in zip(self.l_players, self.colors):
            player.setColor(color)

    def createNextGen(self):
        amount_of_b = round(self.amount/3)

        self.l_loosers.reverse()
        self.l_best = self.l_loosers[0:amount_of_b]

        self.l_players = []
        self.l_loosers = []

        for elem in self.l_best:
            self.l_players.append(elem)

        while len(self.l_players) < self.amount:

            parentA = random.choice(self.l_best)
            parentB = random.choice(self.l_best)

            self.l_players.append(parentA.crossover(parentB))

        for player in self.l_players:
            player.mutation()

        self.setColor()
        self.no_of_generation += 1


class Game(object):

    def __init__(self):
        pygame.init()
        self.fpsClock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        self.surface = pygame.Surface(self.screen.get_size())
        self.surface = self.surface.convert()
        self.surface.fill(WHITE)
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 25)
        self.logs = ""
        self.best_fitness = ""
        self.results = []

        self.l_players = []
        self.l_loosers = []
        self.obstacles = Obstacles(self.screen, OBS_STEP, OBS_SPACE)
        self.dict_fObs = self.obstacles.first_obstacles()
        self.textsurface = self.myfont.render(str(self.dict_fObs.values()), False, BLUE)
        self.textsurface2 = self.myfont.render(self.logs, False, RED)
        self.gens = Generations(24, self.screen)

        self.l_txt_surfs = []

    def start(self):
        for i in range(0, 10000):
            self.play()
            self.obstacles.restart()
            self.gens.createNextGen()

    def play(self):

        done = False

        while done == False:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            draw_bounds(self.screen)
            self.update()
            self.check_for_crash()

            if len(self.gens.l_players) == 0:
                done = True
                self.logs = "Generation Over"
                self.results.append((self.gens.no_of_generation, self.gens.l_loosers[-1].points))
                #self.best_fitness = str(self.gens.l_loosers[-1].points)
                self.update()
                self.logs = ''
                pygame.time.wait(1000)

            self.fpsClock.tick(FPS)

    def update(self):
        self.move_players()
        self.draw_players()
        self.obstacles.add_obstacle()
        self.obstacles.draw()
        self.obstacles.move()
        self.dict_fObs = self.obstacles.first_obstacles()
        self.textsurface = self.myfont.render(str(self.dict_fObs.values()), False, BLUE)
        self.textsurface2 = self.myfont.render(
            'Generation: '+str(self.gens.no_of_generation), False, RED)
        self.textsurface3 = self.myfont.render(self.logs, False, RED)
        if(len(self.gens.l_players) > 0):
            self.textsurface4 = self.myfont.render(
                'points: '+str(self.gens.l_players[0].points), False, RED)
        else:
            self.textsurface4 = self.myfont.render(
                'points: '+str(self.gens.l_loosers[-1].points), False, GREEN)

        self.screen.blit(self.textsurface, (0, 0))
        self.screen.blit(self.textsurface2, (SCREEN_WIDTH-180, 0))
        self.screen.blit(self.textsurface3, (SCREEN_WIDTH-180, 70))
        self.screen.blit(self.textsurface4, (SCREEN_WIDTH-180, 140))

        self.updateResults()

        pygame.display.flip()

    def updateResults(self):

        START_POINT = (SCREEN_WIDTH-200, 180)
        heigh_step = 30

        self.l_txt_surfs = []

        i = 0
        j = len(self.results)-1

        while i < 6 and j > 0:
            str1 = 'Gen: '+str(self.results[j][0])
            str2 = ' best fit: '+str(self.results[j][1])
            txt_surf = self.myfont.render(str1 + str2, False, BLUE)
            self.screen.blit(txt_surf, (START_POINT[0], START_POINT[1] + i*heigh_step))
            i += 1
            j -= 1

    def check_for_crash(self):
        l_temp_loosers = []
        i = 0
        for player in self.gens.l_players:
            if PLAYER_HEIGHT - self.dict_fObs.get(player.position) - Player.size - Obstacle.size < 0:
                l_temp_loosers.append(player)
                #self.logs += "Nr"+str(i)+" crash\n(points: "+str(player.points)+")\n"
            i += 1
        for player in l_temp_loosers:
            self.gens.l_loosers.append(player)
            self.gens.l_players.remove(player)

    def draw_players(self):
        for player in self.gens.l_players:
            player.draw()
            player.reward()

    def move_players(self):
        for player in self.gens.l_players:
            player.move(self.dict_fObs)


if __name__ == '__main__':
    game = Game()
    game.start()


""""

# GAME
pygame.init()
fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
surface = pygame.Surface(screen.get_size())
surface = surface.convert()
surface.fill(WHITE)
clock = pygame.time.Clock()
pygame.key.set_repeat(100, 40)

pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 25)

# SOME NUMBERS FOR GAME LOOP
done = False
player = Player(screen)
obstacles = Obstacles(screen, OBS_STEP, OBS_SPACE)

# GAME LOOP
while done == False:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    draw_bounds()
    player.draw()

    obstacles.add_obstacle()
    obstacles.draw()
    obstacles.move()

    dict_fObs = obstacles.first_obstacles()

    textsurface = myfont.render(str(dict_fObs.values()), False, BLUE)
    screen.blit(textsurface, (0, 0))

    if middle_track[1] - dict_fObs.get(player.position) - Player.size - Obstacle.size < 0:
        textsurface2 = myfont.render("crash", False, RED)
        screen.blit(textsurface2, (300, 300))
        # pygame.time.wait(100)

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        player.move_left()
        print("left pressed")

    if keys[pygame.K_RIGHT]:
        player.move_right()
        print("right pressed")

    fpsClock.tick(FPS)

    # END OF GAME LOOP
    pygame.display.flip()
# ENDED
"""
