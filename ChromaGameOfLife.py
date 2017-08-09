import pygame
from pygame.locals import *
from random import randint
import operator
import math
import ast

NX = 63
NY = 63

SIZE = 8

WIDTH = NX * SIZE 
HEIGHT = NY * SIZE
BLACK = (0, 0, 0)

GPS = 5


def main():
    pygame.init()
    pygame.display.set_caption("Conway")
    screen = pygame.display.set_mode((WIDTH,HEIGHT))

    mainloop(screen)

def mainloop(screen):
    last_iteration= open("last_iteration.txt","r")
    IA_board = last_iteration.read()
    last_iteration.close
    board = ast.literal_eval(IA_board)
    
    fps_clock = pygame.time.Clock()
    pause  = True
    gen = 0
    CELL_SELECT = [255, 255, 255]
    while True:        
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                quit()

            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    pause = not pause
                if event.key == K_r:
                    CELL_SELECT = [255, 0, 0]
                if event.key == K_v:
                    CELL_SELECT = [0, 255, 0]
                if event.key == K_b:
                    CELL_SELECT = [0, 0, 255]
                if event.key == K_w:
                    CELL_SELECT = [255, 255, 255]
                    
            if event.type == MOUSEBUTTONDOWN:
                x, y  = event.pos
                x //= SIZE
                y //= SIZE
                board[y][x][0] = not board[y][x][0]
                if board[y][x][0]:
                    board[y][x][1] = CELL_SELECT
                

        if not pause:
            fps_clock.tick(GPS)
            board, gen = update(board, gen)
              
        render(screen,board)

def new_board():
    board = [[[False, [0, 0, 0]] for x in range(NX)] for y in range(NY)]
    return board

def update(board, gen):
    n_board = new_board()
    for y in range(NY):
        for x in range(NX):
            voisins,color = process_cell(board, x, y)
            if voisins == 3:
                n_board[y][x] = [True,color]
            elif voisins == 2:
                n_board[y][x] = board[y][x]
                if n_board[y][x][0]:
                    n_board[y][x][1] = color
            else:
                #Cellule morte
                pass
    gen += 1
    print("Generation {}".format(gen))
    return n_board, gen
            
                

def process_cell(board, x, y):
    voisins = 0
    color = [0, 0, 0]
    
    for i in range(9):
        if i == 4: continue 
        xx = (i % 3) - 1
        yy = (i // 3) - 1

        cx = (x + xx + NX) % NX
        
        cy = (y + yy + NY) % NY
        if board[cy][cx][0]:
            voisins += 1
            if  (board[cy][cx][1][0]>board[cy][cx][1][1]) and (board[cy][cx][1][0]>board[cy][cx][1][2]):
                color[0] += ((board[cy][cx][1][0]+1) / 2)
                color[1] -= ((board[cy][cx][1][1]+1) / 8)  
                color[2] -= ((board[cy][cx][1][2]+1) / 8) 
                
            elif  (board[cy][cx][1][1]>board[cy][cx][1][0]) and (board[cy][cx][1][1]>board[cy][cx][1][2]):
                color[0] -= ((board[cy][cx][1][0]+1) / 8)
                color[1] += ((board[cy][cx][1][1]+1) / 2)  
                color[2] -= ((board[cy][cx][1][2]+1) / 8)  
                
            elif  (board[cy][cx][1][2]>board[cy][cx][1][1]) and (board[cy][cx][1][2]>board[cy][cx][1][0]):
                color[0] -= ((board[cy][cx][1][0]+1) / 8) 
                color[1] -= ((board[cy][cx][1][1]+1) / 8)  
                color[2] += ((board[cy][cx][1][2]+1) / 2)  
                
            else:
                color[0] += ((board[cy][cx][1][0]+1) / 2) 
                color[1] += ((board[cy][cx][1][1]+1) / 2)  
                color[2] += ((board[cy][cx][1][2]+1) / 2)  
    
    for i in range(3):
        color[i] = int(math.floor(color[i]))
        if color[i] < 0:
            color[i] = 0
        if color[i] > 255:
            color[i] = 255
    if color == [0, 0, 0]:
        board[cy][cx][0] = False
        
             
    return voisins,color
        

    
def render(screen, board):
    screen.fill(BLACK)
    for y in range(NY):
        for x in range(NX):
            #print(board[y][x])
            if board[y][x][0]:
                pygame.draw.rect(screen, board[y][x][1], (x * SIZE, y*SIZE, SIZE, SIZE))
    pygame.display.update()

if __name__ == "__main__":
    main()


 
