import pygame
from pygame.locals import *
from random import randint
import operator
import math

NX = 63
NY = 63


    
EmptyBoard = [[[False, [0, 0, 0]] for x in range(NX)] for y in range(NY)]
           
def new_board():
    board = [[[False, [0, 0, 0]] for x in range(NX)] for y in range(NY)]
    return board

def update(board, gen,P_red,P_green,P_blue):
    n_board = new_board()
    for y in range(NY):
        for x in range(NX):
            voisins,color = process_cell(board, x, y)
            if voisins == 3:
                n_board[y][x] = [True,color]
                if gen%100 == 0:
                    game_calc(color,P_red,P_green,P_blue)
            elif voisins == 2:
                n_board[y][x] = board[y][x]
                if n_board[y][x][0]:
                    n_board[y][x][1] = color
                    if gen%100 == 0 :
                        P_red,P_green,P_blue = game_calc(color,P_red,P_green,P_blue)
            else:
                pass
    #if gen%100 == 0:
    #    print(gen)        
    gen += 1
    return n_board,gen,P_red,P_green,P_blue

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
                color[0] += ((board[cy][cx][1][0]+1) >> 1)
                color[1] -= ((board[cy][cx][1][1]+1) >> 3)  
                color[2] -= ((board[cy][cx][1][2]+1) >> 3) 
                
            elif  (board[cy][cx][1][1]>board[cy][cx][1][0]) and (board[cy][cx][1][1]>board[cy][cx][1][2]):
                color[0] -= ((board[cy][cx][1][0]+1) >> 3)
                color[1] += ((board[cy][cx][1][1]+1) >> 1)  
                color[2] -= ((board[cy][cx][1][2]+1) >> 3)  
                
            elif  (board[cy][cx][1][2]>board[cy][cx][1][1]) and (board[cy][cx][1][2]>board[cy][cx][1][0]):
                color[0] -= ((board[cy][cx][1][0]+1) >> 3) 
                color[1] -= ((board[cy][cx][1][1]+1) >> 3)  
                color[2] += ((board[cy][cx][1][2]+1) >> 1)  
                
            else:
                color[0] += ((board[cy][cx][1][0]+1) >> 2) 
                color[1] += ((board[cy][cx][1][1]+1) >> 2)  
                color[2] += ((board[cy][cx][1][2]+1) >> 2)  
    
    for i in range(3):
        #color[i] = int(math.floor(color[i]))
        if color[i] < 0:
            color[i] = 0
        if color[i] > 255:
            color[i] = 255
    if color == [0, 0, 0]:
        board[cy][cx][0] = False
            
    return voisins,color

def game_calc(color,P_red,P_green,P_blue):
    
    P_red += color[0]
    P_green += color[1]
    P_blue += color[2]

    return P_red,P_green,P_blue

def main(matriceTF=EmptyBoard):
    P_red = 0
    P_green = 0
    P_blue = 0
    board = matriceTF
    gen = 0
    while True:
        board,gen,P_red,P_green,P_blue = update(board,gen,P_red,P_green,P_blue)
        if gen == 1000:
            break

    return P_red,P_green,P_blue

if __name__ == "__main__":
    main()


 
