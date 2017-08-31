import numpy as np

def main(stat0,stat1,stat2,GEN_Limite=50,NX=60,NY=60):     

    c_state0 = [[[False,0] for x in range(NX//3)] for y in range(NY)] 
    c_state1 = [[[False,0] for x in range(NX//3)] for y in range(NY)]
    c_state2 = [[[False,0] for x in range(NX//3)] for y in range(NY)]
    
    for y in range(NY):
        for x in range(NX//3):
            if state0[y][x] == 1:
                c_state0[y][x] = [True,1]
            if state1[y][x] == 1:
                c_state1[y][x] = [True,2]
            if state2[y][x] == 1:   
                c_state2[y][x] = [True,3]

    board = np.concatenate((c_state0,c_state1,c_state2), axis=1)
    gen = 0
    while True:     
        n_board = [[[False, [0, 0, 0]] for x in range(NX)] for y in range(NY)]
        for y in range(NY):
            for x in range(NX):
                #voisins,color = process_cell(board, x, y)
                voisins = 0
                C_color = [0,0,0]
                for i in range(9):
                    xx = (i % 3) - 1
                    yy = (i // 3) - 1

                    cx = (x + xx + NX) % NX                    
                    cy = (y + yy + NY) % NY
                    if board[cy][cx][0]:
                        voisins += 1
                        if board[cy][cx][1]== 1:
                            C_color[0]+=1
                        elif board[cy][cx][1] == 2:
                            C_color[1]+=1
                        elif board[cy][cx][1] == 3:
                            C_color[2]+=1
                        elif board[cy][cx][1] == 4:
                            C_color[0]+=1
                            C_color[1]+=1
                            C_color[2]+=1
                        else:
                            print("color error")
                        

                
                                                 
                if voisins == 3:
                    if C_color[0]==C_color[1] and C_color[1]==C_color[2] and C_color[0]>0:
                        new_color = 4
                    elif C_color[0]>C_color[1] and C_color[0]>C_color[2] :
                        new_color = 1
                    elif C_color[1]>C_color[0] and C_color[1]>C_color[2] :
                        new_color = 2
                    elif C_color[2]>C_color[0] and C_color[2]>C_color[1] :
                        new_color = 3
                    else:
                        new_color = 0
                    n_board[y][x] = [True,new_color]
                elif voisins == 2:
                    n_board[y][x] = board[y][x]
                else:
                    #Cellule morte
                    pass
                
        board = n_board
        gen += 1
        if gen==GEN_Limite:
            break
    red_point = 0
    green_point = 0
    blue_point = 0
    total = 0
    for y in range(NY):
            for x in range(NX):
                if board[y][x][0]:
                    total+=1
                    if board[y][x][1]==1:
                        red_point+=1
                    elif board[y][x][1]==2:
                        green_point+=1
                    elif board[y][x][1]==3:
                        blue_point+=1
                    elif board[y][x][1]==4:
                        red_point+=1
                        green_point+=1
                        blue_point+=1
                    else:
                        print("End error")
                        
                    
    print(red_point,green_point,blue_point,total)            
    return red_point,green_point,blue_point
           
