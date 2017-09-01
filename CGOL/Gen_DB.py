import calc_point as cp
import random

NX=60
NY=60
GEN_Limite=100

last_iteration= open("BD","a")
for i in range(900):
    if i%20==0:print(i)
    state0=[[ random.randint(0, 1) for x in range(NX//3)] for y in range(NY)]
    state1=[[ random.randint(0, 1) for x in range(NX//3)] for y in range(NY)]
    state2=[[ random.randint(0, 1) for x in range(NX//3)] for y in range(NY)]

    red,green,blue,total=cp.main(state0,state1,state2,GEN_Limite,NX,NY)
    
    last_iteration.write(str(red)+":["+"".join(map(str, state0))+"];\n")
    last_iteration.write(str(green)+":["+"".join(map(str,state1))+"];\n")
    last_iteration.write(str(blue)+":["+"".join(map(str,state2))+"];\n")
    
    
last_iteration.close
print("end")
