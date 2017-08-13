import tensorflow as tf
import pygame
from pygame.locals import *
import numpy as np

def main():
    with tf.name_scope("input"):
        NX = 15 #abscisse [Multiple de 3]
        NY = 15 #ordonnée
        RENDER = True 
        
        
        board = tf.Variable(tf.zeros([1,NY,NX,1],tf.float32)) #plateau
        c_board = tf.Variable(tf.zeros([1,NY,NX,1],tf.float32)) # couleur sur le plateau
        
        state0 = tf.Variable(tf.random_uniform([NY,NX//3], minval=0, maxval=2, dtype=tf.int32), name="state000") #premier tier du plateau
        state1 = tf.Variable(tf.random_uniform([NY,NX//3], minval=0, maxval=2, dtype=tf.int32), name="state001") #deuxième tier du plateau
        state2 = tf.Variable(tf.random_uniform([NY,NX//3], minval=0, maxval=2, dtype=tf.int32), name="state002") #troisième tier di plateau

        c_state0 = state0 * 1 #premier tier du plateau en rouge
        c_state1 = state1 * 2 #deuxième tier du plateau en vert
        c_state2 = state2 * 3 #troisième tier di plateau en bleu 
        
        pre_state = tf.concat([state0,state1,state2],1) # assemblage du premier etat
        c_pre_state = tf.concat([c_state0,c_state1,c_state2],1) # assemblage du premier etat des couleurs
    
        
            # null = 4 
            # red = 1
            # greed = 2
            # blue = 3
            
    with tf.name_scope("CGL"):
        
        state = tf.cast(tf.reshape(pre_state,[1,NY,NX,1]), tf.float32) # formatage du premier état
        c_state = tf.cast(tf.reshape(c_pre_state,[1,NY,NX,1]), tf.float32) # formatage du premier état des couleurs
        kernel = tf.reshape(tf.ones([3,3]), [3,3,1,1]) # filtre 3 par 3
        neighbours = tf.nn.conv2d(board, kernel, [1,1,1,1], "SAME") - board # carte des voisinages
        r_pre_neighbours = tf.cast(tf.equal(c_board, 1),tf.float32) # masque rouge
        g_pre_neighbours = tf.cast(tf.equal(c_board, 2),tf.float32) # masque vert
        b_pre_neighbours = tf.cast(tf.equal(c_board, 3),tf.float32) # masque bleu
        n_pre_neighbours = tf.cast(tf.equal(c_board, 4),tf.float32) # masque null
        r_neighbours = tf.nn.conv2d(r_pre_neighbours, kernel, [1,1,1,1], "SAME") - r_pre_neighbours # carte des voisinages rouge
        g_neighbours = tf.nn.conv2d(g_pre_neighbours, kernel, [1,1,1,1], "SAME") - g_pre_neighbours # carte des voisinages vert
        b_neighbours = tf.nn.conv2d(b_pre_neighbours, kernel, [1,1,1,1], "SAME") - b_pre_neighbours # carte des voisinages bleu
        n_neighbours = tf.nn.conv2d(n_pre_neighbours, kernel, [1,1,1,1], "SAME") - n_pre_neighbours # carte des voisinages null
        survive = tf.logical_and( tf.equal(board, 1), tf.equal(neighbours, 2)) #condition de survie
        r_survive = tf.cast(tf.logical_and( survive , tf.equal(c_board, 1)),tf.int32)*1 #condition de survie rouge
        g_survive = tf.cast(tf.logical_and( survive , tf.equal(c_board, 2)),tf.int32)*2 #condition de survie vert 
        b_survive = tf.cast(tf.logical_and( survive , tf.equal(c_board, 3)),tf.int32)*3 #condition de survie bleu 
        n_survive = tf.cast(tf.logical_and( survive , tf.equal(c_board, 4)),tf.int32)*4 #condition de survie null 
        c_survive = tf.add(tf.add(r_survive,g_survive),tf.add(b_survive,n_survive)) #condition de survie des couleurs
        born = tf.equal(neighbours, 3) #condition de naissance
        r_born = tf.cast(tf.logical_and( born , tf.logical_or(tf.equal(r_neighbours,2),tf.equal(r_neighbours,3))),tf.int32)*1 #condition de naissance rouge
        g_born = tf.cast(tf.logical_and( born , tf.logical_or(tf.equal(g_neighbours,2),tf.equal(g_neighbours,3))),tf.int32)*2 #condition de naissance vert
        b_born = tf.cast(tf.logical_and( born , tf.logical_or(tf.equal(b_neighbours,2),tf.equal(b_neighbours,3))),tf.int32)*3 #condition de naissance bleu
        n_born = tf.cast(tf.logical_and( born , tf.logical_and(tf.logical_and(tf.equal(r_neighbours,1),tf.equal(g_neighbours,1)),tf.equal(b_neighbours,1))),tf.int32)*4 #condition de naissance null
        c_born = tf.add(tf.add(r_born,g_born),tf.add(b_born,n_born)) #condition de naissance des couleurs
        newstate = tf.cast(tf.logical_or(survive, born), tf.float32) # nouvel état
        c_newstate = tf.cast(tf.add(c_born,c_survive), tf.float32) # nouvel état des couleurs
        
       
        init = tf.global_variables_initializer() # initialisation des variable
        
        
    with tf.name_scope("session"):
        sess = tf.Session() # ouverture de la session 
        sess.run(init) # initilisation des vatiable de la session
        sess.run(tf.assign(board,state)) # assignation du premier état au plateau
        sess.run(tf.assign(c_board,c_state)) # assignation du premier état au plateau des couleurs
        sess.run(newstate) # génération du deuxème état
        sess.run(c_newstate) # génération du deuxème état des couleurs
        
        newstate_ = sess.run(tf.reshape(newstate, [NY,NX])) #mise en forma 2d du deuxième état
        c_newstate_ = sess.run(tf.reshape(c_newstate, [NY,NX])) #mise en forma 2d du deuxième état des couleus
        sess.run(tf.assign(board, newstate)) #assignation du deuxième état au plateau 
        sess.run(tf.assign(c_board, c_newstate)) #assignation du deuxième état au plateau des couleurs
        #génération du visuel
       
    def animateFn(): #fonction de génération d'état
        sess.run(newstate)
        sess.run(c_newstate)
        newstate_ = sess.run(tf.reshape(tf.cast(newstate,tf.int32), [NY,NX]))
        c_newstate_ = sess.run(tf.reshape(tf.cast(c_newstate,tf.int32), [NY,NX]))
        sess.run(tf.assign(board, newstate))
        sess.run(tf.assign(c_board, c_newstate))
        print(sess.run(tf.reshape(tf.cast(tf.add(r_born,r_survive),tf.int32), [NY,NX])))
        return c_newstate_
    
    with tf.name_scope("loop"):
        while True:
            if (RENDER == True):
                SIZE = 16
                WIDTH = NX * SIZE 
                HEIGHT = NY * SIZE
                BLACK = (0, 0, 0)
                GPS = 5
                PAUSE = True
                pygame.init()
                pygame.display.set_caption("Conway")
                screen = pygame.display.set_mode((WIDTH,HEIGHT))
                fps_clock = pygame.time.Clock()
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        quit()

                    if event.type == KEYDOWN:
                        if event.key == K_SPACE:
                            PAUSE = not PAUSE

                if not PAUSE:
                    fps_clock.tick(GPS)
                    view = animateFn()
                    
                    screen.fill(BLACK)
                    for y in range(NY):
                        for x in range(NX):
                            
                            if view [y][x] == 0 :
                                pass
                            elif view [y][x] == 1:
                                pygame.draw.rect(screen, [255,0,0], (x * SIZE, y*SIZE, SIZE, SIZE))
                            elif view [y][x] == 2:
                                pygame.draw.rect(screen, [0,255,0], (x * SIZE, y*SIZE, SIZE, SIZE))
                            elif view [y][x] == 3:
                                pygame.draw.rect(screen, [0,0,255], (x * SIZE, y*SIZE, SIZE, SIZE))
                            elif view [y][x] == 4:
                                pygame.draw.rect(screen, [255,255,255], (x * SIZE, y*SIZE, SIZE, SIZE))
                    pygame.display.update() 
                
            else:
                animateFn()
                
if __name__ == "__main__":
    main()
