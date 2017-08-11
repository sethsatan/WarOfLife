import tensorflow as tf
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import numpy as np

def main():
    with tf.name_scope("input"):
        NX = 27 #Multiple of 3
        NY = 27
        
        state000 = tf.Variable(tf.random_uniform([NY,NX//3], minval=0, maxval=2, dtype=tf.int32), name="state000")
        state001 = tf.Variable(tf.random_uniform([NY,NX//3], minval=0, maxval=2, dtype=tf.int32), name="state001")
        state002 = tf.Variable(tf.random_uniform([NY,NX//3], minval=0, maxval=2, dtype=tf.int32), name="state002")
        
        prestate = tf.concat([state000,state001,state002],1)
        
    
       
        #board = tf.placeholder(tf.zero([NY,NX,1]))
        #colorimetrie = tf.placeholder(tf.zero([NY,NX,1]))
            # white = 0 
            # red = 1
            # greed = 2
            # blue = 3
    with tf.name_scope("CGL"):
        
        state = tf.cast(tf.reshape(prestate,[1,NY,NX,1]), tf.float32)
        kernel = tf.reshape(tf.ones([3,3]), [3,3,1,1])
        neighbours = tf.nn.conv2d(state, kernel, [1,1,1,1], "SAME") - state
        survive = tf.logical_and( tf.equal(state, 1), tf.equal(neighbours, 2))
        born = tf.equal(neighbours, 3)
        newstate = tf.cast(tf.logical_or(survive, born), tf.float32)
       
        init = tf.global_variables_initializer()
        fig = plt.figure()
        
    with tf.name_scope("session"):
        sess = tf.Session()   
        sess.run(init)
        
        newstate_ = sess.run(tf.reshape(newstate, [NY,NX]))
        state = newstate
        sess.run(state)
        plot = plt.imshow(newstate_, cmap='Greys', interpolation='nearest')

       
        
    def animateFn(num, sess, state, newstate):
        sess.run(newstate)
        newstate_ = sess.run(tf.reshape(newstate, [NY,NX]))
        state = newstate
        sess.run(state)
        plot.set_array(newstate_)
        return plot
	
   
    ani = animation.FuncAnimation(fig, animateFn, 5, fargs=(sess, state, newstate), interval=2, blit=False)
    plt.show()
       
        
        

        
        



        
def  last_iteration():       
    last_iteration= open("last_iteration.txt","w")
    last_iteration.write(str(matriceFT))
    last_iteration.close

def formCGOL(_S1,_S2,_S3):
    board = cgol.new_board()
    for sy in range(63):
        for sx in range(21):
            
            if _S1[sy][sx] == 1:
                board [sy][sx] = [True,[255,0,0]]
            if _S2[sy][sx] == 1:
                board [sy][sx+21] = [True,[0,255,0]]
            if _S1[sy][sx] == 1:
                board [sy][sx+42] = [True,[0,0,255]]

    return board

if __name__ == "__main__":
    main()
