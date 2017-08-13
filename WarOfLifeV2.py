import tensorflow as tf
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import numpy as np

def main():
    with tf.name_scope("input"):
        NX = 9 #Multiple of 3
        NY = 9
        
        board = tf.Variable(tf.zeros([1,NY,NX,1],tf.float32))
        c_board = tf.Variable(tf.zeros([1,NY,NX,1],tf.float32))
        
        state0 = tf.Variable(tf.random_uniform([NY,NX//3], minval=0, maxval=2, dtype=tf.int32), name="state000")
        state1 = tf.Variable(tf.random_uniform([NY,NX//3], minval=0, maxval=2, dtype=tf.int32), name="state001")
        state2 = tf.Variable(tf.random_uniform([NY,NX//3], minval=0, maxval=2, dtype=tf.int32), name="state002")

        c_state0 = state0 * 1
        c_state1 = state1 * 2
        c_state2 = state2 * 3
        
        pre_state = tf.concat([state0,state1,state2],1)
        c_pre_state = tf.concat([c_state0,c_state1,c_state2],1)
    
        
            # black/null = 4 
            # red = 1
            # greed = 2
            # blue = 3
            
    with tf.name_scope("CGL"):
        
        state = tf.cast(tf.reshape(pre_state,[1,NY,NX,1]), tf.float32)
        c_state = tf.cast(tf.reshape(c_pre_state,[1,NY,NX,1]), tf.float32)
        kernel = tf.reshape(tf.ones([3,3]), [3,3,1,1])
        neighbours = tf.nn.conv2d(board, kernel, [1,1,1,1], "SAME") - board
        r_pre_neighbours = tf.cast(tf.equal(c_board, 1),tf.float32)
        g_pre_neighbours = tf.cast(tf.equal(c_board, 2),tf.float32)
        b_pre_neighbours = tf.cast(tf.equal(c_board, 3),tf.float32)
        n_pre_neighbours = tf.cast(tf.equal(c_board, 4),tf.float32)
        r_neighbours = tf.nn.conv2d(r_pre_neighbours, kernel, [1,1,1,1], "SAME") - r_pre_neighbours
        g_neighbours = tf.nn.conv2d(g_pre_neighbours, kernel, [1,1,1,1], "SAME") - g_pre_neighbours
        b_neighbours = tf.nn.conv2d(b_pre_neighbours, kernel, [1,1,1,1], "SAME") - b_pre_neighbours
        n_neighbours = tf.nn.conv2d(n_pre_neighbours, kernel, [1,1,1,1], "SAME") - n_pre_neighbours
        survive = tf.logical_and( tf.equal(board, 1), tf.equal(neighbours, 2))
        r_survive = tf.cast(tf.logical_and( survive , tf.equal(c_board, 1)),tf.int32)*1
        g_survive = tf.cast(tf.logical_and( survive , tf.equal(c_board, 2)),tf.int32)*2
        b_survive = tf.cast(tf.logical_and( survive , tf.equal(c_board, 3)),tf.int32)*3
        n_survive = tf.cast(tf.logical_and( survive , tf.equal(c_board, 4)),tf.int32)*4
        c_survive = tf.add(tf.add(r_survive,g_survive),tf.add(b_survive,n_survive))
                
        born = tf.equal(neighbours, 3)
        newstate = tf.cast(tf.logical_or(survive, born), tf.float32)
       
        init = tf.global_variables_initializer()
        fig = plt.figure()
        
    with tf.name_scope("session"):
        sess = tf.Session()   
        sess.run(init)
        sess.run(newstate)
        sess.run(tf.assign(board,state))
        sess.run(tf.assign(c_board,c_state))
        print(sess.run(c_survive))
        
        newstate_ = sess.run(tf.reshape(newstate, [NY,NX]))
        sess.run(tf.assign(board, newstate))
        plot = plt.imshow(newstate_, cmap='Greys', interpolation='nearest')

       
        
    def animateFn(num, sess, state, newstate):
        sess.run(newstate)
        newstate_ = sess.run(tf.reshape(newstate, [NY,NX]))
        sess.run(tf.assign(board, newstate))
        plot.set_array(newstate_)
        return plot
	
   
    ani = animation.FuncAnimation(fig, animateFn, 5, fargs=(sess, board, newstate), interval=2, blit=False)
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
