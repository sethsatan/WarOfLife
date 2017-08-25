import tensorflow as tf
import ChromaGameOfLifeForIA as cgol


def main():
    with tf.name_scope("input"):
        shape = (63, 21)
        S1 = tf.Variable(tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32),name='S1')
        S2 = tf.Variable(tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32),name='S2')
        S3 = tf.Variable(tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32),name='S3')

        RS1 = tf.placeholder(tf.float32, shape=(None))
        RS2 = tf.placeholder(tf.float32, shape=(None))
        RS3 = tf.placeholder(tf.float32, shape=(None))
        
  
    with tf.name_scope("session"):
        sess = tf.Session() 
        _S1 = sess.run(S1)
        _S2 = sess.run(S2)
        _S3= sess.run(S3)
        matriceFT = formCGOL(_S1,_S2,_S3)
        _RS1,_RS2,_RS3 = cgol.main(matriceFT)
        
        
        
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
