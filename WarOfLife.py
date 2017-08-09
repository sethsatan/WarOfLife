import tensorflow as tf
import ChromaGameOfLifeForIA as cgol


def main(): 
    shape = (63, 21)
    S1 = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)
    S2 = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)
    S3 = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

    with tf.Session() as sess:
        _S1 = sess.run(S1)
        _S2 = sess.run(S2)
        _S3= sess.run(S3)
        matriceFT = formCGOL(_S1,_S2,_S3)
        #RS1,RS2,RS3 = cgol.main(matriceFT)
        
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
