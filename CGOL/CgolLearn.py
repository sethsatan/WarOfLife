import gym
import random
import tflearn as learn
import tensorflow as tf
import numpy as np


    
def main():    
    with tf.name_scope("input"):
        print("input")
        NX = 60 
        NY = 60 
        RENDER = True
        n_step = 500
        px = NY*(NX//3)
        py_= (((NY*NX)//(5*5))*8)
        
    with tf.name_scope("model"):
        print("model")
        
        s0 = tf.placeholder(tf.float32, shape=[NY, NX//3], name="s0")
        s1 = tf.placeholder(tf.float32, shape=[NY, NX//3], name="s1")
        s2 = tf.placeholder(tf.float32, shape=[NY, NX//3], name="s2")
        
        _red_ = tf.placeholder(tf.float32, shape=[None], name="_red_")                  
        _green_ = tf.placeholder(tf.float32, shape=[None], name="_green_")
        _blue_ = tf.placeholder(tf.float32, shape=[None], name="_blue_")

        
        P_r = _red_*s0+(_red_*-1)*(s0*-1+1)
        P_g = _green_*s1+(_green_*-1)*(s1*-1+1)
        P_b = _blue_*s2+(_blue_*-1)*(s2*-1+1)

        Pool_red = tf.nn.avg_pool(tf.reshape(P_r,[1,NY,NX//3,1]),ksize=[1,NY,NX//3,1],strides=[1,1,1,1],padding="SAME")
        Pool_green = tf.nn.avg_pool(tf.reshape(P_g,[1,NY,NX//3,1]),ksize=[1,NY,NX//3,1],strides=[1,1,1,1],padding="SAME")
        Pool_blue = tf.nn.avg_pool(tf.reshape(P_b,[1,NY,NX//3,1]),ksize=[1,NY,NX//3,1],strides=[1,1,1,1],padding="SAME")
        Pool = tf.nn.avg_pool(tf.reshape(P_r+P_g+P_b,[1,NY,NX//3,1]),ksize=[1,NY,NX//3,1],strides=[1,1,1,1],padding="SAME")
        

        Loss_red = tf.nn.l2_loss(Pool-Pool_red,"Loss_red")
        Loss_green = tf.nn.l2_loss(Pool-Pool_green,"Loss_green")
        Loss_blue = tf.nn.l2_loss(Pool-Pool_blue,"Loss_blue")
        Loss_Pool = tf.nn.l2_loss(Pool,"Loss_Pool")
        

        Relu = tf.nn.relu(Loss_red)


        #y = tf.matmul(x,W) + b


       # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    with tf.name_scope("train"):
        print("train")
        
       # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    with tf.name_scope("init"):
        print("tnit")
        init = tf.global_variables_initializer()   
       
    with tf.name_scope("session"):
        print("session")
        
        sess = tf.Session() 

        sess.run(init)

        p = sess.run([Relu],feed_dict={s0: stat0, _red_: [red_point], s1: stat1, _green_: [green_point], s2: stat2, _blue_: [blue_point]})
        print(p)
          
        print("end")       

if __name__ == "__main__":
    main()                


