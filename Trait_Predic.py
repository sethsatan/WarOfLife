import tensorflow as tf
import numpy as np


    
def main():
    
    with tf.name_scope("input"):
        
        NX = 60 #abscisse [Multiple de 3]
        NY = 60 #ordonnée
        RENDER = True
        n_step = 500
        px = NY*(NX//3)
        py_= (((NY*NX)//(5*5))*8)
        
    with tf.name_scope("model"):
        
        
        s0 = tf.placeholder(tf.float32, shape=[NY, NX], name="s0")
        s1 = tf.placeholder(tf.float32, shape=[NY, NX], name="s1")
        s2 = tf.placeholder(tf.float32, shape=[NY, NX], name="s2")
        
        _red_ = tf.placeholder(tf.float32, shape=[None], name="_red_")                  
        _green_ = tf.placeholder(tf.float32, shape=[None], name="_green_")
        _blue_ = tf.placeholder(tf.float32, shape=[None], name="_blue_")

        #P les 1 => point 0=>-point P = point*grille + (point*-1)*((grille*-1)+1)
        
        P_r1 = _red_*s0
        P_r2 = _red_*-1
        P_r3 =  s0*-1
        P_r4 = P_r3+1
        P_r5 = P_r2*P_r4
        P_r = P_r1+P_r5
       


        
        #P_g = tf.add(tf.matmul(_green_,s1),tf.matmul((_blue_*-1),tf.add(tf.matmul(s1,[-1]),1)))
        #P_b = tf.add(tf.matmul(_blue_,s2),tf.matmul((_blue_*-1),tf.add(tf.matmul(s2,[-1]),1)))

        

        #Wr = tf.Variable(tf.zeros(), name="Wr")
        #br = tf.Variable(tf.zeros(), name="br")
        #Wg = tf.Variable(tf.zeros(), name="Wg")
        #bg = tf.Variable(tf.zeros(), name="bg")
        #Wb = tf.Variable(tf.zeros(), name="Wb")
        #bb = tf.Variable(tf.zeros(), name="bb")

        #y = tf.matmul(x,W) + b


        
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

   #with tf.name_scope("train"):
        
        
        #train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    with tf.name_scope("init"):
        
        init = tf.global_variables_initializer() # initialisation des variable    
       
    with tf.name_scope("session"):
        
        
        sess = tf.Session() # ouverture de la session
        sess.run(init)
        
        i=0
        with open("BdD_Test", "r") as f:
            
            for line in f.readlines():
                i+=1
                if i>3 :
                    break

                if i%3==1:
                    red_point,row = line.strip().split(":")
                    stat0,foo = row.split(";")
                    stat0 = stat0
                    
                    
                    print(stat0[0])
                    
                if i%3==2:
                    green_point,row = line.strip().split(":")
                    stat1,foo = row.split(";")
                    stat1 = stat1.replace(" ",".,").replace("][",".][").replace("]]",".]]")
                    
                if i%3==0:
                    blue_point,row = line.strip().split(":")
                    stat2,foo = row.split(";")
                    stat2 = stat2.replace(" ",".,").replace("][",".][").replace("]]",".]]")

                    p = sess.run([P_r],feed_dict={s0: stat0, _red_: [red_point],s1: stat1, _green_: [green_point],s2: stat2, _blue_: [blue_point]})
                    #print(p)
            
        
    print("end")   
        

if __name__ == "__main__":
    main()                


