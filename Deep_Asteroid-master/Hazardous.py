

import os
import tensorflow as tf
import numpy as np 
import sys
from tensorflow.python.platform import gfile
from matplotlib import pyplot as plt

def label_encode(label):

    val=[]
    
    if label=='TRUE':
        val = [1,0]
    if label=='FALSE':
        val= [0,1]
     

    return val

def data_encode(file):


    X = []
    Y = []       

    train_file = open(file,'r')
    for line in train_file.read().split('\n'):
        if(len(line)>0):
                data_list=[]
                data=line.split(',')
                for i in range(len(data)-1):

                   data_list.append(data[i])
                X.append(data_list)
                
                label = data[31]
                Y.append(label_encode(label))    

            

    return X,Y

        
        

file = "nasa.csv"
test_file = "nasa.csv"

train_X,train_Y = data_encode(file)
test_X,test_Y = data_encode(test_file)

train_X = np.array(train_X)
train_Y = np.array(train_Y)

print(train_Y)
print(train_X.shape,train_Y.shape)

print(train_X[0])
print(train_Y[0])

## Learning Parameters

learning_rate = 0.01

training_epochs = 1000

display_steps = 50

n_input = 31



n_hidden_1 = 35
n_hidden_2 = 20
n_hidden_3 = 8
 
n_output = 2

# Placeholders

X = tf.placeholder("float",[None,n_input])
Y = tf.placeholder("float",[None,n_output])

weights ={

   "hidden_1":tf.Variable(tf.random_normal([n_input,n_hidden_1]),name="Weight_Hidden_1"),
   "hidden_2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]),name="Weight_Hidden_2"),
   "hidden_3":tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3]),name="Weight_Hidden_3"),
   "output":tf.Variable(tf.random_normal([n_hidden_3,n_output]),name="Weight_Output")


}

bias ={

   "hidden_1":tf.Variable(tf.random_normal([n_hidden_1]),name="Bias_Hidden_1"),
   "hidden_2":tf.Variable(tf.random_normal([n_hidden_2]),name="Bias_Hidden_2"),
   "hidden_3":tf.Variable(tf.random_normal([n_hidden_3]),name="Bias_Hidden_3"),
   "output":tf.Variable(tf.random_normal([n_output]),name="Bias_Output")


}


def model(X,weights,bias):

    layer1 = tf.add(tf.matmul(X,weights["hidden_1"]),bias["hidden_1"])
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.add(tf.matmul(layer1,weights["hidden_2"]),bias["hidden_2"])
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.add(tf.matmul(layer1,weights["hidden_3"]),bias["hidden_3"])
    layer1 = tf.nn.relu(layer1)


    output_layer = tf.matmul(layer1,weights["output"])+bias["output"]

    return output_layer

pred = model(X,weights,bias)

out = tf.identity(pred,name="output")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

epochs_list = []
loss_history = []


with tf.Session() as sess:

    sess.run(init)
    

    
    for epochs in range(training_epochs):

        _,c = sess.run([optimizer,loss],feed_dict={X:train_X,Y:train_Y})
        if(epochs+1)%display_steps==0:

            print("Epochs: ",epochs+1," Loss: ",c)
            loss_history.append(c)
            epochs_list.append(epochs)

    print("Optimisation finished")
    test_result = sess.run(pred,feed_dict={X:train_X})
    corrected_pred = tf.equal(tf.argmax(test_result,1),tf.argmax(train_Y,1))
    accuracy = tf.reduce_mean(tf.cast(corrected_pred,"float"))


    print("Accuracy: ",accuracy.eval({X:test_X, Y: test_Y})  ) 

    save_path = saver.save(sess,"model/heart.ckpt")
    print("Save to path:", save_path)


plt.figure()
plt.plot(epochs_list,loss_history)
plt.show()


         

    

            


     

    


    
