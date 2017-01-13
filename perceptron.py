import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
for i in range(3):
    plt.imshow(np.reshape(mnist.train.images[i], (28,28)), cmap='Greys_r')  
    #plt.show()
def sigmoid(values):
    output = []
    for i in values:
	
	if i>=100: output.append(1)
	elif i <= (-100): 
		output.append(0)
	else: 
		output.append(1. / (1. + math.exp(-i)))
    return output
#print(sigmoid( [-10,-1,0,1,10] ))


def network_activations(w1,w2,x):
    # ... YOUR CODE CALCULATING THE ACTIVATIONS OF THE HIDDEN LAYER COMES HERE ...
    input_vec = np.zeros(len(x)+1)
    input_vec[0] =1 # add bais to the input vector
    input_vec[1:] = x
    a1 = np.dot(input_vec,np.transpose(w1)) 
    a1 =  sigmoid(a1)
 
    out_vec = np.zeros(len(a1)+1)
    out_vec[0] =1 # add bais to the input vector
    out_vec[1:] = a1
     # ... YOUR CODE CALCULATING THE ACTIVATIONS OF THE OUTPUT LAYER COMES HERE ...
    a2 = np.dot(out_vec,np.transpose(w2)) 
    a2 =  sigmoid(a2)
    return (a1,a2)



x = np.array([0.5,1.2,0.8])

# Note that the first column contains bias weights therefore, the number of columns in
# the w1 matrix is one more than the length of the vector, and the number of columns in 
# the w2 matrix is one more than the number of hidden units
#w1 = np.array( [ [ 0.5, 0.1, -0.2, 0.3 ], \
#                 [ -1.2, 0.2, 0.5, -0.8], ] )
#w2 = np.array( [ [ 0.1, 0.5, -0.3 ] ])
#print(network_activations(w1,w2,x))

	
def predict(w1,w2,x):
	hidden, output = network_activations(w1,w2,x);	
	return output.index(max(output))
    # ...YOUR CODE COMES HERE...


x1 = np.array([-0.5,1.2,-0.5])
x2 = np.array([0.5,0.3,0.2])

# Note that the first column contains bias weights therefore, the number of columns in
# the w1 matrix is one more than the length of the vector, and the number of columns in 
# the w2 matrix is one more than the number of hidden units
w1 = np.array( [ [  0.1,  0.1, -0.2,  0.3 ], \
                 [ -0.1,  0.2,  0.5, -0.8 ], ] )
w2 = np.array( [ [  0.1,  0.5, -0.3 ], \
                 [ -0.1, -0.2,  0.7 ] ])
#print(predict(w1,w2,x1))
#print(predict(w1,w2,x2))


def gradient_for_an_instance(w1,w2,x,y):
    #...YOUR CODE COMES HERE...
    hidden, output = network_activations(w1,w2,x);	
    input_vec = np.zeros((1,len(x)+1))
    hidden_out = np.zeros((1,w1.shape[0]+1))
    hidden_out[0] = [1]+ list(hidden)
    
    input_vec[0] = [1] + list(x)
    target_v= np.asarray(y)
    delta_k = np.zeros((1,len(y)))
    delta_k[0] = (target_v - output )

    w2_t = np.transpose(w2)
    #transpose w2_t
    delta_h = np.transpose(np.dot(w2_t,np.transpose(delta_k)))
    temp = delta_h[0][1:]  
    delta_h = np.zeros((1,delta_h.shape[1]-1))
    delta_h[0] = np.asarray(temp)* ((np.asarray( hidden))*(1-np.asarray( hidden)))


    grad_w2 = np.dot(np.transpose(delta_k),hidden_out)
    grad_w1 = np.dot(np.transpose(delta_h),input_vec)
    return (grad_w1, grad_w2)

MINIBATCH_SIZE   = 100
NUM_HIDDEN_UNITS = 100
NUM_ATTRIBUTES   = len(mnist.train.images[0])
NUM_CLASSES      = len(mnist.train.labels[0])
np.random.seed(42)
w1 = (np.random.random((NUM_HIDDEN_UNITS,NUM_ATTRIBUTES+1)))-0.5
w2 = (np.random.random((NUM_CLASSES,NUM_HIDDEN_UNITS+1)))-0.5
eps = 0.5

for iteration in range(10000):
    grad_w1all = np.zeros((NUM_HIDDEN_UNITS,NUM_ATTRIBUTES+1))
    grad_w2all = np.zeros((NUM_CLASSES,NUM_HIDDEN_UNITS+1))
    train_images, labels = mnist.train.next_batch(MINIBATCH_SIZE )
    for instance in range(MINIBATCH_SIZE):
        grad_w1,grad_w2 = gradient_for_an_instance( \
            w1,w2,train_images[instance],labels[instance])
        grad_w1all += grad_w1/MINIBATCH_SIZE
        grad_w2all += grad_w2/MINIBATCH_SIZE
    w1 += eps*grad_w1all
    w2 += eps*grad_w2all
    
    if iteration%100==0:
        print(iteration, " out of 10000 finished")
correct = 0
for instance in range(len(mnist.test.images)):
    if np.argmax(mnist.test.labels[instance]) == predict(w1,w2,mnist.test.images[instance]):
        correct += 1.
print " Accuracy of naive Neural network is"
print(correct/len(mnist.test.images))
########################## End of Naive  Multi layer Neural Network ########################



########################### Code For Neural Network  using tensor flow#############################
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
    initial =  tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

	# One hidden layer with 100 neurons
NUM_HIDDEN_NEURONS = 100

W1 = weight_variable([784,NUM_HIDDEN_NEURONS]) 
# weights of the connections between input layer and hidden layer

W2 = weight_variable([NUM_HIDDEN_NEURONS,NUM_HIDDEN_NEURONS])
# weights of the connections between hidden layer and output layer


W3 = weight_variable([NUM_HIDDEN_NEURONS,NUM_HIDDEN_NEURONS])

W4 = weight_variable([NUM_HIDDEN_NEURONS,10])

b1 = weight_variable([NUM_HIDDEN_NEURONS])
# bias weights of the hidden layer

b2 = weight_variable([NUM_HIDDEN_NEURONS])
b3 = weight_variable([NUM_HIDDEN_NEURONS])
b4 = weight_variable([10])
# bias weights of the output layer

a1 = tf.sigmoid(tf.matmul(x,W1) +b1)  # activations of the hidden layer
a2 = tf.sigmoid(tf.matmul(a1,W2)+b2)  # activations of the output layer
a3 = tf.sigmoid(tf.matmul(a2,W3)+b3)  # activations of the output layer
a_out = tf.sigmoid(tf.matmul(a3,W4)+b4)  # activations of the output layer

loss = tf.reduce_mean(tf.square(y - a_out))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(100000):
    batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

correct_prediction = tf.equal(tf.argmax(a_out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print "Accuracy using tensor flow is:"
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
