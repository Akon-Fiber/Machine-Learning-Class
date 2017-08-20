"""
Author: Rusty Mina (rusty.mina@ee.upd.edu.ph)
        University of the Philippines Diliman

This is a 0 or 1 classifier using the mnist dataset.

You can turn the code into a 2 class classifier by 
changing the values of 0 and/or 1 below in "Global Settings"

You can also tweak the learning_rate and num_iterations below 
to observe how gradient descent behaves with different values 
of learning rate and iterations

For more info about mnist dataset, refer to mnist.py just beside this script :>

This is a logistic regression exercise from the same exercise in deeplearning.ai by Andrew Ng
"""

# Global Settings
class1 = 0
class2 = 1
learning_rate = 0.1
num_iterations = 100

import numpy as np
import matplotlib.pyplot as plt
import mnist
from PIL import Image

def sigmoid(z):
    return 1/( 1 + np.exp(-z) )

def propagate( w, X, b, Y ):
    """
    Arguments: 
    w -- weights for the image of size (height * width * 3, 1)
    X -- image data reshaped to (height * width * 3 , number of examples)
    b -- bias
    Y -- ground truth labels (1 or 0)
    """

    # Number of training examples
    m = X.shape[1]

    # Feed Forward
    Y_hat = np.dot( w.T, X ) + b        # raw predictiom
    A = sigmoid( Y_hat )                # activation / predicted value
    cost = -(np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T)) / m 

    # Backward Propagation
    dz = A - Y                          # derivative of cost with respect to z
    dw = np.dot(X, dz.T) / m            # derivative of cost with respect to w
    db = np.sum( dz ) / m               # derivative of cost with respect to b 

    gradients = {
        "dw": dw,
        "db": db
    }

    return gradients, cost

def optimize( w, X, b, Y, num_iterations, learning_rate ):
    """
    Minimize the cost function using gradient descent
    """
    costs = []

    for i in range(num_iterations):

        # Compute for gradients and cost
        gradients, cost = propagate( w, X, b, Y )
        dw = gradients["dw"]
        db = gradients["db"]

        # Update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs for to plot later
        costs.append(cost)
        print("Iter: " + str(i+1) + " Costs " + str(cost[0][0]))

    return w, b, costs

def predict( w, b, X ):

    Y_hat = np.dot( w.T, X) + b
    A = sigmoid( Y_hat )
    A_len = A.shape[1]

    count = 0
    Y_pred = []
    A = A.reshape(A_len)
    for i in A:
        if (i <= 0.5):
            Y_pred.append(0)
        else:
            Y_pred.append(1)

    Y_pred = np.array(Y_pred).reshape(A_len, 1)
    print(Y_pred.shape)
    return Y_pred        

def load_mnist_dataset(train_test):

    # Training or Testing dataset?
    if (train_test == "train"):
        data_orig = mnist.load_dataset_image( mnist.mnist_folder_path + mnist.train_image_path )
        label_orig = mnist.load_label( "train" )
    else:
        data_orig = mnist.load_dataset_image( mnist.mnist_folder_path + mnist.test_image_path )
        label_orig = mnist.load_label( "test" )

    # Get only 0 and 1 dataset
    label = []
    data = []
    example_num = label_orig.shape[0]
    for i in range(example_num):
        if (label_orig[i] == class1):
            data.append( data_orig[i] )
            label.append( 0 )
        elif (label_orig[i] == class2):
            data.append( data_orig[i] )
            label.append(1)

    data = np.array(data)
    data = data/255
    data = data.T
    label = np.array(label)
    label = label.reshape(label.shape[0], 1)
    label = label.T
    return data, label

# Main

# Training Dataset
train_data, train_label = load_mnist_dataset("train")

# Testing Dataset
test_data, test_label = load_mnist_dataset("test")

# Initialize parameters weight and bias 
w = np.zeros( 28 * 28 ).reshape( 28 * 28 , 1)
b = 0

# Gradient Descent
print("Doing the magic aka training :D")
w, b, costs = optimize(w, train_data, b, train_label, num_iterations, learning_rate)

# Costs plot
cost_len = len(costs)
costs = np.array(costs).reshape(cost_len)
plt.plot(costs)
plt.ylabel("Cost")
plt.xlabel("iterations")
plt.title("Learning rate = " + str(learning_rate))
plt.show()

# Get predictions
Y_pred_train = predict(w, b, train_data)
Y_pred_test = predict(w, b, test_data)

# Get accuracy and conclusions
num_label_train = train_label.shape[1]
train_acc = 1 - np.sum( np.abs( Y_pred_train.T - train_label )) / num_label_train
print("Train Accuracy = " + str(train_acc * 100) + "%")

num_label_test = test_label.shape[1]
test_acc = 1 - np.sum( np.abs( Y_pred_test.T - test_label )) / num_label_test
print("Test Accuracy = " + str(test_acc * 100) + "%")

# Custom Dataset
custom = 1  # set to 0 to turn this off
while (custom):
    """
    Use only 28 * 28 - 8-bit image
    """
    image_file_name = input("Filename: ")
    image_file_name = image_file_name + ".jpeg"

    im = Image.open(image_file_name)
    im.load()
    data = np.asarray( im ).reshape( 28 * 28 , 1 ) / 255
    print(data.shape)

    helo = predict(w, b, data)
    print(helo)
