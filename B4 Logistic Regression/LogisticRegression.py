# Logistic Regression largely adapted from deeplearning.ai by Andrew Ng

import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

def load_image( path ):
    im = Image.open( path )
    im.load()
    data = np.asarray( im, dtype="uint8" )

    return data

def plot_image( data ):
    plt.imshow( data )
    plt.show()

def save_image( data, name ):
    im = Image.fromarray( data )
    im.save( name + ".jpeg" )

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
    A = sigmoid( y_hat )                # activation / predicted value
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
        costs = costs.append(cost)

# Main
foo = load_image("dataset/1.jpeg")
save_image(foo, "try")
print(foo.shape)
plot_image(foo)