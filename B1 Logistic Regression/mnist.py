"""
Author: Rusty Mina (rusty.mina@eee.upd.edu.ph)
        University of the Philippines Diliman

This is a script containing funtions that can load and save images and labels from mnist dataset
Before running this script, download the dataset from http://yann.lecun.com/exdb/mnist/
and extract to mnist_orig folder, see the folder structure below for reference 

Folder structure 

root
    mnist.py
    mnist_extract
        test_image
           *.jpg
        train_image
           *.jpg
    mnist_orig
        t10k-images.idx3-ubyte
        t10k-labels.idx1-ubyte
        train-images.idx3-ubyte
        train-labels.idx1-ubyte
    mnist_sorted
        test
            1
            .
            .
            9
        train
            1
            .
            .
            9            

"""
import numpy as np
import struct
from PIL import Image

# mnist folder info
mnist_folder_path = "mnist_orig/"
train_image_path = "train-images.idx3-ubyte"
train_label_path = "train-labels.idx1-ubyte"
test_image_path = "t10k-images.idx3-ubyte"
test_label_path = "t10k-labels.idx1-ubyte"

# Output Image Settings
output_train_path = "mnist_extract/train_image/"
output_test_path = "mnist_extract/test_image/"

def load_dataset_image( filename ):
    """
    Load image pixel values from binary and store as numpy array of shape (examples, row_size * col_size)
    Images are greyscale (8 Bit Depth)
    """

    print("Loading images")

    with open(filename, "rb") as f:

        # Extracts dataset info e.g. magic number, number of examples, row size, col size
        data = f.read(16)
        magic, examples, row, col = struct.unpack( '>IIII', data )
        print(magic, examples, row, col)

        # Extract pixel values
        image_size = row * col
        parsed = []
        for i in range( examples ):
            data = f.read( image_size )
            parsed.append( [pixel for pixel in data] )

        parsed = np.array(parsed)
        print(parsed.shape)

    print("Done loading images")
    return parsed

def save_dataset_image(data, test_train):
    """
    Save numpy arrays as greyscale image of shape (28,28,1)
    To save numpy arrays as RGB, use convert('RGB') instead of convert('L) 
    """

    print("Saving images. This may take a while")
    if (test_train == "train"):
        output_folder = output_train
    else:
        output_folder = output_test

    data_shape = data.shape
    for i in range( data_shape[0] ):
        name = str(i + 1)
        foo = data[i].reshape( 28, 28 )
        im = Image.fromarray( foo ).convert('L')
        im.save(output_folder + name + ".jpeg" )

    print("Done saving images")

def load_label(train_test):

    print("Loading labels")
    if (train_test == "train"):
        filename = mnist_folder_path + train_label_path
    else:
        filename = mnist_folder_path + test_label_path

    with open(filename, "rb") as f:

        # Extracts label dataset info e.g. magic number and number of examples
        data = f.read(8)
        magic, examples = struct.unpack( '>II', data )
        print(magic, examples)

        # Extract labels
        parsed = []
        data = f.read( examples )
        parsed = [i for i in data]
        parsed = np.array(parsed).reshape(examples, 1)
        
        return parsed

def save_dataset_sorted(train_test):
    """
    Extract images and store them in mnist_sorted where folder name is the label
    See the folder structure above for reference
    """

    if (train_test == "train"):
        label_value = load_label("train")
        image_data = load_dataset_image(mnist_folder_path + train_image_path)
    else:
        label_value = load_label("test")
        image_data = load_dataset_image(mnist_folder_path + test_image_path)

    label_value = label_value.reshape( label_value.shape[0] )

    for i in range(1):
        value = i
        label_name = str(value)
        print( "Extracting images of " + label_name )

        locations = np.where(label_value == value)[0]

        count = 0
        for i in locations:
            name = str(count + 1)
            poo = image_data[i].reshape( 28,28 )
            im = Image.fromarray( poo ).convert('L')
            im.save("mnist_sorted/" + train_test + "/" + label_name + "/" + name + ".jpeg" )
            count += 1

# Main
"""
# Unsorted Training Images
print("Train Images")
foo = load_dataset_image(mnist_folder_path + train_image_path)
save_dataset_image( foo, "train" )

# Unsorted Test Images
print("Test Images")
foo = load_dataset_image(mnist_folder_path + test_image_path)
save_dataset_image( foo, "test" )
"""
"""
# Sorted Training Images
save_dataset_sorted("train")

# sorted Test Images
save_dataset_sorted("test")
"""