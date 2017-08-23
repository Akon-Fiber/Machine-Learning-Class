"""
Author: Rusty Mina (rusty.mina@eee.upd.edu.ph)
        University of the Philippines Diliman

This is a script containing funtions that can load and save images and labels from mnist dataset
Before running this script, download the dataset from http://yann.lecun.com/exdb/mnist/
and extract to mnist_extract folder, see the folder structure below for reference 

Folder structure 

root
    mnist.py
    mnist_download
        t10k-images-idx3-ubyte.gz
        t10k-labels-idx1-ubyte.gz
        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz 
    mnist_extract
        t10k-images-idx3-ubyte
        t10k-labels-idx1-ubyte
        train-images-idx3-ubyte
        train-labels-idx1-ubyte
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
    mnist_unsorted
        test
           *.jpg
        train
           *.jpg

"""

import numpy as np
import struct
import os
import urllib.request as ur
import gzip
from PIL import Image

# Folders
mnist_unsorted_folder = "mnist_unsorted/"
mnist_sorted_folder = "mnist_sorted/"
mnist_download_folder = "mnist_download/"
mnist_extract_folder = "mnist_extract/"

# External file info
mnist_url = "http://yann.lecun.com/exdb/mnist/"

# Filenames
mnist_filenames =   {   
                        "test_image" : "t10k-images-idx3-ubyte", 
                        "test_label": "t10k-labels-idx1-ubyte", 
                        "train_image" : "train-images-idx3-ubyte", 
                        "train_label": "train-labels-idx1-ubyte" 
                    }
file_bytes =        {   
                        "test_image" : 1648877, 
                        "test_label": 4542, 
                        "train_image" : 9912422, 
                        "train_label": 28881 
                    }
            

def mnist_download( filename, expected_bytes ):
    """
    Download a file if not present, and make sure it's the right size.
    Files are stored in mnist_download
    """
    filename = filename + ".gz"
    filepath = mnist_download_folder + filename

    if not os.path.exists( mnist_download_folder ):
        os.makedirs( mnist_download_folder )

    if not os.path.exists( filepath ):
        print( "Downloading ", filename, " ..." )
        file_download = ur.URLopener()
        file_download.retrieve( mnist_url + filename, filepath )
        statinfo = os.stat( filepath )
        if statinfo.st_size == expected_bytes:
            print( "Found and verified", filepath )
        else:
            raise Exception( "Failed to verify " +
                            filename + ". Can you get to it with a browser? \nDownload .gz files from http://yann.lecun.com/exdb/mnist/ and store in mnist_download folder" )
    else:
        print( "Found and verified", filepath )

    return filepath

def mnist_extract( filename ):
    """
    Extract the contents of gzip
    Extracted files are stored in mnist_extract
    """
    filename = filename + ".gz"
    filepath = mnist_download_folder + filename

    if not os.path.exists( filepath ):
        print("[ERROR ERROR] Please run mnist_download() first")
    else:

        if not os.path.exists( mnist_extract_folder ):
            os.makedirs( mnist_extract_folder )

        # fts stands for File To Store
        fts = filename[ :-3 ]   # remove ".gz"    

        print("Extracting", filename, "to", mnist_extract_folder + fts )

        with gzip.open( filepath, 'rb' ) as f:
            file_content = f.read()

        with open( mnist_extract_folder + fts, 'wb' ) as f2:
            f2.write( file_content )

def load_mnist_extract( filename ):
    """
    Load image pixel values from binary and store as numpy array of shape (examples, row_size * col_size)
    Images are greyscale (8 Bit Depth)
    """

    print("Loading MNIST Dataset")

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

    print("Done loading MNIST dataset")
    return parsed

def mnist_unsorted( test_train ):
    """
    Extract images and store them in mnist_unsorted unsorted
    See the folder structure above for reference

    Save numpy arrays as greyscale image of shape (28,28,1)
    To save numpy arrays as RGB, use convert('RGB') instead of convert('L) 
    """

    if (test_train == "train"):
        data = load_mnist_extract( mnist_extract_folder + mnist_filenames["train_image"] )
        output_folder = mnist_unsorted_folder + "train/"
    else:
        data = load_mnist_extract( mnist_extract_folder + mnist_filenames["test_image"] )
        output_folder = mnist_unsorted_folder + "test/"

    if not os.path.exists( output_folder ):
        os.makedirs( output_folder )

    print("Saving unsorted images. This may take a while...")

    data_shape = data.shape
    for i in range( data_shape[0] ):
        name = str(i + 1)
        foo = data[i].reshape( 28, 28 )
        im = Image.fromarray( foo ).convert('L')
        im.save( output_folder + name + ".jpeg" )

    print("Done saving images")

def mnist_sorted( train_test ):
    """
    Extract images and store them in mnist_sorted where folder name is the label
    See the folder structure above for reference
        
    Save numpy arrays as greyscale image of shape (28,28,1)
    To save numpy arrays as RGB, use convert('RGB') instead of convert('L) 
    """

    if (train_test == "train"):
        label_value = load_mnist_label("train")
        image_data = load_mnist_extract( mnist_extract_folder + mnist_filenames["train_image"] )
    else:
        label_value = load_mnist_label("test")
        image_data = load_mnist_extract( mnist_extract_folder + mnist_filenames["test_image"] )
    
    print("Saving sorted images. This may take a while...")

    label_value = label_value.reshape( label_value.shape[0] )

    for i in range(10):

        label_name = str( i )
        current_folder  = mnist_sorted_folder + train_test + "/" + label_name
        if not os.path.exists( current_folder ):
            os.makedirs( current_folder )
        
        print( "Extracting images of " + label_name )

        locations = np.where(label_value == i )[0]

        count = 0
        for i in locations:
            name = str( count + 1 )
            poo = image_data[i].reshape( 28,28 )
            im = Image.fromarray( poo ).convert('L')
            im.save( current_folder + "/" + name + ".jpeg" )
            count += 1

def load_mnist_label(train_test):

    print("Loading labels")
    if (train_test == "train"):
        filename = mnist_extract_folder + mnist_filenames["train_label"]
    else:
        filename = mnist_extract_folder + mnist_filenames["test_label"]

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

def mnist_download_and_extract():
    """
    Download and extract mnist dataset found in http://yann.lecun.com/exdb/mnist/
    """

    print("Downloading and extracting MNIST Dataset. This may take a while...")
    count = 0
    for key in mnist_filenames:
        mnist_download( mnist_filenames[key], file_bytes[key] )
        mnist_extract( mnist_filenames[key] )
        count += 1

    print("Done downloading and extracting MNIST Dataset")

def mnist_all():

    print("Preparing MNIST Dataset")

    # Download and extract MNIST dataset
    mnist_download_and_extract()

    # Unsorted Training Images
    print("Unsorted Training Images")
    mnist_unsorted("train")

    # Unsorted Test Images
    print("Unsorted Test Images")
    mnist_unsorted("test")

    # Sorted Training Images
    print("Sorted Training Images")
    mnist_sorted("train")

    # Sorted Test Images
    print("Sorted Test Images")
    mnist_sorted("test")

    # Load train labels
    train_label_values = load_mnist_label( "train" )
    print(train_label_values.shape)

    # Load test labels
    test_label_values = load_mnist_label( "test" )
    print(test_label_values.shape)

    print("Done preparing MNIST Dataset")

"""
mnist_all()
"""