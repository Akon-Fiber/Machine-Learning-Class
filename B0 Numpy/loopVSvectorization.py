import numpy as np
import time

# Let function be
# z = Wx + b

# Initialize W, x and b
W = np.random.rand(1000000)
x = np.random.rand(1000000)
b = 1

# Vectorization approach
tic = time.time()
z = np.dot(W,x) + b     # z = Wx + b
toc = time.time()

print("z = " + str(z))
print("Vectorization approach: " + str(1000*(toc - tic)) +"ms")

# Explicit loop approach
tic = time.time()
z = 0
for i in range(1000000):
    z += W[i] * x[i]
z += b
toc = time.time()

print("z = " + str(z))
print("Explicit loop approach: " + str(1000*(toc - tic)) +"ms")
