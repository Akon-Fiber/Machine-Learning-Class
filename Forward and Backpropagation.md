# Steps for Forward Propagation and Back Propagation on m examples

## One-step Forward and Backward Propagation

Let:

* (x,y) be the input sample where x is the input and y is the ground truth
* Let W be the collective weights of w1, w2
* let b be the bias 
* alpha is the learning rate
* sigma(z) = (1+e^-z)^-1

1. Initialize the variables to any values:
```    
        J = 0
      dw1 = 0
      dw2 = 0
       db = 0
```    
2. Feed-forward Propagation
```
       z = Wx + b               ; raw output
       a = sigma(z)             ; predicted output
L = Loss = loss_func(a, y)      ; y is the ground truth
J = Cost += Loss                ; Compute all loss and add
```
3. Backward Propagation
```
       dz  += a - y             ; Compute derivatives and add
       dw1 += x1dz              ; Compute derivatives and add
       dw2 += x2dz              ; Compute derivatives and add
       db  += dz                ; Compute derivatives and add
```
4. Normalize computed values (iow. get the average)
```
        J = J/m
      dw1 = dw1/m
      dw2 = dw2/m
       db = db/m
```
5. Update weights and parameters
```
    w1 : w1 - (alpha)dw1
    w2 : w2 - (alpha)dw2
    b  : b  - (alpha)db
```

This is only for 1 iteration of Forward and Backward propagation, do more iterations until cost function J is minimized.

Note that this algorithm of minimizing the cost function can be done in 3-nest of loop
* one for the insides of feed-forward and backward propation
* another level on the whole process of one-step forward and backprop for the whole training set
* and the highest level being the minimization of cost function

This could be further optimized by using vectorization techniques.


I hope github implements LaTeX support for markdowns, *cough cough*