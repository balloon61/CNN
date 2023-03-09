from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # compute the loss and the gradient
    
    # in this part, I use a for loop to calculate the loss and the gradient. First, I get the score function for every training data X[i]W
    # and second I want to prevent the exp(score) become too large, so I minus the score by its maximum value, by doing this all of the elements       # expoential in the function will all <= 1 and > 0 and then I calculate the possbility matrix, and then we can use the definition of the           # softmax function to calculate loss. The calculation of grad is use another for loop go through every class, and when the class is equal to       # the correct class, it have to minus 1, and then * X[i]
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        
        score = X[i].dot(W)
        shift_score = score - np.max(score)
        P_distribution = np.exp(shift_score)
        P_distribution = P_distribution / np.sum(P_distribution)
        #print(shift_score.shape, P_distribution[y[i]])
        loss = loss - np.log(P_distribution[y[i]])
        
        for j in range(num_classes):
            if j != y[i]:
                dW[:, j] = dW[:, j] + P_distribution[j] * X[i]
            else:
                dW[:, j] = dW[:, j] + (P_distribution[j] - 1) * X[i]


    # pass
    
    loss = loss / num_train + reg * np.sum(W * W) 
    dW = dW / num_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # in this part, I calculate the loss and the gradient without a for loop. First, I get the score function for every training data XW
    # second I want to prevent the exp(score) become too large, so I minus the score by its maximum value, by doing this all of the elements           # expoential in the function will all <= 1 and > 0 and then I calculate the possibility matrix, calculate the possibitlty matrix is more           # difficult than using a for loop, I need to calculate the sum of every row of the matrix and divide it. and then we can use the definition of     # the softmax function to calculate loss (get every data's correct classes) divide by sum and use np.log. The calculation of grad can use          # possibility matrix got from previous calculation, and minus one at every correct classes, and then multiple by X.
    
    score = X.dot(W)
    # print(X.shape, W.shape, score.shape, y.shape)
    shift_score = score - np.max(score, axis=1)[:, np.newaxis]
    P_distribution = np.exp(shift_score)
    P = P_distribution / P_distribution.sum(axis=1, keepdims=True)

    sy = P[range(X.shape[0]), y]
   # total = np.log(sy) / np.sum(P)
    loss = -np.sum(np.log(sy) / np.sum(P, axis=1))

    loss = loss / X.shape[0] + reg * np.sum(W * W) 
    # loss = loss - np.log(P_distribution[y[i]])
    # pass
    
    minus_one = np.zeros_like(P)

    minus_one[np.arange(y.shape[0]), y] = -1
    P = P + minus_one
    dW = X.T.dot(P)
    dW = dW / X.shape[0] + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
