from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
# compute the loss and the gradient

    
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    
    
    # This part is trying to use the two for loop to calculate the dW and tbe loss
    # Input: W, X, y
    # Outpur: loss, dW

    # The outer for loop will go through all training data I have.
    for i in range(num_train):
        # calculate the score from X and W, y[i] means the correct class, so scores[y[i]] means the scores of the correct course
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        
        # the inner loop will go through every classes, calculate the margin by eq from svm: max(0, xiwj -xiwyi + delta)
        # and we know that when j equal to the correct class y[i] dLi/dwj = (xi * wj - xi * wj + delta > 0) * xi, 
        for j in range(num_classes):
            if j == y[i]:
                continue
            # 
            margin = scores[j] - correct_class_score + 1  # note delta = 1
        
            # dLi/dwy[i] = - (xi * wj - xi * wj + delta > 0) * xi and if it's not equal to y[i]
            # the equation will become sigma for all j not equal to y[i]: - (xi * wj - xi * wj + delta > 0) * xi
            if margin > 0:
                dW[:, j] = dW[:, j] + X[i].T
                dW[:, y[i]] = dW[:, y[i]] - X[i].T

    # because I consider every data (num train), so I have to divide by num train to get the average, and the add reg to prevent it fit too well.
    dW = dW / num_train + W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    
    Structured SVM loss function.

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
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    # This part is trying to use the two for loop to calculate the dW and tbe loss
    # Input: W, X, y
    # Outpur: loss

    
    
    # calculate the score from X and W, y[i] means the correct class, so scores[y[i]] means the scores of the correct course
    # the inner loop in the navie function which go through every classes is substituted by [range(num_train), y],     
    delta = 1.
    num_train = X.shape[0] # number of train (N, )
    w_j_score = X.dot(W) # X: (N, D), W: (D, C), so XW: (N, C)
    
    w_yi_score = w_j_score[range(num_train), y]
    # print(w_yi_score.shape)

    # calculate the new scores by eq from svm: max(0, xiwj -xiwyi + delta), get the loss by sum up every elements in the new scores matrix.
    correct_score = np.maximum(0, w_j_score - w_yi_score.reshape(num_train, 1) + delta)
    correct_score[range(num_train), y] = 0
    #print(correct_score.shape)
    
    # 
    loss = np.sum(correct_score)
    loss = loss / num_train 
    #W_square = W.dot(W)
    loss = loss + reg * np.linalg.norm(W) * np.linalg.norm(W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    # This part is trying to use the two for loop to calculate the dW and tbe loss
    # Input: W, X, y
    # Outpur: loss, dW

    dW = np.zeros(W.shape) 
    
    # and we know that when j equal to the correct class y[i] dLi/dwj = (xi * wj - xi * wj + delta > 0) * xi, 
    # dLi/dwy[i] = - (xi * wj - xi * wj + delta > 0) * xi and if it's not equal to y[i]
    # the equation will become sigma for all j not equal to y[i]: - (xi * wj - xi * wj + delta > 0) * xi
    # the first step we have to do is find out which elements is larger than 0, and then we change it to 1, 
    Binary = correct_score
    Binary[correct_score > 0] = 1
    # Binary[correct_score < 0] = 0
    # because in the previous section, I have already change every elements which less than 0 so we do not need to do it again.
    # the second step is calculate how many elements in a training data are not zero and minus it at every correct classes [range(num_train), y]
    sum_dl_dwyi = -np.sum(Binary, axis=1)
    Binary[range(num_train), y] = sum_dl_dwyi.T
    # and then multiply X and the Binary matrix and divide number of training points and then add the reg.
    dW = X.T.dot(Binary) / num_train + reg * W
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
