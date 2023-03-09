from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # affline forward and save wx + b in out 
    x_reshape = x.reshape(x.shape[0], -1)
    # print(x_reshape.shape)
    out = x_reshape.dot(w) + b
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # affline backward, calcualte the dx, dw, db based on x, w, b, which are from forward and dout (upstreaming gradient)
    dx = dout.dot(w.T).reshape(x.shape)
    
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Relu forward, nonlinear layers in out two layers network
    out = np.maximum(0, x)
    
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Relu backward, nonlinear layers in out two layers network
    dx = (x > 0) * dout
    
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement loss and gradient for multiclass SVM classification.    #
    # This will be similar to the svm loss vectorized implementation in       #
    # cs231n/classifiers/linear_svm.py.                                       #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # This part is trying to use the two for loop to calculate the dW and tbe loss
    # Input: x, y
    # Outpur: loss

    
    
    # calculate the score from X and W, y[i] means the correct class, so scores[y[i]] means the scores of the correct course
    # the inner loop in the navie function which go through every classes is substituted by [range(num_train), y],     
    delta = 1.
    num_train = x.shape[0] # number of train (N, )
    w_j_score = x # X: (N, D), W: (D, C), so XW: (N, C)
    
    w_yi_score = w_j_score[range(num_train), y]
    # print(w_yi_score.shape)

    # calculate the new scores by eq from svm: max(0, xiwj -xiwyi + delta), get the loss by sum up every elements in the new scores matrix.
    correct_score = np.maximum(0, w_j_score - w_yi_score.reshape(num_train, 1) + delta)
    correct_score[range(num_train), y] = 0
    
    loss = np.sum(correct_score)
    loss = loss / num_train 
    #W_square = W.dot(W)
    loss = loss #+ reg * np.linalg.norm(W) * np.linalg.norm(W)

    dx = np.zeros(x.shape) 
    
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
    dx = (Binary) / num_train #+ reg * W

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the loss and gradient for softmax classification. This  #
    # will be similar to the softmax loss vectorized implementation in        #
    # cs231n/classifiers/softmax.py.                                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # in this part, I calculate the loss and the gradient without a for loop. First, I get the score function for every training data XW
    # second I want to prevent the exp(score) become too large, so I minus the score by its maximum value, by doing this all of the elements
    # expoential in the function will all <= 1 and > 0 and then I calculate the possibility matrix, calculate the possibitlty matrix is more
    # difficult than using a for loop, I need to calculate the sum of every row of the matrix and divide it. and then we can use the definition of
    # the softmax function to calculate loss (get every data's correct classes) divide by sum and use np.log. The calculation of grad can use
    # possibility matrix got from previous calculation, and minus one at every correct classes, and then multiple by X.
    
    score = x
    # print(X.shape, W.shape, score.shape, y.shape)
    # shift score to prevent value too large after exp
    shift_score = score - np.max(score, axis=1, keepdims=True)
    # expoential
    P_distribution = np.exp(shift_score)
    # get the possiability matrix
    P = P_distribution / P_distribution.sum(axis=1, keepdims=True)
    # find every correct class 
    sy = P[range(x.shape[0]), y]
    
    # calculate the loss based on softmax
    loss = -np.sum(np.log(sy) / np.sum(P, axis=1))
    
    loss = loss / x.shape[0] #+ reg * np.sum(W * W) 
    # loss = loss - np.log(P_distribution[y[i]])
    
    # define a matrix for calculate
    minus_one = np.zeros_like(P)
    # make all correct class value become -1
    minus_one[np.arange(y.shape[0]), y] = -1
    
    P = P + minus_one
    
    # get the dx
    dx = P / x.shape[0] #+ reg * W

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
