from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # affline forward and save wx + b in out 
    x_reshape = x.reshape(x.shape[0], -1)
    # print(x_reshape.shape)
    out = x_reshape.dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
   
    # affline backward, calcualte the dx, dw, db based on x, w, b, which are from forward and dout (upstreaming gradient)
    dx = dout.dot(w.T).reshape(x.shape)
    
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Relu forward, nonlinear layers in out two layers network
    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    # Relu backward, nonlinear layers in out two layers network
    dx = (x > 0) * dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # get the mean of all feature
        mu = np.sum(x, axis=0) / N
        # mu = np.average(x, axis=0)
        # get the variance from the data
        var = np.var(x, axis=0)
        # normlize x caleld x hat, prevent divide by 0 so add eps
        x_hat = (x - mu) / np.sqrt(var + eps)
        
        out = gamma * x_hat + beta
        # equation from above
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        # save the data to cache
        cache = (mu, var, x_hat, gamma, beta, eps, x)
        # pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # use the mean and var from the test mode
        x_hat = (x - running_mean) / np.sqrt(eps + running_var)
        out = gamma * x_hat + beta
        
        # pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # get the save data from cache
    mu, var, x_hat, gamma, beta, eps, x = cache

    
    # in the last part is gamma * x, so the backpropagation of dx become dout * gamma
    dx_hat = dout * gamma
    # get the derivative of the variance
    dvar = np.sum(dx_hat*(x - mu), axis=0)
    # backpropagationof mean / variance
    dxmu1 = dx_hat / np.sqrt(var + eps)
    # derivative of sqrt variance
    dsqrtvar = -1 / (var + eps) * dvar
    # derivative of dvariance from the dsqrtvariance
    dvar = (1 / 2) / np.sqrt(var + eps) * dsqrtvar
    # derivative of summation
    dsu = np.ones((dout.shape[0], dout.shape[1])) / dout.shape[0]
    dsu = dsu * dvar
    # backpropagation of the square
    dxmu2 = 2 * (x - mu) * dsu
    # add them up
    dx1 = (dxmu1 + dxmu2)
    # derivative of mu
    dmu = -1 * np.sum(dx1, axis=0)
    dx2 = np.ones((dout.shape[0], dout.shape[1])) / dout.shape[0] * dmu
    # reconstruct the partical derivative of x
    dx = dx1 + dx2
    # calculate the dbeta and dgamma 
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0) 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # get the store value in forward
    mu, var, x_hat, gamma, beta, eps, x = cache
    # get the dbeta and dgamma from eq 
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    # get dmean by average the dout
    dmean = np.sum(dout, axis=0) / dout.shape[0]
    # caluclate the dvariance
    dvar = 2 * np.sum((x - mu) * dout, axis=0)/ dout.shape[0]
    # because we use std in the forward, so we need to calculate dsqrtvariance there
    dsqvar = dvar/(2 * np.sqrt(var + eps))
    # get the gradient by using the derivative of the batch normlization equation
    dx = gamma * ((dout - dmean) * np.sqrt(var + eps) - dsqvar*(x-mu))/np.sqrt(var + eps)**2
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # do the same thing with the batch norm forward, just transpose the x 
    xT = x.T
    mu = np.sum(xT, axis=0) / x.shape[1]
    # mu = np.average(x, axis=1)
    var = np.var(xT, axis=0)
    x_hat = (xT - mu) / np.sqrt(var + eps)
        
    out = (gamma.reshape(-1,1) * x_hat + beta.reshape(-1,1)).T
    
        
    cache = (mu, var, x_hat, gamma, beta, eps, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # get the store value in forward
    mu, var, x_hat, gamma, beta, eps, x = cache
    # transpose the dout
    dout = dout.T
    # in the last part is gamma * x, so the backpropagation of dx become dout * gamma
    dx_hat = dout * gamma.reshape(-1,1)
    # derivative of variance
    dvar = np.sum(dx_hat*(x - mu.reshape(-1,1)).T, axis=0)
    # backpropagationof mean / variance
    dxmu1 = dx_hat / np.sqrt(var + eps)
    # backpropagation of std
    dsqrtvar = -1 / (var + eps) * dvar
    #derivative of the sqrt
    dvar = (1 / 2) / np.sqrt(var + eps) * dsqrtvar
    # derivative of summation
    dsu = np.ones((dout.shape[0], dout.shape[1])) / dout.shape[0]
    dsu = dsu * dvar
    #derivative of square
    dxmu2 = 2 * (x - mu.reshape(-1,1)).T * dsu
    # add all dout up
    dx1 = (dxmu1 + dxmu2)
    # derivative of mu
    dmu = -1 * np.sum(dx1, axis=0)
    dx2 = np.ones((dout.shape[0], dout.shape[1])) / dout.shape[0] * dmu
    # reconstruct the partical derivative of x
    dx = dx1 + dx2
    dx = dx.T
    # calculate the dbeta and dgamma 
    dbeta = np.sum(dout, axis=1)
    dgamma = np.sum(dout * x_hat, axis=1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

def affine_normalization_relu_forward(x, w, b, gamma, beta, bn_param, type_of_norm):
    """
    Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    #define the variable
    normalization_out =  None
    normalization_cache = None
    out = None
    relu_cache = None
    aff_out = None
    fc_cache = None
    # affine first
    aff_out, fc_cache = affine_forward(x, w, b)
    # check which type of norm we are using
    if type_of_norm == 'batchnorm':
        # norm -> relu
        normalization_out, normalization_cache = batchnorm_forward(aff_out, gamma, beta, bn_param)
        out, relu_cache = relu_forward(normalization_out)
    elif type_of_norm == 'layernorm':
        # norm -> relu
        ln_param = bn_param
        normalization_out, normalization_cache = layernorm_forward(aff_out, gamma, beta, ln_param)
        out, relu_cache = relu_forward(normalization_out)
    else: # If we do not want to use any normlization, it will go into this condition
        out, relu_cache = relu_forward(normalization_out)

    cache = (fc_cache, normalization_cache, relu_cache)
    return out, cache

def affine_normalization_relu_backward(dout, cache, type_of_norm):
    """
    Backward pass for the affine-relu convenience layer.
    """
    # define the variables
    dx, dgamma, dbeta, normalization_out = None, None, None, None
    # get the store data from cache
    fc_cache, normalization_cache, relu_cache = cache
    #relu first
    relu_out = relu_backward(dout, relu_cache)
    # check which tyoe of the norm
    if type_of_norm == 'batchnorm':
        dnormalization_out, dgamma, dbeta = batchnorm_backward(relu_out, normalization_cache)
    elif type_of_norm == 'layernorm':
        dnormalization_out, dgamma, dbeta = layernorm_backward(relu_out, normalization_cache)
    # affine in the end
    dx, dw, db = affine_backward(dnormalization_out, fc_cache)
    return dx, dw, db, dgamma, dbeta
    
    
def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # get the possiblity
        poss = dropout_param["p"]
        # use random to create mask also divide by possibility
        mask = np.random.rand(*x.shape) > poss / poss
        #print(mask)
        
        out = mask * x
        # pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # when in the test mode, do not need the mask
        out = x 
        # pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #pass
        # backward propogation
        dx = dout * mask
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # get the size of input x, w
    N, C, H, W = x.shape
    F, _, HW, WW = w.shape
    # check if the size of filter and your input is reasonable
    H_ = 1 + (H + 2 * conv_param['pad'] - HW) / int(conv_param['stride'])
    assert H_ % 1 == 0
    H_ = int(H_)
    W_ = 1 + (W + 2 * conv_param['pad'] - WW) / int(conv_param['stride'])
    assert W_ % 1 == 0
    W_ = int(W_)
    # expand the x with the pad number you defined
    pad = int(conv_param['pad'])
    x_expand = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # go through every data 
    total = 0
    stride = int(conv_param['stride'])
    out = np.zeros((N, F, H_, W_))
    
    for data in range(N):
        # go through every filter 
        for ft in range(F):
            # go through every pixels or you can say the value of data
            #for width in range(0, int(W_), stride):
            #    for height in range(0, int(H_), stride):
            for width in range(W_):
                for height in range(H_):
                    # get the result
                    # for sub_h in range(conv_param['stride'] * height, conv_param['stride'] + H_):
                    #    for sub_w in range(conv_param['stride'] * width, conv_param['stride'] + W_):
                    #        total = w[ft][:][sub_h][sub_w] * x[n][:][sub_h][sub_w] + b[ft][:][sub_h][sub_w] + total
                    width_range = range(width, width + W_)
                    height_range = range(height, height + H_)
                    # print(width_range)
                    # sum of W*x + b for every element (equation from cov forward)
                    out[data, ft, height, width] =  np.sum(w[ft] * x_expand[data, :, height * stride: height * stride + HW, width * stride: width * stride + WW]) + b[ft]
                    # total = 0  

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # get the parameteres
    x, w, b, conv_param = cache
    
    # get the conv params
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    # Add the pad to the x, the value of pad is all 0
    x_expand = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_expand = np.zeros(x_expand.shape)

    # define the parameters with correct shape
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)    

    # get the shape of x and w
    N, C, H, W = x.shape
    F, _, HW, WW = w.shape
    # check if the size of filter and your input is reasonable
    H_ = 1 + (H + 2 * conv_param['pad'] - HW) / int(conv_param['stride'])
    assert H_ % 1 == 0 #'vertical does not match'
    H_ = int(H_)
    W_ = 1 + (W + 2 * conv_param['pad'] - WW) / int(conv_param['stride'])
    assert W_ % 1 == 0 #'horizontal does not match'
    W_ = int(W_)
    # go through every data
    for data in range(N):
        # go through every filter
        for ft in range(F): 
            # go through every elements
            for height in range(H_): 
                for width in range(W_):
                    # calculate dx, dw, db
                    dx_expand[data, :, height * stride:height * stride + HW, width * stride:width * stride + WW] = w[ft] * dout[data, ft, height, width] + dx_expand[data, :, height * stride:height * stride + HW, width * stride:width * stride + WW]
                    dw[ft] = dw[ft] + x_expand[data, :, height * stride:height * stride + HW, width * stride:width * stride + WW] * dout[data, ft, height, width]
                    db[ft] = db[ft] + dout[data, ft, height, width] 

    # get rid of the pad
    dx = dx_expand[:, :, pad:-pad, pad:-pad]
    # check the shape
    assert dx.shape == x.shape #'shape is different'

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # get the size
    N, C, H, W = x.shape
    # get the height
    HP = pool_param['pool_height']
    WP = pool_param['pool_width']
    # get output height and check if its valid
    H_ = (H - HP) / pool_param['stride'] + 1
    assert H_ % 1 == 0
    H_ = int(H_)
    # get output width and check if its valid
    W_ = (W - WP) / pool_param['stride'] + 1
    assert W_ % 1 == 0
    W_ = int(W_)    # initialize out
    
    stride = int(pool_param['stride'])
    # initialize outpur
    out = np.zeros((N, C, H_, W_))
    
    # go through every elements
    for data in range(N):
        for channel in range(C):
            # get the maximum value in the pool
            for width in range(W_):
                for height in range(H_):
                    out[data, channel, height, width] = np.max(x[data, channel, height * stride:height * stride + int(pool_param['pool_height']), width * stride:width * stride + int(pool_param['pool_width'])])
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # get the parameters from forward
    x, pool_param = cache
    # define the output variable with correct shape
    dx = np.zeros(x.shape)
    # get the spape
    N, F, H, W = x.shape
    stride = int(pool_param['stride'])
#    print(x)
    # go through every date points to construct the grad from dout
    for data in range(N):
        for channel in range(F):
            for height in range(0, dout.shape[2]):
                for width in range(0, dout.shape[3]):
                    # find the maximum locaation from the pools
                    maximum = np.argmax(x[data, channel, height*stride:height*stride + pool_param['pool_height'], width*stride:width*stride + pool_param['pool_width']])
                    # print(int(maximum / pool_param['pool_width'], )
                    sub_h = int(maximum / pool_param['pool_width'])
                    sub_w = maximum % pool_param['pool_width']
                    # let the maximum locaation value become dout[data, channel, height, width] and others value are all 0
                    dx[data, channel, height*stride:height*stride + pool_param['pool_height'], width*stride:width*stride + pool_param['pool_width']][sub_h, sub_w] = dout[data, channel, height, width]


    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    #########][##################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # reconstruct the input with different order to fit the batchnorm_forward
    xt = np.transpose(x, (0, 2, 3, 1)).reshape(x.shape[0] * x.shape[2] * x.shape[3], x.shape[1])
    # use the batchnorm_forward with the reorder data
    out_not_tp_back, cache = batchnorm_forward(xt, gamma, beta, bn_param)
    # transfer it back
    out = np.transpose(out_not_tp_back.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1]), (0, 3, 1, 2))
    
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # reconstruct the dout with different order to fit the batchnorm_forward
    doutt = np.transpose(dout, (0, 2, 3, 1)).reshape(dout.shape[0] * dout.shape[2] * dout.shape[3], dout.shape[1])
    # use backward
    dx_not_tp_back, dgamma, dbeta = batchnorm_backward(doutt, cache)
    # transfer it back
    dx = np.transpose(dx_not_tp_back.reshape(dout.shape[0], dout.shape[2], dout.shape[3], dout.shape[1]), (0, 3, 1, 2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # check if I can correctly divide the input into G
    assert x.shape[1] / G % 1 == 0
    # get the shape
    N, C, H, W = x.shape
    # reshape it
    x_reshape = x.reshape(N * G, -1)
    
    # this part is mpstly following batchnormaization forward
    xT = x_reshape.T
    # calculate the mean 
    mu = np.mean(xT, axis=0) 
    # mu = np.average(x, axis=1)
    # get the variance
    var = np.var(xT, axis=0)
    # normlization it
    x_hat = (xT - mu) / np.sqrt(var + eps)
    # reshape it 
    x_hat = x_hat.T.reshape(x.shape)
    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)
    # gamma * x + b
    out = (gamma * x_hat + beta)
    
    # print(out)
    cache = (mu, var, x_hat, gamma, beta, eps, x, G)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # get the parameters from forward
    mu, var, x_hat, gamma, beta, eps, x, G = cache
    # get the shape
    N, C, H, W = x.shape
    # reshape it
    x_reshape = x.reshape(N * G, -1).T
    
    # get variance
    var = np.var(x_reshape, axis=0)
    x_hat_origin = x_hat
    x_hat = x_hat.reshape(N * G, -1).T
    # get the dbeta and dgamma from eq 
    
    # this part is mostly follows the batchnormlization_backward_alt
    dout_gamma = (dout * gamma).reshape(N * G, -1).T
    # print(dout.shape, mu.shape, x.reshape(N * G, -1).T)
    dout_gamma_mean = np.sum(dout_gamma, axis = 0) / x_hat.shape[0]
    dout_gamma_mean2 = np.sum(dout_gamma * x_hat, axis = 0) * x_hat / x_hat.shape[0]
    # get the grad dx
    dx = ((dout_gamma - dout_gamma_mean - dout_gamma_mean2) / np.sqrt(var + eps)).T.reshape(x.shape)
    # get the grad dgamma
    dgamma = np.sum(dout * x_hat_origin, axis=(0, 2, 3), keepdims = True) 
    # get the grad dbeta
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims = True)
    
    # calculate the dbeta and dgamma 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
