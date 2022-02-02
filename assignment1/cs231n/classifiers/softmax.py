from builtins import range
from curses.ascii import SO
import numpy as np
from random import shuffle
# from past.builtins import xrange


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

    num_classes = W.shape[1]
    num_train = X.shape[0]

    S = X.dot(W)
    Exp_s = np.exp( (S.T - np.max(S, axis=1)).T )
    Soft_S = Exp_s / np.sum(Exp_s, axis=1).reshape(-1,1)
    loss = np.sum(-np.log(Soft_S[range(num_train), y])) / num_train

    # Back propagation
    dL = np.array([1/num_train]*num_train)
    yS = np.zeros_like(Soft_S)
    yS[range(num_train), y] = 1.0
    dS = np.tile(dL, (num_classes, 1)).T * ( Soft_S - yS )
    dW = X.T.dot(dS)

    # # answer
    # F = X.dot(W)
    # exp_normalized_F = np.exp( (F.T - np.max(F, axis=1)).T )
    # sum_i = np.sum(exp_normalized_F, axis=1)
    
    # p_i = exp_normalized_F[range(num_train), y] / sum_i
    # L_i = - np.log(p_i)
    # loss = np.sum(L_i)
    
    # loss /= num_train
    # loss += reg*np.sum(W * W)

    # acc_effect = (exp_normalized_F.T / sum_i).T
    # acc_effect[range(num_train), y] -= 1.0
    # dW = X.T.dot(acc_effect)

    # dW /= num_train

    dW += reg * W * 2

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

    s = X.dot(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    S = X.dot(W)
    Exp_s = np.exp( (S.T - np.max(S, axis=1)).T )
    Soft_S = Exp_s / np.sum(Exp_s, axis=1).reshape(-1,1)
    loss = np.sum(-np.log(Soft_S[range(num_train), y])) / num_train


    # Back propagation
    dL = np.array([1/num_train]*num_train)
    yS = np.zeros_like(Soft_S)
    yS[range(num_train), y] = 1.0
    dS = np.tile(dL, (num_classes, 1)).T * ( Soft_S - yS )
    dW = X.T.dot(dS)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
