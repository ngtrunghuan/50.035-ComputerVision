import numpy as np
from random import shuffle
import math

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
    labelScore = X.dot(W)
    labelScoreIndex = np.argsort(labelScore)
    labels = np.argmax(labelScoreIndex, axis = 1)
    for sample in range (X.shape[0]):
        score = X[sample].dot(W)
        score -= np.max(score)
        exp_sum = np.sum(np.exp(scores))
        softmax = np.exp(scores[y[i]]) / exp_sum
        loss -= np.log(softmax)
    
        scores_p =np.exp(scores)/exp_sum
        for j in range(W.shape[1]):
            if j == y[i]:
                dscore = scores_p[j] - 1
        else:
            dscore = scores_p[j]
            dW[:,j] += dscore * X[i]

        # sampleGradient = 0 if not isGroundTruth else  

    loss = loss / X.shape[0] + 0.5 * reg * np.sum(W * W)
    dW = dW / X.shape[0] + reg * W


    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
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
    scores = np.dot(X, W)

    # prevent numerical instability as noted in
    # http://cs231n/github.io/linear-classify/#softmax
    scores -= np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    # smoothing factor to prevent 'Divide by zero in log' error
    smooth_factor = 1e-14
    N = X.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y] + smooth_factor)) / N

    # as noted in
    # http://cs231n.github.io/neural-networks-case-study/#grad
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N
    dW = np.dot(X.T, dscores)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

