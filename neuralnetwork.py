import numpy as np


def normalize(x):
    """ Normalize x positive vector or matrix between -1 and 1
           @:param x = matrix or vector to modify. Size [n, k]
           @:returns matrix or vector with added row of 1. Size is [n+1, k]"""
    range = 2
    s = x - x.min()
    s = range * s / s.max()
    return s - range / 2


def row1(x):
    """ Add a row of 1 to the matrix or vector x.
           @:param x = matrix or vector to modify. Size [n, k]
           @:returns matrix or vector with added row of 1. Size is [n+1, k]"""

    return np.insert(x, 0, 1, axis=0)


def mlp1def(n, m):
    """ Generate a weight and biais matrix for 1-layer MLP.
        @:param n = number of inputs of the network
        @:param m = number of perceptrons in the 1-layer MLP
        @:returns a [m, n + 1] random uniform matrix of weights (+ bias)"""

    return np.random.uniform(-1, 1, m * (n + 1)).reshape([m, n + 1])


def mlp2def(n, c, m):
    """ Generate a weight and biais matrix for 2-layer, fully connected MLP.
        @:param n = number of inputs of the network
        @:param c = number of perceptrons in the hidden-layer
        @:param m = number of perceptrons in the output-layer
        @:returns random uniform matrices of weights (+ bias) for the hidden and output layer"""

    return mlp1def(n, c), mlp1def(c, m)


def sigmo(v):
    """ Calculate the sigmoid function of each element in a vector.
        @:param v = argument vector
        @:returns a vector the same size as the argument vector activated by a sigmoid, element-wise"""
    limit = 2
    v[v < -limit] = -limit
    v[v > limit] = limit
    return (1 - np.exp(-2 * v)) / (1 + np.exp(-2 * v))
    # return (1 - np.nan_to_num(np.exp(-2 * v))) / (1 + np.nan_to_num(np.exp(-2 * v)))


def sigmop(v):
    """ Calculate the derived sigmoid function of each element in a vector or matrix.
        @:param v = argument vector or matrix
        @:returns a vector the same size as the argument vector"""

    return 1 - sigmo(v)**2


def mlp1run(x, w):
    """ Run a 1-layer MLP
        @:param x = input (vector or matrix). Size [n, k], n = number of input in one example, k = number of examples
        @:param w = matrix of weights and bias. Size [m, n+1], m = number of perceptrons
        @:returns output value of the MLP. Size [m, k]"""

    return sigmo(w.dot(row1(x)))


def mlp2run(x, w1, w2):
    """ Run a 2-layers MLP
        @:param x = input (vector or matrix). Size [n, k], n = number of input in one example, k = number of examples
        @:param w1, w2 = matrix of weights and bias for the 1st (hidden) layer and output layer.
        w1 of size Size [m, n+1], m = number of perceptrons of the 1st layer,
        w2 of size [c, m+1], c = number of perceptions of the last layer,
        @:returns output value of the MLP. Size [c, k]"""

    return mlp1run(mlp1run(x, w1), w2)


def mlpnrun(x, ws):
    """ Run a multi-layers MLP
        @:param x = input (vector or matrix). Size [n, k], n = number of input in one example, k = number of examples
        @:param ws = 3-dim matrix of weights and bias for each layer of the MLP. First dimension represents layer
        @:returns output value of the MLP. Size [m, k] with m the number of output-layer perceptrons."""

    for w in ws:
        x = mlp1run(x, w)
    return x


def mlpclass(y):
    """ Get the output label. Does the opposite of label2target().
        @:param y = output of the MLP (vector or matrix). Size [m, k] or [m, 1],
            m = nb of perceptrons of last layer, k = number of input examples
        @:returns output label (scalar or vector [1, k])"""

    return np.argmax(y, axis=0)


def label2target(c):
    """ Convert a label into a target with desired values (-1 or 1). Does the opposite of mlpclass().
    (assuming class are designed by a integer value from 0 to nclass - 1)
        @:param c = label. scalar or vector [1, k], k = number of examples
        @:returns a vector or matrix of targets. Labels are sorted from 0 to (nb of labels - 1). Size = [m, k]"""

    mat = -1. * np.ones([len(np.unique(c)), c.size], dtype=float)
    for i, p in enumerate(c):
        mat[p, i] = 1
    return mat


def score(label, labelD):
    """ Calculate the score and rate of the MLP.
        @:param label = output labels of the MLP. scalar or vector [1, k], k = number of labels
        @:param labelD = desired label. scalar or vector [1, k]
        @:returns (score) = number of successful classification. Size [1, k]
        @:returns (rate) = rate of successful classification. Size [1, k]"""

    return np.sum(label == labelD), np.sum(label == labelD) / label.size


def mlperror(y, target):
    """ Calculate the error as the difference between the output - target.
        @:param y = output of the MLP. Size [l, k], l = nb of possible labels, k = nb of examples
        @:param target = desired output. Size [l, k]
        @:returns a vector or matrix of errors. Size [l, k]"""

    return y - target


def sqrerror(error):
    """ Calculate quadratic error.
        @:param error = error. Size [l, k], l = nb of possible labels, k = nb of examples
        @:returns quadratic error (scalar) """

    return 0.5 * np.sum(np.square(error))


def mlp1train(x, target, w, lr, it):
    """ Train a 1-layer MLP.
        @:param x = input (vector [n,1] or matrix [n,k])
        @:param target = desired output (vector [m,1] or matrix [m,k])
        @:param w = matrix of weights and biais (matrix [m,n+1])
        @:param lr = learning rate (scalar)
        @:param it = number of iteration (scalar)
        @:returns L = quadratic error after the training (scalar)
        @:returns new weights and bias matrix after the training (matrix [m,n+1]) """

    L = []
    for i in range(it):
        error = mlperror(mlp1run(x, w), target)
        L.append(sqrerror(error))

        # Calculate deltas of layer
        deltas = error * sigmop(w.dot(row1(x)))

        # Calculate new weigths and bias
        w = w - lr * deltas.dot(row1(x).T)

    return w, L


def mlp2train(x, target, w1, w2, lr, it):
    """ Train a 2-layer MLP.
            @:param x = input (vector [n,1] or matrix [n,k])
            @:param target = desired output (vector [m,1] or matrix [m,k])
            @:param w1, w2 = matrix of weights and biais for the hidden and output layers (matrix [c,n+1] and [m,n+1])
            @:param lr = learning rate (scalar)
            @:param it = number of iteration (scalar)
            @:returns L = quadratic error after the training (scalar)
            @:returns new weights and bias matrix after the training (matrix [c,n+1] and [m,n+1] """

    L = []  # Quadratic error
    K = x.shape[1]

    for _ in range(it):
        # Input of each layer
        x1 = row1(x)
        x2 = row1(mlp1run(x, w1))

        # Calculate derivative of sigmoid for both layers
        sigp1 = sigmop(w1.dot(x1))
        sigp2 = sigmop(w2.dot(x2))

        # Calculate the output, error and squared error
        error = mlperror(mlp2run(x, w1, w2), target)
        L.append(sqrerror(error))

        # Calculate deltas of layers
        delta2 = error * sigp2
        delta1 = sigp1 * (w2[:, 1:].T.dot(delta2))

        # Calculate new weigths and bias
        w1 = w1 - lr * delta1.dot(x1.T) / K
        w2 = w2 - lr * delta2.dot(x2.T) / K

    return w1, w2, L

# DO NOT WORK YET !!
#
# def mlpntrain(x, target, ws, lr, it):
#     """ Train a n-layer MLP.
#             @:param x = input (vector [n,1] or matrix [n,k])
#             @:param target = desired output (vector [m,1] or matrix [m,k])
#             @:param ws, w2 = 3-dim matrix of weights and bias. First dim is the layer.
#             @:param lr = learning rate (scalar)
#             @:param it = number of iteration (scalar)
#             @:returns L = quadratic error after the training (scalar)
#             @:returns new weights and bias matrix after the training """
#
#     # ws = np.asarray(ws).squeeze()
#     #
#     # if ws.ndim == 2:
#     #     return mlp1train(x, target, ws, lr, it)
#     #
#     # elif ws.ndim == 3:
#
#     L = []  # Quadratic error
#
#     for _ in range(it):
#
#         # Generate the list of inputs for each layer
#         xs = [x]
#         for w in ws[:-1]:
#             xs.append(mlp1run(xs[-1], w).squeeze())
#
#         # Generate the list of sigmoid derivative for each layer
#         sigps = [w.transpose().dot(row1(x)) for w, x in zip(ws, xs)]
#
#         # Calculate the output, error and squared error
#         y = mlp1run(xs[-1], ws[-1])
#         error = mlperror(y, target)
#         L.append(sqrerror(error))
#
#         # Calculate deltas for each layer, in reverse order, then flip them
#         deltas = [np.multiply(error, sigps[-1])]
#         for sigp, w in zip(reversed(sigps[:-2]), reversed(ws)):
#             deltas.append([np.multiply(sigp, w[1:, :].transpose().dot(deltas[-1]))])
#
#         deltas.reverse()
#
#         # Calculate new weights and bias
#         # (A*B')' = B*A'
#         # ws = ws - lr * np.transpose(deltas.dot(np.transpose(xs, (0, 2, 1))), (0, 2, 1))
#         xs = [row1(x) for x in xs]      # Add 1's to matrix of inputs
#         ws = [w - lr * x.dot(np.asarray(d).T) for x, w, d in zip(xs, ws, deltas)]
#
#     return ws, L
