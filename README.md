# simpleMLP
A very simple Python code to generate and train 1 or 2-layer fully connected neural network (MLP)

The MLP relies only on Python and Numpy for calculation and matplotlib for display of graphs

## How to use
Everything must be done in the `test.py` file.
`neuralnetwork.py` includes the algorithms necessary for pass-forwarding, retro-propagation and formatting tools.

### Loading your data
All data must be in the form of numpy.array. Sample data must be of`float` type, while Labels must be of `int` type.
* `x` = training data. Shape = [dimensions of sample, examples]
* `labels` = training labels. Shape = [nb of classes, examples]
* `xtest` = test data. Same shape as x.
* `labeltest` = test labels. Same shape as labels

### Parameters
* `c` = number of hidden-layer perceptrons (only used in 2-layers MLP)
* `lr` = learning rate
* `it_train` = number of training iterations between each test on the test database
* `epoch` = number of test iterations

Therefore, the total number of training is it_train * epoch.

### Choose between 1 layer or 2 layers
In `test.py`, 2 successive zones of code represents the training for 1-layer and 2-layers MLP. Simply comment the one you don't want to use.
