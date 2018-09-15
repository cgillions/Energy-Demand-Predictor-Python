from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import random
import numpy

# Seed the rng so that experiments are comparable.
random.seed(10)

# Load the training, testing, and validation data.
validationData = numpy.loadtxt(open("validation.csv", "rb"), delimiter=",")
trainData      = numpy.loadtxt(open("train.csv",      "rb"), delimiter=",")
testData       = numpy.loadtxt(open("test.csv",       "rb"), delimiter=",")

# Separate the input from the target.
validationTarget = validationData[:, 3]
trainTarget      = trainData[:, 3]
testTarget       = testData [:, 3]

validationData = validationData[:, [0, 1, 2]]
trainData      = trainData[:, [0, 1, 2]]
testData       = testData [:, [0, 1, 2]]

# Normalize the input data.
validationData[:, 0] /= 366
validationData[:, 1] /= 24
validationData[:, 2] /= 7

trainData[:, 0] /= 366
trainData[:, 1] /= 24
trainData[:, 2] /= 7

testData[:, 0] /= 366
testData[:, 1] /= 24
testData[:, 2] /= 7

# Initialise the network.
mlp = MLPRegressor(hidden_layer_sizes=(50, 60, 40), momentum=0.9, max_iter=750, early_stopping=True)

# Train the network.
mlp.fit(X=trainData, y=trainTarget)

# Get the network's prediction for the validation data.
results = mlp.predict(validationData)

# Calculate the error).
error = numpy.sqrt(mean_squared_error(validationTarget, results))
print("Error = {}".format(error))

# Store the results in a file.
with open("validation-output.txt", "w") as file:
    for value in results:
        file.write("{}\n".format(value))
