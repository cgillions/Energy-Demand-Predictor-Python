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

# Remove old configuration data.
filename = "optimization-results.txt"
open(filename, "w")

# Iterate through a range of values that could cover the input space.
for firstIndex in range(50, 201, 50):
    for secondIndex in range(5, 101, 5):
        for thirdIndex in range(1, 11):

            # Initialise the network with this configuration.
            mlp = MLPRegressor(hidden_layer_sizes=(firstIndex, secondIndex, thirdIndex),
                               max_iter=750, early_stopping=True)
            # Train the network.
            mlp.fit(X=trainData, y=trainTarget)

            # Get the network's prediction for this unseen data.
            results = mlp.predict(validationData)

            # Calculate the RMS error for this configuration on the validation data set.
            error = numpy.sqrt(mean_squared_error(validationTarget, results))

            # Write the configuration & output to a file.
            # Do this within the nested loop so that if the program is interrupted,
            # the data from previously run configurations is not lost.
            outputString = "{}\t{}\t{}\t{}\n".format(firstIndex, secondIndex, thirdIndex, error)
            with open(filename, "a") as file:
                file.write(outputString)
            print(outputString)

# Now we've optimised the network, get the validation output for the optimal MLP. #

# First find the optimal configuration.
minError = 1000000
optimalConfig = []
with open(filename, 'r') as file:
    for line in file:
        values = line.split('\t')
        if float(values[3]) < minError:
            optimalConfig = [values[0], values[1], values[2]]
            minError = float(values[3])

# Initialise the network with this configuration.
mlp = MLPRegressor(hidden_layer_sizes=(int(optimalConfig[0]), int(optimalConfig[1]), int(optimalConfig[2])),
                   momentum=0.9, max_iter=750, early_stopping=True)
# Train the network.
mlp.fit(X=trainData, y=trainTarget)

# Get the network's prediction for the validation data.
results = mlp.predict(validationData)

# Store the results in a file.
with open("validation-output.txt", "w") as file:
    for value in results:
        file.write("{}\n".format(value))
