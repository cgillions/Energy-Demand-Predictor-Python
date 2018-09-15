% Load the relevant data.
optimisationResults = load("optimization-results.txt");
validationResults   = load("validation-output.txt");
validationData      = load("validation.csv");

% Find the index of the optimal MLP configuration.
[~, index] = min(optimisationResults(:, 4));

% Display the configuration & final error.
fprintf("Optimal MLP hidden layer configuration: (%d, %d, %d). Final error: %.4f.\n", optimisationResults(index, :));

% Plot the input vs output.
figure;
plot(validationData(:, 1), validationResults(:));
hold on;

% Plot the input vs target.
plot(validationData(:, 1), validationData(:, 4));
title("2017 Energy Demand");
xlabel("Day of year");
ylabel("Energy demand");
legend(["Predicted", "Target"]);