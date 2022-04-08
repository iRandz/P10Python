from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

import Functions

nnHiddenLayers = (10, 15, 6)  # NN


def regress(x_train, y_train, x_test, y_test, print_stuff):

	# ML -----------------------------------------------------------------------------------------
	# Prepare and train model
	mlp = MLPRegressor(hidden_layer_sizes=nnHiddenLayers, random_state=1, max_iter=1000)
	mlp.fit(x_train, y_train)

	# Validate models
	if print_stuff:
		print("\n -----------------------------")
		print("Multi layer perceptron")
		print("Iterations: " + str(mlp.n_iter_))
		print("Layers: " + str(mlp.hidden_layer_sizes))
	mlp_score = Functions.validate_regression_model(mlp, x_test, y_test, print_stuff)

	return mlp_score
