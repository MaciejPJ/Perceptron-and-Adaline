import numpy as np


class AdalineSGD:
    """
    Representation of an Adaptive Linear Neuron
    with stochastic gradient descent.

    Parameters
    ----------
    learning_rate : float, optional
        Controls the magnitude of weight updates.
        Default=0.01. Must be in (0, 1]
    n_iters : int, optional
        A number of iterations through test set (epochs).
        Default=20.      
    random_state : int, optional
        Seed for weight initialization   

    Attributes
    -----------
    weights : ndarray of shape (n_features)
        Weights after fitting.
    bias : float
        Bias unit after fitting.
    cost : list
        Sum of squared error after each epoch.
    """
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 20,
                 random_state=None):
        
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1]")
        
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdalineSGD':
        """
        Fit the perception model to training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix where n_samples is the number of samples
            and n_features is the number of features.
        y : ndarray of shape (n_samples,)
            Target values (class labels) as integers.

        Returns
        -------
        self : AdalineSGD

        Notes
        -----
            The method used is Stochastic Gradient Descent, weights
            update occurs for every example.
        """
        rgen = np.random.RandomState(self.random_state)
        _, n_features = X.shape
        self.weights = rgen.normal(loc=0.0, scale=0.01,
                                   size=n_features)
        self.bias = rgen.normal(loc=0.0, scale=0.01)

        self.cost = []

        for epoch in range(self.n_iters):
            # Shuffle data in each epoch
            shuffled_indices = rgen.permutation(len(y))
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            # Update weights for each epoch
            for xi, target in zip(X_shuffled, y_shuffled):
                output = self._net_input(xi)
                error = target - output
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

            # Calculate the cost after each epoch
            cost = ((y - self._net_input(X)) ** 2).mean()
            self.cost.append(cost)

        return self

    def _net_input(self, x: np.array) -> np.ndarray:
        """
        Activate function for adaline.

        Parameters
        ----------
        x : float
            Linear combination of inputs and weights (y = w·x + b).

        Returns
        -------
        float
            Raw linear output
        """
        return np.dot(x, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels (0 if net_input < 0.5, 1 otherwise)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.5, 1, 0)
    
