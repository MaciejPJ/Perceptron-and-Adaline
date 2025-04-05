import numpy as np
import matplotlib.pyplot as plt


class Adaline:
    """
    Representation of an Adaptive Linear Neuron.

    Parameters
    ----------
        learning_rate : float, optional
            Controlls the magnitude, of weight updates.
            Default=0.01. Must be in (0, 1]
        n_iters : int, optional
            A number of iterations through test set (epochs).
            Default=20. 
            

    Attributes
    -----------
    weights : ndarray of shape (n_features)
        Weights after fitting.
    bias : float
        Bias unit after fitting.
    losses : list
        MSE loss function in each epoch.
    """
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 20):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Adaline':
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
        self : Adaline

        Notes
        -----
            The method used is Batch Gradient Descent, thereofre
            there is only one loop and weights are updated once 
            per epoch.
        """
        m, n_features = X.shape
        self.weights = np.zeros(n_features)
        # Set biass as zero but could be also a very small random number.
        self.bias = 0.0
        self.losses = []
        
        for epoch in range(self.n_iters):
            # Calculate lin. output for all samples
            linear_output = self._activate_function(np.dot(X, self.weights) + self.bias)

            errors = y - linear_output

            # Batch gradient descent: weights and bias update.
            # The number 2 below comes from the result of partial derivative
            # of weights and bias. It could be skipped just included in
            # learning_rate or even deleted.
            self.weights += self.learning_rate * (2.0 / m) * np.dot(X.T, errors)
            self.bias += self.learning_rate * 2.0 * errors.mean()

            # Compute MSE for epoch.
            loss = (errors**2).mean()
            self.losses.append(loss)
        return self

    def _activate_function(self, y: float) -> int:
        """
        Activate function for adaline.

        Parameters
        ----------
        y : float
            Linear combination of inputs and weights (y = w·x + b).

        Returns
        -------
        float
            Raw linear output
        """
        return y

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
            Predicted class labels (0 or 1) for each input sample.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.5, 1, 0)
    