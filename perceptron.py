import numpy as np


class Perceptron:
    """
    Representation of a perceptron implementing the Rosenblatt's
    algorithm.

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
    errors : list of int
        Number of misclassifications in each epoch.
    """
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 20):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
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
        self : Perceptron
            Fitted perceptron.    

        Notes
        -----
        Modifies the model's internal state by updating:
        - weights vector
        - bias term
        - errors list tracking misclassifications per epoch
        """
        _, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.errors = [] # Array of errors to plot them for analysis
        # Set biass as zero but could be also a very small random number.
        self.bias = 0.0
        
        for _ in range(self.n_iters):
            error_counter = 0 # Errors counter in the given epoch, differrent than self.errors
            for i, x_i in  enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activate_function(linear_output)

                # Weights and bias update
                update = self.learning_rate * (y[i] - y_predicted)
                self.weights += update * x_i
                self.bias += update

                error_counter += int(update != 0)
            self.errors.append(error_counter)
            # If there are no errors, terminate
            if error_counter == 0:
                break

    def _activate_function(self, y: float) -> int:
        """
        Activate function as step function.

        Parameters
        ----------
        y : float
            Linear combination of inputs and weights (y = w·x + b).

        Returns
        -------
        int
            Thresholded output (0 or 1).
        """
        return np.where(y > 0, 1, 0)

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
        return self._activate_function(linear_output)
    