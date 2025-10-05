import random

import numpy as np
from scipy import spatial
from scipy import stats

class KNN:

    """
    Implementation of the k-nearest neighbors algorithm for classification
    and regression problems.
    """

    def __init__(self, k, aggregation_function):

        """
        Takes two parameters.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point. The
        aggregation_function is either "mode" for classification or
        "average" for regression.
        
        Parameters
        ----------
        k : int
           Number of neighbors
        
        aggregation_function : {"mode", "average"}
           "mode" : for classification
           "average" : for regression.
        """

        self.k = k
        self.aggregation_function = aggregation_function
       
        
    def fit(self, X, y):
        
        """
        Stores the reference points (X) and their known output values (y).
        
        Parameters
        ----------
        X : 2D-array of shape (n_samples, n_features) 
            Training/Reference data.
        y : 1D-array of shape (n_samples,) 
            Target values.
        """

        self.X = X
        self.y = y
        
        
    def predict(self, X):

        """
        Predicts the output variable's values for the query points X.
        
        Parameters
        ----------
        X : 2D-array of shape (n_queries, n_features)
            Test samples.
            
        Returns
        -------
        y : 1D-array of shape (n_queries,) 
            Class labels for each query.
        """

        # ! distance between query and training points
            # ~  X[:, np.newaxis, :] adds a new axis position at position 1 -> shape (n_queries, 1, n_features)
            # ~ self.X[np.newaxis, :, :] adds a new axis at position 0 -> shape (1, n_train, n_features)
            

        # [2,3] - [1,2] = [1,1]
        diff = X[:, np.newaxis, :] - self.X[np.newaxis, :, :]
        
        # sqrt(1^2 + 1^2) = sqrt(2)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        # indices of training points sorted by distance
        nearest = np.argsort(dist, axis=1)[:, :self.k]

        # retrieve labels of nearest neighbors
        labels = self.y[nearest]

        # finding the most common label
            # axis means to look at the rows and perform operations in that row

        if self.aggregation_function == "mode":
            # classification
            y_pred, _ = stats.mode(labels, axis=1, keepdims=False)

        else:
            # regression
            y_pred = np.mean(labels, axis=1)
        
        # return the predictions
        return y_pred