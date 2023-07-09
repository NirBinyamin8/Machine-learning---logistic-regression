import numpy as np
import math
from sklearn.metrics import confusion_matrix

class LogsticRegression :
    """
       Logistic Regression implementation using batch gradient descent.

       Parameters:
       -----------
       lr: float
           Learning rate for the algorithm. Default is 0.001.
       num_iter: int
           Number of iterations for the algorithm. Default is 1000.
       random_state: int
           Seed for the random number generator. Default is None.
       """
    def __init__(self, lr=0.001, num_iter=1000,batch_size=32):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = 0
        self.batch_size = batch_size
    def sigmoid(self, z):
        """
                          Computes the sigmoid function.

                          Parameters:
                          -----------
                          z: array_like
                              Input data to the sigmoid function.

                          Returns:
                          --------
                          sigmoid of the input data.
                          """
        m=np.exp(-z) + 1.0
        s=1.0 /m
        return s
    def predict(self,X,thresholds=0.5):
        """
                Predicts the class labels for the given input data.

                Parameters:
                -----------
                X: array_like
                    Input data for prediction.

                Returns:
                --------
                An array of predicted class labels.
                """
        X=np.array(X)
        y=[]
        for x in X:
            s=x@self.weights
            s=self.sigmoid(s)
            if(s<thresholds):
                s=0
            else:
                s=1
            y.append(s)
        return y
    def score(self,X,y,thresholds=0.5):
        """
                        Computes the accuracy score for the given input data and true labels.

                        Parameters:
                        -----------
                        X: array_like
                            Input data for prediction.
                        y: array_like
                            True labels for the input data.

                        Returns:
                        --------
                        The accuracy score for the input data and true labels.
                        """
        y_pred=self.predict(X,thresholds)
        E=confusion_matrix(y,y_pred)
        TN, FP, FN, TP = E.ravel()
        score=(TN+TP)/(TN+TP+FP+FN)
        return score
    def tpr_fpr(self,y_train,y_prob,thresholds=0.5):
        y_pred=[]
        for y in y_prob:
            if y<thresholds:
                y_pred.append(0)
            else:
                y_pred.append(1)

        E = confusion_matrix(y_train, y_pred)
        TN, FP, FN, TP = E.ravel()
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        return tpr,fpr
    def predict_proba(self,X):
        """
                Computes the probability estimates for the input data.

                Parameters:
                -----------
                X: array_like
                    Input data for prediction.

                Returns:
                --------
                An array of probability estimates.
                """
        X=np.array(X)
        y = []
        for x in X:
            s = x @ self.weights
            s = self.sigmoid(s)
            y.append(s)
        return y
    def fit_batchs(self,X,y):
        """
                Fits the logistic regression model to the input data using batch gradient descent.

                Parameters:
                -----------
                X: array_like
                    Input data for training.
                y: array_like
                    True labels for the input data.
                """
        X = np.array(X)
        y = np.array(y)
        # Initialize weights using Xavier initialization
        self.weights = np.random.randn(X.shape[1])* np.sqrt(1/X.shape[1])
        n_batches = len(X) // self.batch_size
        # Perform stochastic gradient descent for the specified number of iterations.
        for i in range(self.num_iter):
            for j in range(n_batches):
                # Determine the start index of the current batch.
                start = j * self.batch_size
                # Determine the end index of the current batch.
                end = start + self.batch_size
                # Extract the input samples for the current batch.
                X_batch = X[start:end]
                # Extract the target values for the current batch.
                y_batch = y[start:end]
                # Compute the predictions for the current batch.
                y_pred = self.sigmoid(X_batch @ self.weights)
                # Error
                error = y_pred-y_batch
                # Compute the gradient of the cost function with respect to the weights.
                gradient = X_batch.T @ error
                # Update the weights based on the gradient and the learning rate.
                self.weights -= self.lr * gradient
    def fit(self, X, y):
        """
        Fits the logistic regression model to the input data using batch gradient descent.

        Parameters:
        -----------
        X: array_like
            Input data for training.
        y: array_like
            True labels for the input data.
        """
        X = np.array(X)
        y = np.array(y)
        # Initialize weights using Xavier initialization
        self.weights = np.random.randn(X.shape[1]) * np.sqrt(1 / X.shape[1])
        # Perform gradient descent for the specified number of iterations.
        for i in range(self.num_iter):
            # Compute the predictions for the entire dataset.
            y_pred = self.sigmoid(X @ self.weights)
            # Compute the error between the predictions and the true labels.
            error = y_pred - y
            # Compute the gradient of the cost function with respect to the weights.
            gradient = X.T @ error
            # Update the weights based on the gradient and the learning rate.
            self.weights -= self.lr * gradient































