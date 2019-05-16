import numpy as np
from sklearn.metrics import accuracy_score
from scipy.misc import logsumexp
from sklearn.externals import six
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import safe_sparse_dot

from util import atleast2d_or_csr, count_trans, validate_lengths

from skmultiflow.core.base import StreamModel
from skmultiflow.utils.data_structures import InstanceWindow


class HMM(StreamModel):
    def __init__(self, rate=.01, window_size=100, max_models=100, delay=1):
        self.rate = rate 
        self.H = []
        self.h = None
        # TODO
        
        self.counter = 0
        self.window_size = window_size
        self.max_models = max_models
        self.window = InstanceWindow(window_size)
        self.delay = delay
        self.delay_counter = 0

        self.reset() 

    def fit(self, X, y, classes=None):
        """fit
        Train the HMM model to fit X, y        
        
        Parameters
        ----------
        X: Array-like
            The feature's matrix.
        
        y: Array-like
            The class labels for all samples in X.
            
        classes: list, optional
            A list with all the possible labels of the classification problem.

        Returns
        -------
        HMM
            self
        
        """

        rate = self.rate
        if rate <= 0:
            raise ValueError("rate should be >0, got {0!r}".format(rate))

        X = atleast2d_or_csr(X)
        classes, y = np.unique(y, return_inverse=True)
        lengths = [X.shape[0]]
        lengths = np.asarray(lengths)
        Y = y.reshape(-1, 1) == np.arange(len(classes))

        end = np.cumsum(lengths)
        start = end - lengths

        init_prob = np.log(Y[start].sum(axis=0) + rate)
        init_prob -= logsumexp(init_prob)
        final_prob = np.log(Y[start].sum(axis=0) + rate)
        final_prob -= logsumexp(final_prob)

        feature_prob = np.log(safe_sparse_dot(Y.T, X) + rate)
        feature_prob -= logsumexp(feature_prob, axis=0)

        trans_prob = np.log(count_trans(y, len(classes)) + rate)
        trans_prob -= logsumexp(trans_prob, axis=0)

        self.coef_ = feature_prob
        self.intercept_init_ = init_prob
        self.intercept_final_ = final_prob
        self.intercept_trans_ = trans_prob

        self.classes_ = classes

        return self

    def predict(self, X):
        """ predict
        
        Predict expected y of given X        

        Parameters
        ----------
        X: Array like
            The feature's matrix.
        
        Returns
        -------
        list
            A list containing the predicted labels for all instances in X.
        
        """
        N, D = X.shape

        # we create the vector with the length of H
        if len(self.H) > 0 :
            res = zeros(len(self.H)) 
        else:
            0

        # do the prediction for every model in H
        for i in range(len(self.H)):
            res[i] = self.H[i].predict_proba(X)

        # You also need to change this line to return your prediction instead of 0s:
        return res

    def score(self, X, y):
        '''score
        
        Calculate the accuracy of the test data by using the HMM algorithm

        Parameters
        ----------
        X: Array-like
            The features matrix.

        y: Array-like
            An array-like containing the class labels for all samples in X.

        Returns
        -------
        float
            Score.

        '''        
        return accuracy_score(y, self.predict_proba(X))

    def reset(self):
        '''reset
        Reset the parameters of the HMM
        '''
        self.coef_ = None
        self.intercept_init_ = None
        self.intercept_final_ = None
        self.intercept_trans_ = None
        self.classes_ = None

    def get_info(self):
        return 'Hidden Markov Model'
    
    def predict_proba(self, X):
        ''' predict_proba
        Calculate the probability of the X, given the model

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
  
        '''
        X = atleast2d_or_csr(X)
        scores = safe_sparse_dot(X, self.coef_.T)
        if hasattr(self, "coef_trans_"):
            n_classes = len(self.classes_)
            coef_t = self.coef_trans_.T.reshape(-1, self.coef_trans_.shape[-1])
            trans_scores = safe_sparse_dot(X, coef_t.T)
            trans_scores = trans_scores.reshape(-1, n_classes, n_classes)
        else:
            trans_scores = None

        y = self._viterbi(scores, trans_scores, self.intercept_trans_,
                   self.intercept_init_, self.intercept_final_)
        
        return y

    def _viterbi(self, score, trans_score, b_trans, init, final):
        """First-order Viterbi algorithm.

        Parameters
        ----------
        score : array, shape = (n_samples, n_states)
            Scores per sample/class combination; in a linear model, X * w.T.
            May be overwritten.
        trans_score : array, shape = (n_samples, n_states, n_states), optional
            Scores per sample/transition combination.
        trans : array, shape = (n_states, n_states)
            Transition weights.
        init : array, shape = (n_states,)
        final : array, shape = (n_states,)
            Initial and final state weights.

        References
        ----------
        L. R. Rabiner (1989). A tutorial on hidden Markov models and selected
        applications in speech recognition. Proc. IEEE 77(2):257-286.
        """

        n_samples, n_states = score.shape[0], score.shape[1]

        backp = np.empty((n_samples, n_states), dtype=np.intp)

        for j in range(n_states):
            score[0, j] += init[j]

        # Forward recursion. score is reused as the DP table.
        for i in range(1, n_samples):
            for k in range(n_states):
                maxind = 0
                maxval = -np.inf
                for j in range(n_states):
                    candidate = score[i - 1, j] + b_trans[j, k] + score[i, k]
                    if trans_score is not None:
                        candidate += trans_score[i, j, k]
                    if candidate > maxval:
                        maxind = j
                        maxval = candidate

                score[i, k] = maxval
                backp[i, k] = maxind

        for j in range(n_states):
            score[n_samples - 1, j] += final[j]

        # Path backtracking
        path = np.empty(n_samples, dtype=np.intp)
        path[n_samples - 1] = score[n_samples - 1, :].argmax()

        for i in range(n_samples - 2, -1, -1):
            path[i] = backp[i + 1, path[i + 1]]

        return path
    
    def partial_fit(self, X, y = None, classes=None):
        ''' partial_fit
        
        Update the HMM with new X, y

                Parameters
        ----------
        X: Array-like
            The feature's matrix.

        y: Array-like
            The class labels for all samples in X.

        classes: list, optional
            A list with all the possible labels of the classification problem.

        Returns
        -------
        HMM
            self

        '''

        N, D = X.shape;
        
        for i in range(N):
            
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
            self.counter += 1

            # fit the new model if window_size is sufficient
            if self.counter == self.window_size:

                X_iter = self.window.get_attributes_matrix()
                y_iter = self.window.get_targets_matrix()
                self.counter=0
                self.delay_counter += 1
                if self.delay_counter == self.delay:
                    self.fit(X_iter, y_iter)
                
                # remove the most ancient model if the model_size is over than max_models
                if(len(self.H) == self.max_models):
                    self.H.pop(0)
                self.H.append(self)
        return self