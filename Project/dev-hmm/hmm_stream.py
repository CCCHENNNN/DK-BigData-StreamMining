import numpy as np
from sklearn.metrics import accuracy_score
from scipy.misc import logsumexp
from sklearn.externals import six
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import safe_sparse_dot

from util import atleast2d_or_csr, count_trans, validate_lengths

from skmultiflow.core.base import StreamModel
from skmultiflow.utils.data_structures import InstanceWindow
from hmm import HMM
from sklearn.tree import DecisionTreeClassifier


class HMMStream():
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

        print("partial_fit N is:", N)

        if self.h = None
        self.h = HMM()
        
        for i in range(N):
            
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
            self.counter += 1

            # fit the new model if window_size is sufficient
            if self.counter == self.window_size:

                X_iter = self.window.get_attributes_matrix()
                y_iter = self.window.get_targets_matrix()
                self.counter=0
                # self.delay_counter += 1
                # if self.delay_counter == self.delay:
                #     self.h.fit(X_iter, y_iter)
                #     self.delay_counter = 0
                self.h.fit(X_iter, y_iter)
                
                # remove the most ancient model if the model_size is over than max_models
                if(len(self.H) == self.max_models):
                    self.H.pop(0)
                self.H.append(self.h)
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
            res = np.zeros(len(self.H))
        print("number of models is:", len(self.H))

        # do the prediction for every model in H
        for i in range(len(res)):
            print("index id is:", i)
            res[i] = self.H[i].predict(X)
        print("predict successful")

        # You also need to change this line to return your prediction instead of 0s:
        return res

    def get_info(self):
        return 'Hidden Markov Model Streamming'