
import numpy as np
from scipy import optimize
from cvxopt import solvers
from cvxopt import matrix

class SVM:
    def __init__(self, C=1, class_weight = "balanced"):
        self.C = C
        self.class_weight = class_weight

    def fit(self,K,y):
        n_class_1 = np.sum(y)
        y=2*y-1
        N = len(y)
        Dy = np.diag(y)

        # Loss : 0.5 x'Px + q'x
        P = Dy @ K @ Dy
        q=-np.ones(N)

        # Inequality constraints Gx<h
        G=np.vstack((np.eye(N),-np.eye(N)))
        h=np.zeros(2*N)

        if self.class_weight == None:
            h[:N]=self.C*np.ones(N)

        elif self.class_weight == "balanced":
            weights = []
            for label in y:
                if label == 1:
                    weights.append(N/n_class_1)
                else:
                    weights.append(N/(N-n_class_1))
            weights = np.array(weights)

            h[:N]=self.C*weights


        # Equality constraint Ax=b
        A=np.zeros((1,N))+y
        b=np.zeros((1,1))

        
        solvers.options['show_progress'] = False
        sol = solvers.qp(P=matrix(P),q=matrix(q), G=matrix(G), h=matrix(h), A=matrix(A), b=matrix(b))

        dual = np.array(sol['x']).squeeze()
        self.primal = y*dual

    def predict_log_proba(self,K_test_train):
        preds=K_test_train@self.primal
        return preds

        

