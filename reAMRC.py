from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import itertools
import numpy as np
import scipy.io
import time
import typing


class reAMRC:
    def __init__(self, X, Y, order=0, W=200, alpha=0.3, N=100, K=2000, 
                 feature_map="linear", D=200, gamma=0.15625, rndseed=round(time.time())) -> None:
        """
        Initializer: generate an AMRC with given dataset and hyperparameters. Note that the parameter "deterministic" is removed here. You may obtain the randomized result by calling randomized predictor.

        Input:
        ---
        :X: (Dataset) instance matrix, with shape (n_sample, d)

        :y: (Dataset) label vector, with shape (n_sample, ) or (n_sample, 1). y takes value from 0, 1, ..., n_class-1



        :order: (Hyper-dynamic) the order k in the paper. order=0 by default. 


        :W: (Hyper-track) the window size of probability estimation. W=200 by default. See equation (12).

        :alpha: (Hyper-track) the discount factor using for update the R matrix, which measures the variances of noise processes.


        :N: (Hyper-SG) the buffer size of uncertainty set. N=100 by default, but temporarily disabled in the current version.

        :K: (Hyper-SG) the number of iteration of SG. K=2000 by default.



        :feature_map: (Hyper-feature) type of feature map: "linear" or "RFF". "linear" by default
        
        :D: (Hyper-feature) number of random Fourier components. 200 by defult

        :gamma: (Hyper-feature) scaling factor of RFF features. 2**(-6) by default


        :rndseed: assign a fixed random seed if a reproducible experiment is required. This rndseed is independent of that of synthetic dataset generator. If using both, it is better not to have two ranges of random seed overlap (to avoid potential unexpected dependencies). 

        Output:
        ---
        None

        """

        # Read dataset
        self.X = X
        self.n_sample, self.d = self.X.shape
        self.Y = Y.astype("int64").reshape(-1, 1)
        self.n_class = len(np.unique(self.Y))
        # Normalize data
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)

        # Hyperparameter-dynamic
        self.order = order
        self.k = order # alias of self.order in case that I confuse about the notation.

        # Hyperparameter-track
        self.W = W
        self.alpha = alpha

        # Hyperparameter-Subgradient
        self.N = N
        self.K = K

        # Hyperparameter-feature
        self.feature_map = feature_map
        self.D = D # number of random Fourier components. 
        self.gamma = gamma # scaling factor of RFF features.


        ###################################################
        # Other necessary members for the algorithm:
        self.t = 0 # The index of the data to learn.
        self.rndseed = rndseed
        np.random.seed(rndseed)
        # Initialize the predictor h in the paper. (Not the same as the method self.predict())
        self.predictor = np.random.rand(self.n_class, 1)
        self.predictor = self.predictor / self.predictor.sum().item()


        ###################################################
        # For the feature map:
        # The dimension of feature map result
        self.m = self.n_class * self.d
        if feature_map == "RFF":
            self.m = self.n_class * 2 * D
        # Determine the vector u for feature map:
        # This is the vectors of RFF map. Only useful for RFF. u=[u1, u2,.., uD], ui is a column vector with dim d
        self.u = np.random.randn(self.d, self.D) * self.gamma

        ###################################################
        # For uncertainty set tracking:
        # Parameters that characterize an uncertainty set:
        self.lmb = np.zeros((self.m, 1)) # the confidence bound
        self.tau = np.zeros((self.m, 1)) # the estimated expectation

        self.e1 = np.zeros((1, self.order+1)) # This is the e1 in theorem 2, but it is a row vector.
        self.e1[0,0] = 1
        # Generate H matrix in eq(13) and (16)
        deltat, variance_init = 1, 0.001
        self.H = np.eye(N=self.order+1)
        for i in range(self.order):
            for j in range(i+1, self.order+1):
                self.H[i,j] = deltat ** (j - i) / np.math.factorial(j - i)
        # Intepret the matlab code "initialize_tracking.m":
        self.eta0 = np.zeros((self.order+1, self.m))
        self.eta = np.zeros((self.order+1, self.m))
        self.Sigma0 = np.zeros((self.order+1, self.order+1, self.m))
        self.Sigma = np.zeros((self.order+1, self.order+1, self.m))
        self.Q = np.zeros((self.order+1, self.order+1, self.m))
        self.R = np.zeros((1, self.m))
        self.epsilon = np.zeros((1, self.m))
        for i in range(self.m):
            self.Sigma0[:,:,i] = np.eye(N=self.order+1)
            self.Q[:,:,i] = variance_init * np.eye(N=self.order+1)
            self.R[0, i] = variance_init
            self.epsilon[0, i] = 0 - self.e1 @ self.eta0[:, i].reshape(-1, 1)
            self.eta[:,i] = self.H @ self.eta0[:,i]
            self.Sigma[:,:,i] = self.H @ self.Sigma0[:,:,i] @ self.H.T + self.Q[:,:,i]

        # p represents the est probability on y, s represents the std. Both are n_class by 1 vectors.
        self.p = np.zeros((self.n_class, self.n_sample))
        self.s = np.zeros((self.n_class, self.n_sample))

        ###################################################
        # For subgradient descent learning algorithm:
        self.F = np.zeros((0, self.m)) # F is a collection of candidates f1, f2, ..., each is a 1 by m row vectors. (see algo 4.)
        self.h = np.zeros((0, 1))  # h is a collection of corresponding h1, h2, ..., each is a quantity. (see algo 4.) NOT CLASSIFIER!
        # The following are intermediate vectors in SGD algorithm related to gradient and mu. Consider to delete in future versions:
        self.w = np.zeros((self.m, 1))
        self.w0 = np.zeros((self.m, 1))
        # The target variable to learn, which characterizes the classifier:
        self.mu = np.zeros((self.m, 1))  # With mu, the classifier is well-defined.


        ###################################################
        # For log purpose: Only useful when strictly following the original version of code.
        self.mistakes_idx_det = np.zeros((self.n_sample - 1, 1)) # mistakes_idx[t, 0] = 1 means a wrong prediction at time t.
        self.mistakes_det = 0 # number of the mistakes in total.
        self.mistakes_idx_rnd = np.zeros((self.n_sample - 1, 1)) # mistakes_idx[t, 0] = 1 means a wrong prediction at time t.
        self.mistakes_rnd = 0 # number of the mistakes in total.
        self.RUt = np.zeros((self.n_sample,1)) # RUt is the estimated regret of the uncertainty set. Used for calc performance bound.
        self.Rht = np.zeros((self.n_sample,1)) # Rht is the exact error probability (for randomized AMRC only). It will be updated in _update_predictor when self.t does not exceed self.n_sample 
        self.varphi = 0 # the most recent \varphi{\mu_t} obtained from the learning algorithm (so that you won't need to re-calc via mu.)

        return



    def is_finished(self):
        """
        return True if no more data to go, else False. The last data won't be used for training by default.
        """
        if self.t == self.n_sample - 1:
            return True
        return False


    def feature_vector(self, x, y):
        """
        Return the feature of a given instance-label pair (x, y)

        Input:
        ---
        :x: instance: row vector and column vector are both fine.
        :y: label

        Output:
        ---
        :fvec: the feature vector of (x, y), which is a column vector.
        """
        x = x.reshape(-1, 1)
        y = int(y)
        if self.feature_map == 'linear':
            x_phi = x
        else:
            prod = self.u.T @ x  # self.u is d by D, so self.u.T is D by d, and x is d by 1, so it works.
            x_phi = np.vstack([np.cos(prod), np.sin(prod)])
        
        e = np.zeros((self.n_class, 1))
        e[y, 0] = 1   # the e_y in equation (1)

        return np.kron(e, x_phi)

    # Tracking the uncertainty set. It should be called by self.fit()
    def tracking(self):
        """
        Tracking the uncertainty set. It should be called by self.fit()
        """
        # Read data x, y:
        x, y = self.X[self.t, :].reshape(-1, 1), int(self.Y[self.t, 0])
        feature = self.feature_vector(x, y)
        flen_per_label = self.m // self.n_class

        # First we estimate the probability distribution of y using (12), and its standard deviation.
        # p represents the est probability on y, s represents the std. Both are n_class by 1 vectors.
        for class_no in range(self.n_class):
            if self.t + 1 < self.W:
                self.p[class_no, self.t] = (self.Y[:self.t+1, 0] == class_no).sum() / (self.t+1)
                if self.t == 0:
                    self.s[class_no, self.t] = 0
                else: 
                    self.s[class_no, self.t] = self.p[class_no, :self.t+1].std(ddof=1) 
            else:
                self.p[class_no, self.t] = (self.Y[self.t-self.W+1 : self.t+1, 0] == class_no).sum() / self.W
                self.s[class_no, self.t] = self.p[class_no, self.t-self.W+1 : self.t+1].std(ddof=1)

        # Directly assume not unidimensional
        for i in range(self.m):
            if i >= y * flen_per_label and i < (y+1) * flen_per_label:
                innovation = feature[i, 0] - self.e1 @ self.eta[:, i].reshape(-1,1)
                self.R[0, i] = self.alpha * self.R[0,i] + (1 - self.alpha) * (self.epsilon[0,i] ** 2 + self.e1 @ self.Sigma[:,:,i] @ self.e1.T)
                K = (self.Sigma[:,:,i] @ self.e1.T) / (self.e1 @ self.Sigma[:,:,i] @ self.e1.T + self.R[0,i])
                self.eta0[:,i] = (self.eta[:,i].reshape(-1,1) + innovation * K).flatten()
                self.Sigma0[:,:,i] = (np.eye(N=self.order+1) - K @ self.e1) @ self.Sigma[:,:,i]
                self.Q[:,:,i] = self.alpha * self.Q[:,:,i] + (1 - self.alpha) * (innovation ** 2) * (K @ K.T)
                self.epsilon[0,i] = feature[i,0] - self.e1 @ self.eta0[:,i]
                self.eta[:,i] = self.H @ self.eta0[:,i]
                self.Sigma[:,:,i] = self.H @ self.Sigma0[:,:,i] @ self.H.T + self.Q[:,:,i]
                self.tau[i,0] = (self.p[y, self.t]) * (self.e1 @ self.eta[:,i])
                lmb_eta = np.sqrt(self.Sigma[0,0,i])
                self.lmb[i,0] = np.sqrt((lmb_eta ** 2 + (self.e1 @ self.eta[:,i]) ** 2) * (self.s[y,self.t] ** 2 + self.p[y,self.t] ** 2) - (self.e1 @ self.eta[:,i] ** 2) * (self.p[y, self.t] ** 2))
            else:
                self.eta[:,i] = self.H @ self.eta0[:,i]
                self.Sigma[:,:,i] = self.H @ self.Sigma0[:,:,i] @ self.H.T + self.Q[:,:,i]
                self.tau[i,0] = self.p[i // flen_per_label, self.t] * (self.e1 @ self.eta[:,i])
                lmb_eta = np.sqrt(self.Sigma[0,0,i])
                self.lmb[i,0] = np.sqrt((lmb_eta ** 2 + (self.e1 @ self.eta[:, i]) ** 2) * (self.s[i//flen_per_label,self.t] ** 2 + self.p[i//flen_per_label,self.t] ** 2) - ((self.e1 @ self.eta[:,i]) ** 2) * (self.p[i//flen_per_label,self.t]) ** 2)
        
        return

    # One-Step Learning function
    def fit(self):
        """
        Train the model with data indexed by self.t, and then move self.t forward.
        Do nothing if no more data to go.
        """
        if self.t == self.n_sample: # if no more data to learn, directly return.
            return
        
        # First track the uncertainty set, according to the description in algo 1.
        self.tracking()

        # learning the data indexed by self.t. ref: learning.m
        x = self.X[self.t, :].reshape(-1,1)
        theta, theta0 = 1, 1
        muaux = self.mu # Not clear. Can be viewed as an intermediate mu (as \bar{mu} in the algorithm)
        M = np.zeros((self.n_class, self.m)) # M is used to store potential feature vectors generated by (x,y) as y is unknown.
        for class_no in range(self.n_class):
            M[class_no, :] = self.feature_vector(x, class_no).flatten()
        # Find all the combination of classes, update container F and h.
        for class_cnt in range(1, self.n_class+1):
            combs = list(itertools.combinations([*range(self.n_class)], class_cnt)) # combs is the set of all the possible comb's of size class_cnt.
            self.F = np.vstack([self.F, np.zeros((len(combs), self.m))])
            self.h = np.vstack([self.h, np.zeros((len(combs), 1))])
            offset_idx = self.F.shape[0] - len(combs)
            for cno, comb in enumerate(combs):
                self.F[offset_idx + cno, :] = (M[comb, :].mean(axis=0).reshape(-1,1)).flatten()
                self.h[offset_idx + cno, 0] = -1 / class_cnt
        
        # Initial step; then the formal loop
        # The second half of Line 2 of the for loop in algo 4:
        v = self.F @ muaux + self.h # v is max{F mu - h} in the algorithm
        self.varphi = v.max()
        regularization = (self.lmb * np.abs(self.mu)).sum()
        R_Ut_best_value = 1 - self.tau.T @ muaux + self.varphi + regularization
        F_freq = np.zeros(self.F.shape[0])
        for l in range(1, self.K+1):
            muaux = self.w + theta * ((1 / theta0) - 1) * (self.w - self.w0)
            v = self.F @ muaux + self.h
            self.varphi, idx_mv = v.max(), v.argmax() # idx_mv means index of the max value
            fi = self.F[idx_mv, :].reshape(-1,1)
            F_freq[idx_mv] += 1
            subgradient_regularization = self.lmb * np.sign(muaux) # This is the regularization term for subgradient descent
            regularization = (self.lmb * np.abs(muaux)).sum() # This is the regularization term in the risk of uncertainty set
            g = - self.tau + fi + subgradient_regularization
            theta0 = theta
            theta = 2 / (l + 1)
            a = 1 / ((l + 1) ** (3/2))
            self.w0 = self.w
            self.w = muaux - a * g
            R_Ut = 1 - self.tau.T @ muaux + self.varphi + regularization
            if R_Ut < R_Ut_best_value:
                R_Ut_best_value = R_Ut
                self.mu = muaux
        
        v = self.F @ self.w + self.h
        self.varphi = v.max()
        regularization = (self.lmb * np.abs(self.w)).sum()
        R_Ut = 1 - self.tau.T @ self.w + self.varphi + regularization
        if R_Ut < R_Ut_best_value:
            R_Ut_best_value = R_Ut
            self.mu = self.w

        # Drop the least recent used candidates from F and h: follow the logic of learning.m
        not_used = list(np.where(F_freq == 0)[0])
        if self.F.shape[0] > self.N:
            if len(not_used) > self.F.shape[0] - self.N:
                self.h = np.delete(self.h, not_used[:self.h.shape[0]-self.N], axis=0)
                self.F = np.delete(self.F, not_used[:self.F.shape[0]-self.N], axis=0)
            else:
                # Here the matlab code does nothing indeed, so I do nothing here as well.
                ""
        # Record RUt:
        self.RUt[self.t] = R_Ut_best_value
        # Move the index to the next data to learn
        self.t += 1
        # Update the predictor:
        self._predictor_update()
        return
    
    def next_instance_label(self):
        """
        Return the next instance-label pair to go. 
        Output:
        ---
        :X: instance as a column vector.
        
        :Y: label as an integer.
        """
        return self.X[self.t, :].reshape(-1,1), int(self.Y[self.t,0])

    def next_instance(self):
        return self.X[self.t, :].reshape(-1,1)

    def next_label(self):
        return int(self.Y[self.t,0])

    def _predictor_update(self):
        """
        Update the predictor. It should be called by self.fit(). Do not call it elsewhere. 
        """
        # A flag for Rht update. As we mentioned in the definition of self.Rht, we update Rht only if the current self.t does not exceed n_sample.
        update_Rht = False
        x = None
        
        if self.t < self.n_sample:
            update_Rht = True
            x = self.X[self.t, :]
        else:
            x = self.X[self.n_sample-1, :]
        
        class_rule = np.ones((self.n_class, 1)) # the classifier itself, which is a distri over all possible classes.

        potential_feature_vec = np.zeros((self.n_class, self.m)) # each row corresponding to feature map of (x, possible y) 
        for class_no in range(self.n_class):
            potential_feature_vec[class_no, :] = self.feature_vector(x, class_no).T
        # Calculate c_y and c_x terms: where c_y is the terms inside the summation of c_x in algo 2.
        c_y = potential_feature_vec @ self.mu - self.varphi
        c_y[c_y < 0] = 0
        c_x = c_y.sum()
        # Calculate the classification rule
        if c_x == 0:
            class_rule = np.ones((self.n_class, 1)) / self.n_class
        else:
            class_rule = c_y / c_x

        self.predictor = class_rule

        if update_Rht:
            self.Rht[self.t] = 1 - self.predictor[self.Y[self.t].item()]
        return

    def predict(self, x=None, deterministic=True):
        """
        Predict the label of the input instance x, or predict the label of X[self.t, :] if no input. Return 0 if self.t==self.n_sample (so do not call it if the learning is completely done).

        Input:
        ---
        :x: the input instance. x=None by default, which means predicting the label of X[self.t, :]

        :deterministic: whether to use a deterministic decision rule or not.

        Output:
        ---
        :y_hat: the predicted y, which is an integer ranging from 0 to n_class-1.

        """
        if x == None:
            if self.t < self.n_sample:
                x = self.X[self.t, :]
            else:
                x = self.X[self.n_sample-1, :]
        
        x = x.reshape(1, -1) # form x into a row vector, but it does not matter indeed.

        if deterministic:
            y_hat = int(np.argmax(self.predictor))
        else:
            # y_hat = int(np.where(np.random.multinomial(1, class_rule.reshape(1,-1)[0]) == 1)[0])
            y_hat = np.random.choice(self.n_class, p=self.predictor.flatten())

        return y_hat

    def savemat(self, filename):
        mdict = {'mistakes_idx_det': self.mistakes_idx_det, 'mistakes_det': self.mistakes_det, 'mistakes_idx_rnd': self.mistakes_idx_rnd, 'mistakes_rnd': self.mistakes_rnd, 'RUt': self.RUt, 'Rht': self.Rht, 'rndseed': self.rndseed}
        scipy.io.savemat(filename, mdict)


##########################################################
# Helper functions:

def data_reader(filename):
    data = scipy.io.loadmat(filename)
    return data['X'], data['Y']

def train_and_predict(AMRC_obj: reAMRC) -> None:
    T = AMRC_obj.n_sample
    for t in range(T-1):
        # In each iteration, we fit by data t, and test on data (t+1)
        AMRC_obj.fit()
        # print(AMRC_obj.varphi)
        if AMRC_obj.next_label() != AMRC_obj.predict(deterministic=True):
            AMRC_obj.mistakes_idx_det[AMRC_obj.t-1] = 1 # obj.t-1 because obj.t=t+1 now.
            AMRC_obj.mistakes_det += 1
        if AMRC_obj.next_label() != AMRC_obj.predict(deterministic=False):
            AMRC_obj.mistakes_idx_rnd[AMRC_obj.t-1] = 1
            AMRC_obj.mistakes_rnd += 1
    return

##########################################################

if __name__ == "__main__":
    print("hello world!")
    filename = "usenet1"
    X, Y = data_reader("./data/" + filename + ".mat")
    myAMRC = reAMRC(X, Y, feature_map="RFF", order=1)
    x = X[0,:]
    y = Y[0,:]
    # fvec = myAMRC.feature_vector(x,y)
    train_and_predict(myAMRC)
    print("Mistake rate: det={}, rand={}".format(myAMRC.mistakes_det / (myAMRC.n_sample-1), myAMRC.mistakes_rnd / (myAMRC.n_sample-1)))
    myAMRC.savemat("./results/" + filename + "_" + str(myAMRC.rndseed) + ".mat")
    print("Pause")