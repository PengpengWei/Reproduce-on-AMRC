"""

To use hoeffding tree classifier, need to install skmultiflow first:
    $  conda install -c conda-forge scikit-multiflow

See https://scikit-multiflow.readthedocs.io/en/stable/installation.html for details.

"""

from skmultiflow.trees import HoeffdingTreeClassifier
import numpy as np
import scipy.io 

def dwm_hoeffding(X, Y, n_classes, n_chunks, beta, thresh):
    """
    Dynamic Weight Majority: Using Hoeffding Tree

    Input:
    ---
    :X: Instances. n by d. X.dtype=='float64' or 'int64'

    :Y: Labels.

    :n_chunks: # rounds of expert removal, creation and weight update

    :beta: factor for decreasing weights, 0 \leq beta < 1

    Output:
    ---
    :preds: predicting results

    :mistakes: a 0-1 list indicating whether the prediction is wrong.

    """
    n_samples = X.shape[0]
    p = n_samples // n_chunks # period between expert removal, creation and weight update
    # X = X.astype('float64')
    Y = Y.reshape(-1,1).astype('int64')

    # Output
    preds = [0] * n_samples

    # Initialization
    experts = []
    weights = []
    experts.append(HoeffdingTreeClassifier())
    weights.append(1)

    for i in range(n_samples):
        pred_criterion = [0] * n_classes
        for  j, expert in enumerate(experts):
            cur_pred = expert.predict(X[i].reshape(1,-1))[0]
            if cur_pred != Y[i][0] and i % p == 0:
                weights[j] *= beta
            pred_criterion[cur_pred] += weights[j]
        preds[i] = np.argmax(pred_criterion)

        if i % p == 0:
            w_sum = sum(weights)
            weights = [weight / w_sum if weight > thresh else 0 for weight in weights]
            new_experts = [experts[i] for i in range(len(weights)) if weights[i] > 0]
            new_weights = [weights[i] for i in range(len(weights)) if weights[i] > 0]
            weights, experts = new_weights, new_experts
            if preds[i] != Y[i][0]:
                weights.append(1)
                experts.append(HoeffdingTreeClassifier())
        
        for j in range(len(experts)):
            experts[j] = experts[j].partial_fit(X[i].reshape(1,-1), Y[i])

    mistakes = [1 if preds[k] != Y[k][0] else 0 for k in range(n_samples)]

    return preds, mistakes

if __name__ == '__main__':
    data = scipy.io.loadmat("usenet1.mat")
    X, Y = data['X'].astype('float64'), data['Y']
    preds, mistakes = dwm_hoeffding(X, Y, n_classes=2, n_chunks=10, beta=0.5, thresh=0.001)
    print("error rate={}".format(sum(mistakes)/Y.shape[0]))