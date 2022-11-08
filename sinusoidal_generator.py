import numpy as np
import scipy.io
import typing
import time
from reAMRC import reAMRC

def sinGen(n_sample: int=150, mean: float=0.0, sigma: float=2.0, omega: float=0.1, bern: float=0.5, 
            filename: str=None, rndseed=None):
    """
    Generate a synthetic dataset as described in Section 6 in the paper. Save the data to `filename` if given.

    $$

    x_t = [4 \cos (\pi ((\cos (\omega t) - 3) / 2 + y'_t)) + \epsilon_1, 4 \sin (\pi ((\cos (\omega t) - 3) / 2 + y'_t)) + \epsilon_2]^T

    y_t \sim \text{Bernoulli}(0.5), y'_t = y_t + 1

    \epsilon_1, \epsilon_2 \sim N(mean, sigma)

    $$

    Input:
    ---
    :n_sample: number of samples in total.

    :mean: mean of epsilon 1 and 2

    :sigma: standard deviation of epsilon 1 and 2

    :omega: angular velocity

    :bern: Pr(y_t=0)

    :filename: The filename to store the synthetic dataset

    :rndseed: random seed for the synthetic dataset generation, for reproducibility of experiments. 

    Output:
    ---
    :X: the collection of instances: n_sample by 2 matrix.

    :Y: labels: n_sample by 1 vector.

    Output file: if `filename` != None, X, Y would be saved to filename

    """
    if rndseed != None:
        np.random.seed(rndseed)
    # epsilon = np.random.randn(2, )
    epsilon = mean + sigma * np.random.randn(n_sample, 2)
    Y = (np.random.rand(n_sample,1) > bern).astype('int64')

    t = np.arange(start=1, stop=n_sample+1, step=1).reshape(-1,1)
    inside_sin = np.pi * ((np.cos(omega * t) - 3) / 2 + (Y + 1))
    X = np.hstack([4 * np.cos(inside_sin) , 4 * np.sin(inside_sin)]) + epsilon

    if filename:
        mdict = {'X': X, 'Y': Y}
        if rndseed != None:
            mdict['rndseed'] = rndseed
        scipy.io.savemat(filename, mdict)
    return X, Y

if __name__ == '__main__':
    # sinGen()
    for i in range(1,201):
        sinGen(filename='./temp/test' + str(i) + '.mat')
    # sinGen(filename='test.mat')