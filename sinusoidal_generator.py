import numpy as np
import scipy.io
import typing
import time
from reAMRC import reAMRC

def sinGen(n_sample: int=10000, mean: float=0.0, sigma: float=2.0, omega: float=0.1, bern: float=0.5, 
            filename: str=None, rndseed: int=round(time.time())):
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
    np.random.seed(rndseed)
    epsilon = np.random.randn(2, )
    Y = (np.random.rand(n_sample,1) > bern).astype('int64')

    t = np.arange(start=1, stop=n_sample+1, step=1).reshape(-1,1)
    inside_sin = np.pi * ((np.cos(omega * t) - 3) / 2 + (Y + 1))
    X = np.hstack([4 * np.cos(inside_sin) + epsilon[0].item(), 4 * np.sin(inside_sin) + epsilon[1].item()])

    if filename:
        mdict = {'X': X, 'Y': Y, 'rndseed': rndseed}
        scipy.io.savemat(filename, mdict)
    return X, Y

if __name__ == '__main__':
    sinGen()
    # sinGen(filename='test')