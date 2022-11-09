from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import itertools
import numpy as np
import scipy.io
import typing
from reAMRC import reAMRC, data_reader, train_and_predict
from sinusoidal_generator import sinGen
import multiprocessing
import time


mc_time = 120
n_sample = 10000
order_below = 4
def par_operation(iterno):
    save_result = True
    k = iterno // mc_time
    i = iterno % mc_time

    # First generate a synthetic dataset using random seeed iterno.
    # X, Y = sinGen(n_sample=n_sample, rndseed=iterno)
    X, Y = data_reader("./data/synthetic/figure4/" + "test" + str(i) + ".mat")

    myAMRC = reAMRC(X, Y, order=k, feature_map="linear", normalized=False)
    train_and_predict(myAMRC)

    if save_result:
        mdict = {'Rht': myAMRC.Rht, 'RUt': myAMRC.RUt, 'mistakes_idx_det': myAMRC.mistakes_idx_det, 'mistakes_idx_rnd': myAMRC.mistakes_idx_rnd}
        scipy.io.savemat("./results/synthetic/figure4/order" + str(k) + "/test" + str(i) + "result.mat", mdict)

    return myAMRC.Rht, myAMRC.RUt, myAMRC.mistakes_idx_det, myAMRC.mistakes_idx_rnd

if __name__ == "__main__":
    start_time = time.time()
    # mc_time = 100
    # n_sample = 10000

    np.random.seed(553) # This guarantees the same datasets for reproduction, but does not guarantee the same performance
                        # because the behaviors of parallel randomized AMRC cannot be predicted. 
    # The synthetic datasets are generated before parallel computing, otherwise some datasets would be the same...
    for i in range(mc_time):
        sinGen(n_sample=n_sample, filename="./data/synthetic/figure4/" + "test" + str(i) + ".mat")

    cores = multiprocessing.cpu_count()
    pl = multiprocessing.Pool(processes=cores)
    # lst_to_par = [(iterno, n_sample) for iterno in range(mc_time)]
    results = pl.map(par_operation, range(order_below * mc_time))
    pl.close()
    pl.join()


    Rht, RUt = np.zeros((n_sample, mc_time, order_below)), np.zeros((n_sample, mc_time, order_below))
    mistakes_idx_det, mistakes_idx_rnd = np.zeros((n_sample, mc_time, order_below)), np.zeros((n_sample, mc_time, order_below))
    for iterno in range(mc_time * order_below):
        k = iterno // mc_time
        i = iterno % mc_time
        cur_data = results[iterno]
        Rht[:,i,k] = cur_data[0].flatten()
        RUt[:,i,k] = cur_data[1].flatten()
        mistakes_idx_det[:,i,k] = cur_data[2].flatten()
        mistakes_idx_rnd[:,i,k] = cur_data[3].flatten()
    # np.savez("./results/syn_res.npz", Rht=Rht, RUt=RUt, mistakes_idx_det=mistakes_idx_det, mistakes_idx_rnd=mistakes_idx_rnd)
    mdict = {'Rht': Rht, 'RUt': RUt, 'mistakes_idx_det': mistakes_idx_det, 'mistakes_idx_rnd': mistakes_idx_rnd}
    scipy.io.savemat("./results/syn_figure4_res.mat", mdict)
    # print("Average mistake rate (det)={}, (rnd)={}, {} sec spent".format(mistakes_idx_det[1:,:].mean(), mistakes_idx_rnd[1:,:].mean(), time.time()-start_time))
    print("Finish: {} sec".format(time.time()-start_time))