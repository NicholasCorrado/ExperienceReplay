import numpy as np

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn

def plot(x, y, label):

    avg = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    N = len(y)
    ci = 1.96 * std / np.sqrt(N) * 1.96
    q05 = avg - ci
    q95 = avg + ci


    plt.plot(x, avg, label=label)
    plt.fill_between(x, q05, q95, alpha=0.2)
    # return fig

def softmax(x):
    x -= np.max(x)
    return np.exp(x)/np.exp(x).sum()

if __name__ == "__main__":
    seaborn.set_theme(style='whitegrid', palette='colorblind')
    plt.figure(figsize=(6,5))

    num_trials = 1
    k = int(1000)
    T = int(10000)
    batch_size = 64
    counts_uniform = np.zeros(shape=(num_trials, T))
    #
    for trial in range(num_trials):
        for t in range(T-k):
            indices = np.random.randint(low=0, high=k+t+1, size=batch_size)
            # counts_uniform[trial, indices] += 1 # bugged! Wrong when indices has duplicate values!
            np.add.at(counts_uniform[trial], indices, 1)


    all_indices = np.arange(T)
    counts_adaptive = np.zeros(shape=(num_trials, T))
    for trial in range(num_trials):
        for t in range(T-k):
            indices = all_indices[:k+t+1]
            desired = batch_size/(k+t+1)*(t+1)
            empirical = counts_adaptive[trial, indices]
            diff = desired - empirical
            # diff[diff < 0] = -100
            # h = (1/(indices[::-1]+1))
            # p = softmax(h)
            p = softmax(diff/batch_size*8)
            indices = np.random.choice(indices, batch_size, p=p)
            np.add.at(counts_adaptive[trial], indices, 1)
            if (t+1) % 1000 == 0:
                print(t)

    # all_indices = np.arange(T)
    # counts_adaptive = np.zeros(shape=(num_trials, T))
    # for trial in range(num_trials):
    #     for t in range(T-k):
    #         indices = all_indices[:k+t+1]
    #         desired = batch_size/(k+t+1)*(t+1)
    #         empirical = counts_adaptive[trial, indices]
    #         diff = desired - empirical
    #         # diff[diff < 0] = -100
    #         p = softmax(diff)
    #         indices = np.random.choice(indices, batch_size, p=p, replace=False)
    #         np.add.at(counts_adaptive[trial], indices, 1)
    #         if (t+1) % 1000 == 0:
    #             print(t)

    # all_indices = np.arange(T)
    # counts_adaptive = np.zeros(shape=(num_trials, T))
    # for trial in range(num_trials):
    #     for t in range(T-k):
    #         desired = batch_size/(k+t+1) * (t+1) # t+1 since this is our desired value for the *next* sample we will draw.
    #         empirical = counts_adaptive[trial, all_indices[:k+t+1]]/(t+1)
    #         diff = desired - empirical
    #         idx = np.argmax(diff)
    #         counts_adaptive[trial, idx] += 1


    # for trial in range(num_trials):
    #     for t in range(T-k):
    #         desired = batch_size/(k+t) * (t+1) # t+1 since this is our desired value for the *next* sample we will draw.
    #         # empirical = counts_adaptive[trial, all_indices[:k+t+1]]/(t+1)
    #         # diff = desired - empirical
    #         # p = softmax(diff)
    #         # indices = np.random.choice(all_indices[:k+t+1], batch_size, p=p)
    #         # counts_adaptive[trial, indices] += 1
    #
    #         indices = np.random.randint(low=0, high=k+t+1, size=batch_size)
    #         counts_adaptive[trial, indices] += 1

    H = np.zeros(T)
    Ht = 0
    for t in range(T-k):
        Ht += 1/(T-t)
        H[T-t-1] += Ht
    H[:k+1] = Ht

    G = np.zeros(T)
    Gt = 0
    gamma = 0.5772
    for t in range(T-k):
        Gt += np.log(T-t) - np.log(T-t-1)
        G[T-t-1] = Gt
    G[:k+1] = Gt

    ts = np.arange(1, T+1)
    plot(ts, counts_uniform, label='Uniform Sampling')
    plt.plot(ts, batch_size*H, label='Uniform Sampling (Expectation)')
    # plt.plot(ts, batch_size*G, label='Uniform Sampling (Expectation Approximation)')
    # plt.axhline(batch_size*H.mean())
    plot(ts, counts_adaptive, label='Adaptive Sampling')
    plt.axhline(batch_size/T * (T-k), xmin=0, xmax=T, label='Adaptive Sampling (Expectation)')
    plt.legend()
    plt.xlabel(r'Transition Index $t$')
    plt.ylabel('# of Updates')
    plt.title('How many times each transition is sampled during policy updates?')
    plt.savefig('sim.png')
    plt.show()