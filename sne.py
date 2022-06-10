import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def binary_search(eval_fn, target, tol=1e-10, max_iter=10000,
                  lower=1e-20, upper=1000.):
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess


def distance_matrix(X):

    # https://stackoverflow.com/questions/23983748/possible-optimizations-for-calculating-squared-euclidean-distance
    #binomial forumla
    sum_X = np.sum(np.square(X), axis=1)
    distances = -2 * np.dot(X, X.T)
    distances = np.add(distances, sum_X)
    distances = np.add(sum_X, distances.T)
    distances = -distances

    return distances

def build_perp_row(dist_row, sigma, row_ind):
    two_sigma_sq = 2. * (sigma**2)

    prob_row = np.exp(dist_row/two_sigma_sq)
    prob_row[:,row_ind] = 0
    prob_row = prob_row / np.sum(prob_row, axis=1)

    entropy = -np.sum(prob_row * np.log2(prob_row))
    perp = 2 ** entropy   

    return perp

def find_optimal_sigmas(distances, target_perplexity=20):
    sigmas = []
    for i in range(distances.shape[0]):
        eval_fn = lambda sigma: build_perp_row(distances[i:i+1, :], sigma, i)
        correct_sigma = binary_search(eval_fn, target_perplexity)
        sigmas.append(correct_sigma)
    return np.array(sigmas)

def build_jointprob_matrix(distances, sigmas):
    two_sigmas_sq = 2. * np.square(sigmas.reshape(-1,1))

    prob_matrix = np.exp((distances / two_sigmas_sq))
    np.fill_diagonal(prob_matrix, 0.)
    prob_matrix_rowsums = np.sum(prob_matrix, axis=1).reshape(-1,1)
    prob_matrix = prob_matrix / prob_matrix_rowsums

    prob_matrix = (prob_matrix + prob_matrix.T) / (2 )

    return prob_matrix

def sne_grad(P, Q, Y, distances):
    pq_diff = P - Q  
    pq_expanded = np.expand_dims(pq_diff, 2)  
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  
    distances_expanded = np.expand_dims(distances, 2) 
    y_diffs_wt = y_diffs * distances_expanded 
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)  
    return grad

def estimate_sne(P, num_iters, learning_rate, grad_fn):
    Y = np.random.randn(P.shape[0], 2)

    for i in range(num_iters):

        distances = np.power(1. - distance_matrix(Y), -1)
        np.fill_diagonal(distances, 0.)
        Q = distances / np.sum(distances,1).reshape(-1,1)

        grads = grad_fn(P, Q, Y, distances)

        Y = Y - learning_rate * grads

    return Y

rows = 30

mnist = pd.read_csv("mnist_test.csv")
mnist = np.array(mnist)
mnist_img = np.array(mnist[:,1:])
mnist_label = np.array(mnist[:,0])

D = distance_matrix(mnist_img[0:rows,:])
sigmas = find_optimal_sigmas(D,10)
P = build_jointprob_matrix(D,sigmas=sigmas)

Y = estimate_sne(P, 1000, 1, sne_grad)

fig, ax = plt.subplots()
ax.scatter(Y[:,0], Y[:,1])

for i, txt in enumerate(mnist_label[0:rows]):
    ax.annotate(txt, (Y[i,0], Y[i,1]))

plt.show()