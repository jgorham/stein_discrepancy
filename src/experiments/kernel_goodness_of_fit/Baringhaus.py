from stat_test.quadratic_time import GaussianQuadraticTest, QuadraticMultiple2, MultiquadricQuadraticTest


__author__ = 'kcx'
from scipy.spatial.distance import squareform, pdist

import numpy as np
from statsmodels.stats.multitest import multipletests
from stat_test.ar import simulate, simulatepm


# null is Gaussian with id covariance, that is the magic constants here are
# for the id covariance
def baringhaus_stat(samples):
    Y = samples - np.mean(samples)
    n = Y.shape[0]
    d = Y.shape[1]

    R =  squareform(pdist(Y, 'euclidean'))**2
    R2 = np.linalg.norm(Y,axis=1)**2

    T1 = np.sum( np.exp(-0.5 *R))/n
    T2 = -2.0**(1.0-d/2.0)*np.sum(np.exp(-0.25*R2))
    T3 = n*3.0**(-d/2.0)
    return T1+T2+T3





if __name__ == "__main__":

    def grad_log_normal( x):
        return -x

    def run_simulation(sample_size, bootstrap_size=600, average_over=400):

        for d in [2, 5, 10, 15, 20, 25]:
            samples = []
            for i in range(bootstrap_size):
                samples.append(baringhaus_stat(np.random.randn(sample_size, d)))
            samples = np.array(samples)
            pvals_brainghaus = []
            pvals_stein = []
            pvals_imq = []
            for i in range(average_over):
                X = np.random.randn(sample_size, d)
                X[:, 0] += np.random.rand(sample_size)
                # baringhaus p value
                T = baringhaus_stat(X)
                pval = float(len(samples[samples > T])) / bootstrap_size
                pvals_brainghaus.append(pval)
                # gaussian p value
                me = GaussianQuadraticTest(grad_log_normal)
                qm = QuadraticMultiple2(me)
                p = qm.is_from_null(0.1, np.copy(X), 0.5)
                pvals_stein.append(p)
                # IMQ p value
                me2 = MultiquadricQuadraticTest(grad_log_normal, beta=-0.5)
                qm2 = QuadraticMultiple2(me2)
                p2 = qm2.is_from_null(0.1, np.copy(X), 0.5)
                pvals_imq.append(p2)

            print('d :', d)
            pvals_brainghaus = np.array(pvals_brainghaus)
            print('baringhaus :', float(len(pvals_brainghaus[pvals_brainghaus < 0.1])) / average_over)

            pvals_stein = np.array(pvals_stein)
            print('Stein  :', float(len(pvals_stein[pvals_stein < 0.1])) / average_over)
            pvals_imq = np.array(pvals_imq)
            print('IMQ  :', float(len(pvals_imq[pvals_imq < 0.1])) / average_over)

    np.random.seed(23)

    sample_size = 500
    run_simulation(sample_size)
    print("===")
    sample_size = 1000
    run_simulation(sample_size)
