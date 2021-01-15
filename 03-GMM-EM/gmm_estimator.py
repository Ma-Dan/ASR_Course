# Author: Sining Sun , Zhanheng Yang

import numpy as np
from utils import *
import scipy.cluster.vq as vq
from scipy.stats import multivariate_normal

num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class GMM:
    def __init__(self, D, K=5):
        assert(D>0)
        self.dim = D
        self.K = K
        #Kmeans Initial
        self.mu , self.sigma , self.pi = self.kmeans_initial()

    def kmeans_initial(self):
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)
        clusters = [[] for i in range(self.K)]
        for (l,d) in zip(labels,data):
            clusters[l].append(d)

        for cluster in clusters:
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=0))
        pi = np.array([len(c)*1.0 / len(data) for c in clusters])
        return mu , sigma , pi
    
    def gaussian(self , x , mu , sigma):
        """Calculate gaussion probability. 和scipy.stats.multivariate_normal计算相同

            :param x: The observed data, dim*1.
            :param mu: The mean vector of gaussian, dim*1
            :param sigma: The covariance matrix, dim*dim
            :return: the gaussion probability, scalor
        """
        D=x.shape[0]
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma + 0.0001)
        mahalanobis = np.dot(np.transpose(x-mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x-mu))
        const = 1/((2*np.pi)**(D/2))
        return const * (det_sigma)**(-0.5) * np.exp(-0.5 * mahalanobis)
    
    def calc_log_likelihood(self , X):
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model 
        """

        log_llh = 0.0
        n_points, n_clusters = len(X), len(self.pi)
        pdfs = np.zeros(((n_points, n_clusters)))
        for i in range(n_clusters):
            pdfs[:, i] = self.pi[i] * multivariate_normal.pdf(X, self.mu[i], self.sigma[i])
        log_llh = np.mean(np.log(pdfs.sum(axis=1)))
        return log_llh

    def update_W(self, X, Mu, Var, Pi):
        n_points, n_clusters = len(X), len(Pi)
        pdfs = np.zeros(((n_points, n_clusters)))
        for i in range(n_clusters):
            pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], Var[i])
        W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return W

    def update_Pi(self, W):
        Pi = W.sum(axis=0) / W.sum()
        return Pi

    def update_Mu(self, X, W):
        n_clusters = W.shape[1]
        Mu = []
        for i in range(n_clusters):
            Mu.append(np.average(X, axis=0, weights=W[:, i]))
        return Mu

    def update_Var(self, X, Mu, W):
        n_clusters = W.shape[1]
        Var = []
        for i in range(n_clusters):
            Var.append(np.cov(X - Mu[i], rowvar=0, aweights=W[:, i]))
        return Var

    def em_estimator(self , X):
        """Update paramters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated model 
        """

        log_llh = 0.0
        W = self.update_W(X, self.mu, self.sigma, self.pi)
        self.pi = self.update_Pi(W)
        self.mu = self.update_Mu(X, W)
        self.sigma = self.update_Var(X, self.mu, W)
        log_llh = self.calc_log_likelihood(X)

        return log_llh


def train(gmms, num_iterations = num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')
    
    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)   #
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
            print('log-likehood:%.3f'%log_llh)
    return gmms

def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian) #Initial model
    gmms = train(gmms)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


if __name__ == '__main__':
    main()
