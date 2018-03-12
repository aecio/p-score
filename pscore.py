
import numpy as np
from operator import itemgetter

class ReputationScorer:

    def __init__(self):
        self.graph = []

    def solve_markov(self, P, n_steps=10000, rand_init=True, epilson=1e-5):
        n = P.shape[0]
        if rand_init:
            p = np.random.rand(n)
            p = p/np.sum(p)
        else:
            p = np.repeat(1.0/n, n)
        for k in range(n_steps):
            p_prev = P.T.dot(p)
            residual = np.sum(np.absolute(p - p_prev))
            if k % 10000 == 0:
                print('residual: {0:.15f} epilson: {1:.15f}'.format(residual, epilson))
            p = p_prev
            if residual < epilson:
                print('Converged with {} iterations with residual {}'.format(k, residual))
                break
        return p

    def source_target_edge(self, source_id, target_id, weight):
        self.graph.append((source_id, target_id, weight))

    def rank(self, n_steps=100000, rand_init=False, epilson=1e-5):
        '''
        Compute reputation scores considering the a bipartite graph of source
        and targets, i.e., does not consider reputation transfer from sources to
        sources, and from target to targets.
        '''
        # compute number of source and target nodes
        n_g = max(self.graph, key=lambda v: v[0])[0]+1 # number of sources
        n_v = max(self.graph, key=lambda v: v[1])[1]+1 # number of targets

        # Build source-target P_st matrix
        P_st = np.zeros((n_g, n_v), np.float64)
        for edge in self.graph:
          P_st[edge[0]][edge[1]] = edge[2]
        for i in range(P_st.shape[0]):
          P_st[i] = P_st[i] / np.sum(P_st[i])

        # Build target-source P_ts matrix
        P_ts = np.zeros((n_v, n_g), np.float64)
        for edge in self.graph:
          P_ts[edge[1]][edge[0]] = edge[2]
        for i in range(P_ts.shape[0]):
          P_ts[i] = P_ts[i] / np.sum(P_ts[i])

        P_prime = np.matmul(P_st, P_ts)

        p_s = self.solve_markov(P_prime, n_steps=n_steps, rand_init=rand_init, epilson=epilson)
        p_t = np.matmul(p_s, P_st)

        return p_t
