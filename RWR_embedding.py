import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def iterate(A, q, c=0.15, epsilon=1e-9, max_iters=100, handles_deadend=True, norm_type=1):
    """
    Perform power iteration for RWR, PPR, or PageRank

    inputs
        A : csr_matrix
            input matrix (for RWR and it variants, it should be row-normalized)
        q : ndarray
            query vector
        c : float
            restart probability
        epsilon : float
            error tolerance for power iteration
        max_iters : int
            maximum number of iterations for power iteration
        handles_deadend : bool
            if true, it will handle the deadend issue in power iteration
            otherwise, it won't, i.e., no guarantee for sum of RWR scores
            to be 1 in directed graphs
        norm_type : int
            type of norm used in measuring residual at each iteration
    outputs
        x : ndarray
            result vector
    """
    x = q # q: one-hot vector
    old_x = q
    residuals = np.zeros(max_iters)

    pbar = tqdm(total=max_iters)
    for i in range(max_iters):
        if handles_deadend:
            x = (1 - c) * (A.dot(old_x))
            S = np.sum(x)
            x = x + (1 - S) * q
        else:
            x = (1 - c) * (A.dot(old_x)) + (c * q)

        residuals[i] = norm(x - old_x, norm_type)
        pbar.set_description("Residual at %d-iter: %e" % (i, residuals[i]))

        if residuals[i] <= epsilon:
            pbar.set_description("The iteration has converged at %d-iter" % (i))
            #  pbar.update(max_iters)
            break

        old_x = x
        pbar.update(1)

    pbar.close()

    return x, residuals[0:i + 1]


from scipy.sparse import spdiags
def row_normalize(A):
    '''
    Perform row-normalization of the given matrix

    inputs
        A : csr_matrix
            (n x n) input matrix where n is # of nodes
    outputs
        nA : csr_matrix
             (n x n) row-normalized matrix
    '''
    n = A.shape[0]

    # do row-wise sum where d is out-degree for each node
    d = A.sum(axis=1)
    d = np.asarray(d).flatten()

    # handle 0 entries in d
    d = np.maximum(d, np.ones(n))
    invd = 1.0 / d

    invD = spdiags(invd, 0, n, n)

    # compute row normalized adjacency matrix by nA = invD * A
    nA = invD.dot(A)

    return nA


from scipy.sparse import csr_matrix, find

def read_directed_graph(X, weighted):
    rows = X[:, 0]
    cols = X[:, 1]
    data = X[:, 2]

    # assume id starts from 0
    n = int(np.amax(X[:, 0:2])) + 1

    # the weights of redundant edges will be summed (by default in csr_matrix)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    if not weighted:
        # no redundant edges are allowed for unweighted graphs
        I, J, K = find(A)
        A = csr_matrix((np.ones(len(K)), (I, J)), shape=A.shape)

    return A


def swap_columns(X):
    # make src_id <= dst_id
    b_idx = X[:, 0] > X[:, 1]
    a_idx = np.logical_not(b_idx)

    B = X[b_idx, :]
    B[:, [0, 1]] = B[:, [1, 0]] # swap columns

    A = X[a_idx, :]

    return np.vstack((A, B))

def read_graph(edgelist, graph_type):
    '''
    Read the graph from the numpy array

    inputs
        edgelist : numpy.array
            edgelist for the graph (edge_index)
        graph_type : str
            type of graph {'directed', 'undirected', 'bipartite'}
    outputs
        A : csr_matrix
            sparse adjacency matrix
        base : int
            base of node ids of the graph
    '''
    X = edgelist.transpose()
    m, n = X.shape
    print(m, n)

    weighted = True
    if n == 2:
        # the graph is unweighted
        X = np.c_[X, np.ones(m)]
        weighted = False
    elif n <= 1 or n >= 4:
        # undefined type
        raise ValueError('Invalid input format')

    base = np.amin(X[:, 0:2])
    min_weight = np.amin(X[:, 2])

    if base < 0:
        raise ValueError('Out of range of node ids: negative base')
    if min_weight < 0:
        raise ValueError('Negative edge weights')

    # make node id start from 0
    X[:, 0:2] = X[:, 0:2] - base

    A = read_directed_graph(X, weighted)

    return A, base.astype(int)


class PyRWR:
    normalized = False

    def __init__(self):
        pass

    def read_graph(self, input_list, graph_type):
        '''

        inputs
            input_list : numpy.array
                edge_index for the graph data
            graph_type : str
                type of graph {'directed', 'undirected', 'bipartite'}
        '''

        self.A, self.base = read_graph(input_list, graph_type)
        self.m, self.n = self.A.shape
        self.node_ids = np.arange(0, self.n) + self.base
        self.normalize()

    def normalize(self):
        '''
        Perform row-normalization of the adjacency matrix
        '''
        if self.normalized is False:
            nA = row_normalize(self.A) #n*n
            self.nAT = nA.T #转置
            self.normalized = True


class RWR(PyRWR):
    def __init__(self):
        pass

    def compute(self, seed, c=0.15, epsilon=1e-6, max_iters=100,
                handles_deadend=True):
        '''
        Compute the RWR score vector w.r.t. the seed node

        inputs
            seed : int
                seed (query) node id
            c : float
                restart probability
            epsilon : float
                error tolerance for power iteration
            max_iters : int
                maximum number of iterations for power iteration
            handles_deadend : bool
                if true, it will handle the deadend issue in power iteration
                otherwise, it won't, i.e., no guarantee for sum of RWR scores
                to be 1 in directed graphs
        outputs
            r : ndarray
                RWR score vector
        '''

        self.normalize()

        # adjust range of seed node id
        seed = seed - self.base

        #  q = np.zeros((self.n, 1))
        q = np.zeros(self.n)
        if seed < 0 or seed >= self.n:
            raise ValueError('Out of range of seed node id')

        q[seed] = 1.0

        r, residuals = iterate(self.nAT, q, c, epsilon, max_iters, handles_deadend)

        return r


def split_data(grd_truth, ratio):
    grd_truth = grd_truth.transpose()
    total = len(grd_truth)
    np.random.shuffle(grd_truth)
    train = grd_truth[0:int(total*ratio)]
    test = grd_truth[int(total*ratio):total]

    seed1 = []
    seed2 = []
    for pair in train:
        seed1.append(pair[0])
        seed2.append(pair[1])

    return seed1, seed2


def get_candi_seed(g1_feat, g2_feat):
    sim = cosine_similarity(g1_feat, g2_feat)
    pair_set = []
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            pair_set.append([sim[i][j], i, j])
    sorted(pair_set, key=(lambda x: x[0]), reverse=True)
    seed_num = int(0.05*min(sim.shape[0], sim.shape[1]))
    seed = [[pair_set[i][1], pair_set[i][2]] for i in range(seed_num)]
    seed1 = []
    seed2 = []

    for pair in seed:
        seed1.append(pair[0])
        seed2.append(pair[1])

    return seed1, seed2


def rwr_emd(edgelist_1, edgelist_2, seed_1, seed_2, directed=True):
    rwr1 = RWR()
    rwr2 = RWR()
    if directed:
        rwr1.read_graph(edgelist_1, "directed")
        rwr2.read_graph(edgelist_2, "directed")
    else:
        raise ValueError('Should be directed graph!')
    g1_rwr = []
    g2_rwr = []
    for anchor in seed_1:
        emd = rwr1.compute(anchor)
        g1_rwr.append(list(emd)) #anchor_num*n
    for anchor in seed_2:
        emd = rwr2.compute(anchor)
        g2_rwr.append(list(emd))
    g1_rwr_emd = np.array(g1_rwr, dtype=float).T
    g2_rwr_emd = np.array(g2_rwr, dtype=float).T

    return g1_rwr_emd, g2_rwr_emd