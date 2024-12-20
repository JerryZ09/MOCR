import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
import random
import os

import utils

class MLP(nn.Module):
    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.output_dims = out_dims
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)
        if kaiming_init:
            self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)
        init.zeros_(self.out_layer.bias)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        h = torch.tanh_(h)
        return h


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))

    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class SENet(nn.Module):
    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=True):
        super(SENet, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims

        self.net_q = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.net_k = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.thres = AdaptiveSoftThreshold(1)

    def query_embedding(self, queries):
        q_emb = self.net_q(queries)
        return q_emb

    def key_embedding(self, keys):
        k_emb = self.net_k(keys)
        return k_emb

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def forward(self, queries, keys):
        q = self.query_embedding(queries)
        k = self.key_embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out


def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()


def get_sparse_rep(senet, data, view1, view2, batch_size, chunk_size, non_zeros=1000):
    N, D = data.shape

    non_zeros = min(N, non_zeros)

    C1 = torch.empty([batch_size, N])
    C2 = torch.empty([batch_size, N])
    C3 = torch.empty([batch_size, N])

    if (N % batch_size != 0):
        raise Exception("batch_size should be a factor of dataset size.")
    if (N % chunk_size != 0):
        raise Exception("chunk_size should be a factor of dataset size.")

    val = []
    indicies = []
    with torch.no_grad():
        senet.eval()
        for i in range(data.shape[0] // batch_size):
            chunk1 = view1[i * batch_size:(i + 1) * batch_size].cuda()
            q1 = senet.query_embedding(chunk1)
            for j in range(data.shape[0] // chunk_size):
                chunk_samples1 = view1[j * chunk_size: (j + 1) * chunk_size].cuda()
                k1 = senet.key_embedding(chunk_samples1)
                temp1 = senet.get_coeff(q1, k1)
                C1[:, j * chunk_size:(j + 1) * chunk_size] = temp1.cuda()
            rows1 = list(range(batch_size))
            cols1 = [j + i * batch_size for j in rows1]
            C1[rows1, cols1] = 0.0

            chunk2 = view2[i * batch_size:(i + 1) * batch_size].cuda()
            q2 = senet.query_embedding(chunk2)
            for j in range(data.shape[0] // chunk_size):
                chunk_samples2 = view2[j * chunk_size: (j + 1) * chunk_size].cuda()
                k2 = senet.key_embedding(chunk_samples2)
                temp2 = senet.get_coeff(q2, k2)
                C2[:, j * chunk_size:(j + 1) * chunk_size] = temp2.cuda()
            rows2 = list(range(batch_size))
            cols2 = [j + i * batch_size for j in rows2]
            C2[rows2, cols2] = 0.0

            for j in range(data.shape[0] // chunk_size):
                temp3 = senet.get_coeff(q1, k2)
                C3[:, j * chunk_size:(j + 1) * chunk_size] = temp3.cuda()
            rows3 = list(range(batch_size))
            cols3 = [j + i * batch_size for j in rows3]
            C3[rows3, cols3] = 0.0

            C = (C1 + C2 + C3)/3
            #C = C3

            _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)

            val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())
            index = index.reshape([-1]).cpu().data.numpy()
            indicies.append(index)

    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]

    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse

def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


def evaluate(senet, data, exp, methy, mirna, survival, spectral_dim, non_zeros=1000, n_neighbors=3,
             batch_size = 100, chunk_size = 100, affinity='nearest_neighbor', knn_mode='symmetric'):
    max_log = 0.0

    C_sparse1 = get_sparse_rep(senet=senet, data=data, view1=exp, view2=methy, batch_size=batch_size,
                              chunk_size=chunk_size, non_zeros=non_zeros)
    C_sparse2 = get_sparse_rep(senet=senet, data=data, view1=methy, view2=mirna, batch_size=batch_size,
                               chunk_size=chunk_size, non_zeros=non_zeros)
    C_sparse3 = get_sparse_rep(senet=senet, data=data, view1=mirna, view2=exp, batch_size=batch_size,
                               chunk_size=chunk_size, non_zeros=non_zeros)

    C_sparse = (C_sparse1 + C_sparse2 + C_sparse3)/3
    C_sparse_normalized = normalize(C_sparse).astype(np.float32)

    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")

    normal_arr = Aff.toarray()
    n_clusters = utils.get_n_clusters(normal_arr, range(2, 6))

    preds = utils.spectral_clustering(Aff, n_clusters[0], spectral_dim)
    survival["label"] = np.array(preds)
    df = survival
    res = utils.log_rank(df)
    if (res['log10p'] > max_log):
        max_log = res['log10p']
        max_label = preds

    return res, max_log, max_label, df, n_clusters[0]

def sim_evaluate(senet, data, exp, methy, mirna, label, num_subspaces, spectral_dim, non_zeros=1000, n_neighbors=3,
             batch_size=10000, chunk_size=10000, affinity='nearest_neighbor', knn_mode='symmetric'):
    C_sparse1 = get_sparse_rep(senet=senet, data=data, view1=exp, view2=methy, batch_size=batch_size,
                               chunk_size=chunk_size, non_zeros=non_zeros)
    C_sparse2 = get_sparse_rep(senet=senet, data=data, view1=methy, view2=mirna, batch_size=batch_size,
                               chunk_size=chunk_size, non_zeros=non_zeros)
    C_sparse3 = get_sparse_rep(senet=senet, data=data, view1=mirna, view2=exp, batch_size=batch_size,
                               chunk_size=chunk_size, non_zeros=non_zeros)

    C_sparse = (C_sparse1 + C_sparse2 + C_sparse3) / 3
    C_sparse_normalized = normalize(C_sparse).astype(np.float32)
    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")
    pred = utils.spectral_clustering(Aff, num_subspaces, spectral_dim)
    simres = utils.cluster_evaluate(label, pred)
    return simres



def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loss_function(n_step_per_iter, batch_idx, senet, view1, view2, block_size, lmbd,
                            gamma, batch_size):
    batch1 = view1[batch_idx].cuda()
    # print("batch1")
    # print(batch1.shape)
    q_batch1 = senet.query_embedding(batch1)
    # print("Q")
    # print(q_batch1.shape)
    k_batch1 = senet.key_embedding(batch1)
    # print("K")
    # print(k_batch1.shape)
    rec_batch1 = torch.zeros_like(batch1).cuda()
    reg1 = torch.zeros([1]).cuda()

    batch2 = view2[batch_idx].cuda()
    q_batch2 = senet.query_embedding(batch2)
    k_batch2 = senet.key_embedding(batch2)
    rec_batch2 = torch.zeros_like(batch2).cuda()
    reg2 = torch.zeros([1]).cuda()

    rec_batch12 = torch.zeros_like(batch2).cuda()
    reg12 = torch.zeros([1]).cuda()

    for j in range(n_step_per_iter):
         block1 = view1[j * block_size: (j + 1) * block_size].cuda()
         block2 = view2[j * block_size: (j + 1) * block_size].cuda()
         # view1 self refactor
         k_block1 = senet.key_embedding(block1)
         c1 = senet.get_coeff(q_batch1, k_block1)
         rec_batch1 = rec_batch1 + c1.mm(block1)
         reg1 = reg1 + regularizer(c1, lmbd)

         # view2 self refactor
         k_block2 = senet.key_embedding(block2)
         c2 = senet.get_coeff(q_batch2, k_block2)
         rec_batch2 = rec_batch2 + c2.mm(block2)
         reg2 = reg2 + regularizer(c2, lmbd)

         # view1 refactor view2
         c12 = senet.get_coeff(q_batch1, k_block2)
         rec_batch12 = rec_batch12 + c12.mm(block2)
         reg12 = reg12 + regularizer(c12, lmbd)

    diag_c1 = senet.thres((q_batch1 * k_batch1).sum(dim=1, keepdim=True)) * senet.shrink
    rec_batch1 = rec_batch1 - diag_c1 * batch1
    reg1 = reg1 - regularizer(diag_c1, lmbd)
    rec_loss1 = torch.sum(torch.pow(batch1 - rec_batch1, 2))

    diag_c2 = senet.thres((q_batch2 * k_batch2).sum(dim=1, keepdim=True)) * senet.shrink
    rec_batch2 = rec_batch2 - diag_c2 * batch2
    reg2 = reg2 - regularizer(diag_c2, lmbd)
    rec_loss2 = torch.sum(torch.pow(batch2 - rec_batch2, 2))

    diag_c12 = senet.thres((q_batch1 * k_batch2).sum(dim=1, keepdim=True)) * senet.shrink
    rec_batch12 = rec_batch12 - diag_c12 * batch2
    reg12 = reg12 - regularizer(diag_c12, lmbd)
    rec_loss12 = torch.sum(torch.pow(batch2 - rec_batch12, 2))

    reg = (reg1 + reg2 + reg12) / 3
    rec_loss = (rec_loss1 + rec_loss2 + rec_loss12) / 3
    loss = (0.5 * gamma * rec_loss + reg) / batch_size

    return reg, rec_loss, loss



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim[2], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class AEs(nn.Module):
    def __init__(self, view, input_dim, hidden_dim):
        super(AEs, self).__init__()
        self.view = view # 视图
        self.hidden_dim = hidden_dim
        self.encoders = nn.ModuleList() # encoder列表
        self.decoders = nn.ModuleList() # decoder列表
        self.layer_num = len(self.hidden_dim) - 1
        for v in range(view):
            encoder = Encoder(input_dim[v], hidden_dim)
            self.encoders.append(encoder).cuda()
            decoder = Decoder(input_dim[v], hidden_dim)
            self.decoders.append(decoder).cuda()

    def forward(self, X):
        Xrs = []
        Zv = []
        for v in range(self.view):
            Z = self.encoders[v](X[v])
            Zv.append(Z)
            Xrs.append(self.decoders[v](Z))
        return Xrs, Zv

