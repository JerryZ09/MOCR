import numpy as np
import torch
import os
from sklearn import cluster
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array, check_symmetric
from scipy.linalg import orth
import scipy.sparse as sparse
from munkres import Munkres
from lifelines.statistics import multivariate_logrank_test
import math
from scipy.stats import kruskal, chi2_contingency
import pandas as pd
import difflib
import re
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from lifelines.utils import median_survival_times


from sklearn.metrics import adjusted_rand_score, accuracy_score, normalized_mutual_info_score, pair_confusion_matrix
from scipy.optimize import linear_sum_assignment

def regularizer_pnorm(c, p):
    return torch.pow(torch.abs(c), p).sum()


def sklearn_predict(A, n_clusters):
    spec = cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    res = spec.fit_predict(A)
    return res


def accuracy(pred, labels):
    err = err_rate(labels, pred)
    acc = 1 - err
    return acc


def subspace_preserving_error(A, labels, n_clusters):
    one_hot_labels = torch.zeros([A.shape[0], n_clusters])
    for i in range(A.shape[0]):
        one_hot_labels[i][labels[i]] = 1.0
    mask = one_hot_labels.matmul(one_hot_labels.T)
    l1_norm = torch.norm(A, p=1, dim=1)
    masked_l1_norm = torch.norm(mask * A, p=1, dim=1)
    e = torch.mean((1. - masked_l1_norm / l1_norm)) * 100.
    return e


def normalized_laplacian(A):
    D = torch.sum(A, dim=1)
    D_sqrt = torch.diag(1.0 / torch.sqrt(D))
    L = torch.eye(A.shape[0]) - D_sqrt.matmul(A).matmul(D_sqrt)
    return L


def connectivity(A, labels, n_clusters):
    c = []
    for i in range(n_clusters):
        A_i = A[labels == i][:, labels == i]
        L_i = normalized_laplacian(A_i)
        eig_vals, _ = torch.symeig(L_i)
        c.append(eig_vals[1])
    return np.min(c)


def topK(A, k, sym=True):
    val, indicies = torch.topk(A, dim=1, k=k)
    Coef = torch.zeros_like(A).scatter_(1, indicies, val)
    if sym:
        Coef = (Coef + Coef.t()) / 2.0
    return Coef


def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, noise_level=0.0):
    data = np.empty((num_points_per_subspace * num_subspaces, ambient_dim))
    label = np.empty(num_points_per_subspace * num_subspaces, dtype=int)

    for i in range(num_subspaces):
        basis = np.random.normal(size=(ambient_dim, subspace_dim))
        basis = orth(basis)
        coeff = np.random.normal(size=(subspace_dim, num_points_per_subspace))
        coeff = normalize(coeff, norm='l2', axis=0, copy=False)
        data_per_subspace = np.matmul(basis, coeff).T

        base_index = i * num_points_per_subspace
        data[(0 + base_index):(num_points_per_subspace + base_index), :] = data_per_subspace
        label[0 + base_index:num_points_per_subspace + base_index, ] = i

    data += np.random.normal(size=(num_points_per_subspace * num_subspaces, ambient_dim)) * noise_level

    return data, label


def dim_reduction(X, dim):
    if dim == 0:
        return X

    w, v = np.linalg.eigh(X.T @ X)

    return X @ v[:, -dim:]


def p_normalize(x, p=2):  # p:指定的范数。 dim:指定在哪个维度进行，如果不指定，则是在所有维度进行计算。keepdim:True or False，如果True，则保留dim指定的维度，False则不保留。
    return x / (torch.norm(x, p=p, dim=1, keepdim=True) + 1e-6)  # （p = 2，dim = 1）每行的每一列数据进行2范数运算


def minmax_normalize(x, p=2):
    rmax, _ = torch.max(x, dim=1, keepdim=True)
    rmin, _ = torch.min(x, dim=1, keepdim=True)
    x = (x - rmin) / (rmax - rmin)
    return x


def spectral_clustering(affinity_matrix_, n_clusters, k, seed=1, n_init=20):
    affinity_matrix_ = check_symmetric(affinity_matrix_)
    random_state = check_random_state(seed)

    laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian,
                                 k=k, sigma=None, which='LA')
    embedding = normalize(vec)
    _, labels_, _ = cluster.k_means(embedding, n_clusters,
                                    random_state=seed, n_init=n_init)
    return labels_

def log_rank(df):
    '''
    :param df: 传入生存数据
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :return: res 包含了p log2p log10p
    '''
    res = dict()
    results = multivariate_logrank_test(df['Survival'], df['label'], df['Death'])
    res['p'] = results.summary['p'].item()
    res['log10p'] = -math.log10(results.summary['p'].item())
    res['log2p'] = -math.log2(results.summary['p'].item())
    return res

# 富集分析
def clinical_enrichement(label,clinical):
    cnt = 0
    # age 连续 使用KW检验
    # print(label,clinical)
    stat, p_value_age = kruskal(np.array(clinical["age"]), np.array(label))
    if p_value_age < 0.05:
        cnt += 1
        # print("---age---")
    # 其余离散 卡方检验
    stat_names = ["gender","pathologic_T","pathologic_M","pathologic_N","pathologic_stage"]
    for stat_name in stat_names:
        if stat_name in clinical:
            c_table = pd.crosstab(clinical[stat_name],label,margins = True)
            stat, p_value_other, dof, expected = chi2_contingency(c_table)
            if p_value_other < 0.05:
                cnt += 1
                # print(f"---{stat_name}---")
    return cnt

def get_clinical(path,survival,cancer_type):
    clinical = pd.read_csv(f"{path}/{cancer_type}",sep="\t")
    if cancer_type == 'kirc':
        replace = {'gender.demographic': 'gender','submitter_id.samples': 'sampleID'}
        clinical = clinical.rename(columns=replace)  # 为某个 index 单独修改名称
        clinical["sampleID"] = [re.sub("A", "", x) for x in clinical["sampleID"].str.upper()]
    clinical["sampleID"] = [re.sub("-", ".", x) for x in clinical["sampleID"].str.upper()]
    survival['age'] = pd.NA # 初始化年龄
    survival['gender'] = pd.NA # 初始化年龄
    if 'pathologic_T' in clinical.columns:
        survival['T'] = pd.NA # 初始化年龄
    if 'pathologic_M' in clinical.columns:
        survival['M'] = pd.NA # 初始化年龄
    if 'pathologic_N' in clinical.columns:
        survival['N'] = pd.NA # 初始化年龄
    if 'tumor_stage.diagnoses' in clinical.columns:
        survival['stage'] = pd.NA # 初始化年龄
    i = 0
    # 找对应的参数
    for name in survival['PatientID']:
        # print(name)
        flag = difflib.get_close_matches(name,list(clinical["sampleID"]),1,cutoff=0.6)
        if flag:
            idx = list(clinical["sampleID"]).index(flag[0])
            survival['age'][i] = clinical['age_at_initial_pathologic_diagnosis'][idx]
            survival['gender'][i] = clinical['gender'][idx]
            if 'pathologic_T' in clinical.columns:
                survival['T'][i] = clinical['pathologic_T'][idx]
            if 'pathologic_M' in clinical.columns:
                survival['M'][i] = clinical['pathologic_M'][idx]
            if 'pathologic_N' in clinical.columns:
                survival['N'][i] = clinical['pathologic_N'][idx]
            if 'tumor_stage.diagnoses' in clinical.columns:
                survival['stage'][i] = clinical['tumor_stage.diagnoses'][idx]
        else: print(name)
        i = i + 1
    return survival.dropna(axis=0, how='any')

def lifeline_analysis(df, cancer):
    '''
    :param df:
    生存分析画图，传入参数为df是一个DataFrame
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :param title_g: 图标题
    :return:
    '''
    n_groups = len(set(df["label"]))
    kmf = KaplanMeierFitter()
    plt.figure()
    for group in range(n_groups):
        idx = (df["label"] == group)
        kmf.fit(df['Survival'][idx], df['Death'][idx], label='class_' + str(group))

        ax = kmf.plot()
        plt.title(cancer)
        plt.xlabel("lifeline(days)")
        plt.ylabel("survival probability")
        #用来计算中位生存时间的置信区间，这在医疗分析中很重要
        # treatment_median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
    plt.show()


def get_n_clusters(arr, n_clusters=range(2, 6)):
    #print(arr)
    """
    Finds optimal number of clusters in `arr` via eigengap method

    Parameters
    ----------
    arr : (N, N) array_like
        Input array (e.g., the output of :py:func`snf.compute.snf`)
    n_clusters : array_like
        Numbers of clusters to choose between

    Returns
    -------
    opt_cluster : int
        Optimal number of clusters
    second_opt_cluster : int
        Second best number of clusters
    """

    # confirm inputs are appropriate
    n_clusters = check_array(n_clusters, ensure_2d=False)
    n_clusters = n_clusters[n_clusters > 1]
    #print(n_clusters)
    # don't overwrite provided array!
    graph = arr.copy()

    graph = (graph + graph.T) / 2
    graph[np.diag_indices_from(graph)] = 0
    degree = graph.sum(axis=1)
    #print(graph)
    #print(degree)
    degree[np.isclose(degree, 0)] += np.spacing(1)
    di = np.diag(1 / np.sqrt(degree))
    laplacian = di @ (np.diag(degree) - graph) @ di

    # perform eigendecomposition and find eigengap
    eigs = np.sort(np.linalg.eig(laplacian)[0])
    eigengap = np.abs(np.diff(eigs))
    eigengap = eigengap * (1 - eigs[:-1]) / (1 - eigs[1:])
    n = eigengap[n_clusters - 1].argsort()[::-1]

    return n_clusters[n[:2]]


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta ** 2) * (p * r / ((beta ** 2) * p + r))
    return ri, ari, f_beta

def cluster_evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f_measure = get_rand_index_and_f_measure(label,pred)[2]
    return nmi, ari, acc, pur,f_measure