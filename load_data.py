# 读取原始数据
import os
import re

import numpy as np
import pandas as pd

def load_data_mean(path,cancer_type):
    path = os.path.join(path,cancer_type)
    exp = pd.read_csv(os.path.join(path, "exp"), sep = " ")
    methy = pd.read_csv(os.path.join(path, "methy"), sep = " ")
    mirna = pd.read_csv(os.path.join(path, "mirna"), sep = " ")
    survival = pd.read_csv(os.path.join(path, "survival"), sep = "\t")
    survival = survival.dropna(axis=0)
    survival = survival.drop_duplicates()

    if cancer_type.lower() == "ov":  # Unlike other datasets, the names of DNA methylation patients in the ov dataset are '-'.
        methy.columns = [re.sub("-", ".", x) for x in methy.columns.str.upper()]
    name_list = list()
    survival["PatientID"] = [re.sub("-", ".", x) for x in survival["PatientID"].str.upper()]
    if len(survival["PatientID"][survival.index[0]]) <= len('tcga.16.1060'):
        survival["PatientID"] += ".01"

    exp_fill = exp.mean(axis=1)
    methy_fill = methy.mean(axis=1)
    mirna_fill = mirna.mean(axis=1)
    for token in survival["PatientID"]:
        if token[-2] != "0":
            survival.drop(survival[survival["PatientID"] == token].index, inplace=True)
            continue
        name_list.append(token)
        if token not in exp:
            exp[token] = exp_fill
        if token not in methy:
            methy[token] = methy_fill
        if token not in mirna:
            mirna[token] = mirna_fill
    exp = exp[name_list]
    methy = methy[name_list]
    mirna = mirna[name_list]

    return [exp, methy, mirna, survival]


def load_data_knn_mean(path, cancer_type):
    path = os.path.join(path, cancer_type)
    exp = pd.read_csv(os.path.join(path, "exp"), sep=" ")
    methy = pd.read_csv(os.path.join(path, "methy"), sep=" ")
    mirna = pd.read_csv(os.path.join(path, "mirna"), sep=" ")
    survival = pd.read_csv(os.path.join(path, "survival"), sep="\t")
    survival = survival.dropna(axis=0)
    survival = survival.drop_duplicates()

    if cancer_type.lower() == "ov":  # Unlike other datasets, the names of DNA methylation patients in the ov dataset are '-'.
        methy.columns = [re.sub("-", ".", x) for x in methy.columns.str.upper()]
    name_list = list()
    survival["PatientID"] = [re.sub("-", ".", x) for x in survival["PatientID"].str.upper()]
    if len(survival["PatientID"][survival.index[0]]) <= len('tcga.16.1060'):
        survival["PatientID"] += ".01"
    for token in survival["PatientID"]:
        if token[-2] != "0":
            survival.drop(survival[survival["PatientID"] == token].index, inplace=True)
            continue
        name_list.append(token)
        if token not in exp:
            exp[token] = pd.NA
        if token not in methy:
            methy[token] = pd.NA
        if token not in mirna:
            mirna[token] = pd.NA

    exp = exp[name_list]
    methy = methy[name_list]
    mirna = mirna[name_list]

    from sklearn.impute import KNNImputer
    data = pd.concat([exp, methy, mirna], axis=0, join='outer').T
    imputer = KNNImputer(n_neighbors=5)
    data_np = imputer.fit_transform(data)
    data_df = pd.DataFrame(data_np)
    data_df.index = data.index
    data_df.columns = data.columns
    exp = data_df.T.iloc[0:exp.shape[0], :]
    methy = data_df.T.iloc[exp.shape[0]:exp.shape[0] + methy.shape[0], :]
    mirna = data_df.T.iloc[exp.shape[0] + methy.shape[0]:exp.shape[0] + methy.shape[0] + mirna.shape[0], :]

    return [exp, methy, mirna, survival]
def load_data_knn(path, cancer_type):
    path = os.path.join(path, cancer_type)
    exp = pd.read_csv(os.path.join(path, "exp"), sep=" ")
    methy = pd.read_csv(os.path.join(path, "methy"), sep=" ")
    mirna = pd.read_csv(os.path.join(path, "mirna"), sep=" ")
    survival = pd.read_csv(os.path.join(path, "survival"), sep="\t")
    survival = survival.dropna(axis=0)
    survival = survival.drop_duplicates()

    if cancer_type.lower() == "ov":  # Unlike other datasets, the names of DNA methylation patients in the ov dataset are '-'.
        methy.columns = [re.sub("-", ".", x) for x in methy.columns.str.upper()]
    name_list = list()
    survival["PatientID"] = [re.sub("-", ".", x) for x in survival["PatientID"].str.upper()]
    if len(survival["PatientID"][survival.index[0]]) <= len('tcga.16.1060'):
        survival["PatientID"] += ".01"
    for token in survival["PatientID"]:
        if token[-2] != "0":
            survival.drop(survival[survival["PatientID"] == token].index, inplace=True)
            continue
        if (token in exp) | (token in methy) | (token in mirna):
            name_list.append(token)
            if token not in exp:
                exp[token] = pd.NA
            if token not in methy:
                methy[token] = pd.NA
            if token not in mirna:
                mirna[token] = pd.NA
        else:
            survival.drop(survival[survival["PatientID"] == token].index, inplace=True)

    exp = exp[name_list]
    methy = methy[name_list]
    mirna = mirna[name_list]

    from sklearn.impute import KNNImputer
    data = pd.concat([exp, methy, mirna], axis=0, join='outer').T
    # add to erro
    data = data.replace({pd.NA: np.nan})
    # add end
    imputer = KNNImputer(n_neighbors=5)
    data_np = imputer.fit_transform(data)
    data_df = pd.DataFrame(data_np)
    data_df.index = data.index
    data_df.columns = data.columns
    exp = data_df.T.iloc[0:exp.shape[0], :]
    methy = data_df.T.iloc[exp.shape[0]:exp.shape[0] + methy.shape[0], :]
    mirna = data_df.T.iloc[exp.shape[0] + methy.shape[0]:exp.shape[0] + methy.shape[0] + mirna.shape[0], :]

    return [exp, methy, mirna, survival]


def load_TCGA(path, cancer_type, pre_type):
    if pre_type == "mean":
        return load_data_mean(path,cancer_type)
    elif pre_type == "knn_2":
        data = load_data_knn(path,cancer_type)
        print(data[0].shape[1])
        return data
    else:
        print("pre_type error!")