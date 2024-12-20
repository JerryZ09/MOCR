import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import os
import csv
import utils
import load_data
import model
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATASET_PATH = "/zjr/gs1/MOCR"
os.chdir(os.path.dirname(__file__))
aesmodelname = 'aes.pth'

if __name__ == "__main__":
    # parameter
    conf = dict()
    conf['dataset'] = "skcm"
    conf['gamma'] = 200
    conf['lmbd'] = 0.9
    conf['hid_dims'] = [1024, 1024, 1024]
    conf['out_dims'] = 1024
    conf['total_iters'] = 1200
    conf['save_iters'] = 800
    conf['eval_iters'] = 200
    conf['lr'] = 1e-4
    conf['lr_min'] = 0.0
    conf['non_zeros'] = 1000
    conf['n_neighbors'] = 3
    conf['spectral_dim'] = 15
    conf['affinity'] = "nearest_neighbor"
    conf['train_batch_size'] = 100
    conf['view'] = 3
    conf['AEs_total_iters'] = 600
    conf['AEs_lr'] = 1e-3
    conf['AEs_batch_size'] = 128

    if conf['dataset'] == "aml":
        conf['batch_size'] = 8
        conf['chunk_size'] = 23
        conf['total_iters'] = 600
        conf['save_iters'] = 200
        conf['eval_iters'] = 50
        conf['lmbd'] = 1
        conf['AEs_total_iters'] = 400

    if conf['dataset'] == "brca":
        conf['batch_size'] = 2
        conf['chunk_size'] = 547
        conf['total_iters'] = 500
        conf['save_iters'] = 500
        conf['eval_iters'] = 500
        conf['lmbd'] = 0.001
        conf['AEs_total_iters'] = 1600

    if conf['dataset'] == "skcm":
        conf['batch_size'] = 1
        conf['chunk_size'] = 463
        conf['total_iters'] = 1200
        conf['save_iters'] = 800
        conf['eval_iters'] = 200
        conf['lmbd'] = 0.0009
        conf['AEs_total_iters'] = 600

    if conf['dataset'] == "lihc":
        conf['batch_size'] = 1
        conf['chunk_size'] = 373
        conf['total_iters'] = 300
        conf['save_iters'] = 300
        conf['eval_iters'] = 150
        conf['lmbd'] = 0.03
        conf['AEs_total_iters'] = 400

    if conf['dataset'] == "coad":
        conf['batch_size'] = 5
        conf['chunk_size'] = 59
        conf['total_iters'] = 800
        conf['save_iters'] = 400
        conf['eval_iters'] = 100
        conf['lmbd'] = 0.91
        conf['AEs_total_iters'] = 400

    if conf['dataset'] == "kirc":
        conf['batch_size'] = 13
        conf['chunk_size'] = 41
        conf['total_iters'] = 1500
        conf['save_iters'] = 1000
        conf['eval_iters'] = 250
        conf['gamma'] = 210.0
        conf['lmbd'] = 0.01
        conf['AEs_total_iters'] = 800

    if conf['dataset'] == "gbm":
        conf['batch_size'] = 1
        conf['chunk_size'] = 571
        conf['total_iters'] = 2000
        conf['save_iters'] = 1000
        conf['eval_iters'] = 250
        conf['lmbd'] = 1.1
        conf['AEs_total_iters'] = 800

    if conf['dataset'] == "ov":
        conf['batch_size'] = 1
        conf['chunk_size'] = 593
        conf['total_iters'] = 2000
        conf['save_iters'] = 1000
        conf['eval_iters'] = 250
        conf['lmbd'] = 0.1
        conf['AEs_total_iters'] = 800

    if conf['dataset'] == "lusc":
        conf['batch_size'] = 11
        conf['chunk_size'] = 45
        conf['total_iters'] =1200
        conf['save_iters'] = 800
        conf['eval_iters'] = 200
        conf['gamma'] = 400.0
        conf['lmbd'] = 0.001
        conf['AEs_total_iters'] = 600

    if conf['dataset'] == "sarc":
        conf['batch_size'] = 5
        conf['chunk_size'] = 53
        conf['total_iters'] = 300
        conf['save_iters'] = 300
        conf['eval_iters'] = 100
        conf['lmbd'] = 0.011
        conf['AEs_total_iters'] = 400

    seed = 123456
    model.setup_seed(seed)
    tic = time.time()
    # >>>>>>>>>>>>>>>>>>>>>>>>>print now>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    fit_msg = "Experiments on {}, total_iters={}, lambda={}, gamma={}".format(conf['dataset'],
                                                                              conf['total_iters'],
                                                                              conf['lmbd'],
                                                                              conf['gamma'])
    print(fit_msg)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>Save result>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    folder = "result/{}_result".format(conf['dataset'])
    if not os.path.exists(folder):
        os.makedirs(folder)

    result = open("{}/{}_{}_{}.csv".format(folder, conf['dataset'],conf['gamma'],conf['lmbd']), 'w+')
    writer = csv.writer(result)
    writer.writerow(['p', 'logp', 'log10p', 'epoch', 'step'])
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>load data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    cancer_type = conf['dataset']
    exp_df, methy_df, mirna_df, survival = load_data.load_TCGA(DATASET_PATH, cancer_type,
                                                               'knn_2')  # Preprocessing method
    exp_raw = torch.from_numpy(exp_df.values).float().T.cuda()
    # print(exp_raw.shape)
    methy_raw = torch.from_numpy(methy_df.values).float().T.cuda()
    mirna_raw = torch.from_numpy(mirna_df.values).float().T.cuda()
    X = [exp_raw, methy_raw, mirna_raw]
    # >>>>>>>>>>>>>>>>>>>>>>>>>>AEsModel train and evaluate>>>>>>>>>>>>>>>>>>>>>>>>
    input_dim = [exp_raw.shape[1], methy_raw.shape[1], mirna_raw.shape[1]]
    hidden_dim = [700, 600, 500]
    # initialize AEsmodel
    AEsModel = model.AEs(conf['view'], input_dim, hidden_dim).cuda()
    optimizer = optim.Adam(AEsModel.parameters(), lr=conf['AEs_lr'])
    loss_MSE = torch.nn.MSELoss(reduction='sum')
    AEs_iter_per_epoch = exp_raw.shape[0] // conf['AEs_batch_size']
    AEs_epochs = conf['AEs_total_iters'] // AEs_iter_per_epoch

    # train
    pbar = tqdm(range(AEs_epochs), ncols=120)
    print("\nAutoEncoders is training......")
    for epoch in pbar:
        train_loss = 0
        pbar.set_description(f"Epoch {epoch}")
        AEsModel.train()
        for i in range(AEs_iter_per_epoch):
            re_X, _ = AEsModel(X)
            loss_train = 0.0
            for v in range(conf['view']):
                loss_train += loss_MSE(re_X[v], X[v])
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
        # evaluate
        AEsModel.eval()
        re_X, _ = AEsModel(X)
        loss_test = 0.0
        for v in range(conf['view']):
            loss_test += loss_MSE(re_X[v], X[v])
        pbar.set_postfix(loss="{:3.4f}".format(loss_test.item()))
        # save model
        torch.save(AEsModel.state_dict(), aesmodelname)

    # data process
    _, Zv =AEsModel(X)
    exp, methy, mirna = Zv
    full_samples = torch.concat([exp, methy, mirna], dim=1)
    full_data = utils.p_normalize(full_samples)
    # >>>>>>>>>>>>>>>>>>train and initialize>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for N in [128]:
        block_size = min(N, 10000)
        view1 = utils.p_normalize(exp)
        view2 = utils.p_normalize(methy)
        view3 = utils.p_normalize(mirna)
        X = [view1, view2, view3]
        all_samples, ambient_dim = full_samples.shape[0], exp.shape[1]
        n_iter_per_epoch = full_samples.shape[0] // conf['train_batch_size']
        n_step_per_iter = round(all_samples // block_size)
        n_epochs = conf['total_iters'] // n_iter_per_epoch
        # initialize the model
        senet = model.SENet(ambient_dim, conf['hid_dims'], conf['out_dims'], kaiming_init=True).cuda()
        optimizer = optim.Adam(senet.parameters(), lr=conf['lr'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=conf['lr_min'])

        # train
        n_iters = 0
        pbar = tqdm(range(n_epochs), ncols=120)
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            randidx = torch.randperm(full_data.shape[0])

            for i in range(n_iter_per_epoch):
                senet.train()
                batch_idx = randidx[i * conf['train_batch_size']: (i + 1) * conf['train_batch_size']]

                regs = []
                rec_losses = []
                loss_raw = 0.0
                for v in range(conf['view']):
                    v1 = v
                    v2 = (v+1) % conf['view']
                    reg, rec_loss, loss_temp = model.loss_function(n_step_per_iter, batch_idx, senet, X[v1], X[v2],
                                           block_size, conf['lmbd'], conf['gamma'], conf['train_batch_size'])
                    regs.append(reg)
                    rec_losses.append(rec_loss)
                    loss_raw = loss_raw + loss_temp
                loss = loss_raw/conf['view']
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(senet.parameters(), 0.001)
                optimizer.step()

                n_iters += 1
                if n_iters % conf['save_iters'] == 0:
                    with open('{}/SENet_{}_N{:d}_iter{:d}.pth.tar'.format(folder, conf['dataset'], N, n_iters), 'wb') as f:
                        torch.save(senet.state_dict(), f)
                    print("Model Saved.")

                if n_iters % conf['eval_iters'] == 0:
                    print("\nING——Evaluating on {}-full...".format(conf['dataset']))
                    # 50epoch-evaluate
                    res, max_log, max_label, df, clusternum = model.evaluate(senet, data=full_data, exp=view1, methy=view2, mirna=view3,
                                                        survival=survival,
                                                        affinity=conf['affinity'],
                                                        spectral_dim=conf['spectral_dim'], non_zeros=conf['non_zeros'],
                                                        n_neighbors=conf['n_neighbors'],
                                                        batch_size=conf['batch_size'],
                                                        chunk_size=conf['chunk_size'], knn_mode='symmetric')
                    writer.writerow([res['p'], res['log2p'], res['log10p'], epoch, "pre"])
                    result.flush()
                    print("{}:    ING_max_log:   {:.1f}".format(conf['dataset'], max_log))

            pbar.set_postfix(loss="{:3.4f}".format(loss.item()),
                             reg_exp="{:3.4f}".format(regs[0].item() / conf['train_batch_size']),
                             rec_loss_exp="{:3.4f}".format(rec_losses[0].item() / conf['train_batch_size']))
            scheduler.step()

        print("Evaluating on {}-full...".format(conf['dataset']))
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> evaluate >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        res, max_log, max_label, df, clusternum = model.evaluate(senet, data=full_data, exp=view1, methy=view2, mirna=view3,
                                            survival=survival,
                                            affinity=conf['affinity'],
                                            spectral_dim=conf['spectral_dim'], non_zeros=conf['non_zeros'],
                                            n_neighbors=conf['n_neighbors'],
                                            batch_size=conf['batch_size'],
                                            chunk_size=conf['chunk_size'], knn_mode='symmetric')
        writer.writerow([res['p'], res['log2p'], res['log10p'], epoch, "pre"])
        result.flush()
        utils.lifeline_analysis(df, conf['dataset'])

        print("{}:    max_log:   {:.2f}     clusternum:   {:}".format(conf['dataset'], max_log, clusternum))
        torch.cuda.empty_cache()
    result.close()

