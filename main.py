import os
import os.path as osp
import MLdataset
import argparse
import time
from model_new_mulC import MVAE
import utils
from utils import AverageMeter
import evaluation
import torch
import numpy as np
from loss_mulC import Loss
from torch import nn
from torch.optim import AdamW
import copy
import math


def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0.0)


def calculate_loss(loss_fn, pred, label, inc_L_ind, data, rec_v, pred_views, inc_V_ind,
                   shared_mu, shared_logvar, weights, 
                   quality_scores=None):
    total_loss, loss_dict = loss_fn.total_disentangled_loss(
        pred, label, inc_L_ind, data, rec_v, pred_views, inc_V_ind,
        shared_mu, shared_logvar, weights,
        quality_scores=quality_scores
    )
    return total_loss


def _test_first(loader,  model, epoch, logger, mode='train'):
    batch_time = AverageMeter()
    total_labels = []
    total_preds = []

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
            inc_V_ind = inc_V_ind.float().to(device)
            data = [v_data.to(device) for v_data in data]

            model_output = model(data, inc_V_ind)

            pred, _, _, _, _, _, _, _ = model_output

            
            pred = pred.cpu()

            total_labels = np.concatenate((total_labels, label.numpy()), axis=0) if len(
                total_labels) > 0 else label.numpy()
            total_preds = np.concatenate((total_preds, pred.detach().numpy()), axis=0) if len(total_preds) > 0 else \
                pred.detach().numpy()
            batch_time.update(time.time() - end)
            end = time.time()
    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)
    if mode == 'train':
        evaluation_results = [evaluation.compute_average_precision(total_preds, total_labels)]
        logger.info('Epoch:[{0}]\t'
                    'Mode:{mode}\t'
                    'Time {batch_time.avg:.3f}\t'
                    'AP {ap:.4f}\t'.format(
            epoch, mode=mode, batch_time=batch_time,
            ap=evaluation_results[0],
        ))
    else:
        evaluation_results = evaluation.do_metric(total_preds, total_labels)  # compute auc is very slow
        logger.info('Epoch:[{0}]\t'
                    'Mode:{mode}\t'
                    'Time {batch_time.avg:.3f}\t'
                    'AP {ap:.4f}\t'
                    'HL {hl:.4f}\t'
                    'RL {rl:.4f}\t'
                    'AUC {auc:.4f}\t'.format(
            epoch, mode=mode, batch_time=batch_time,
            ap=evaluation_results[0],
            hl=evaluation_results[1],
            rl=evaluation_results[2],
            auc=evaluation_results[3]
        ))
    return evaluation_results


def train_first(loader, model, loss_fn, optimizer, scheduler, epoch, logger):
    losses = AverageMeter()
    model.train()

    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data = [v_data.to(device) for v_data in data]
        label = label.to(device)
        inc_V_ind = inc_V_ind.to(device)
        inc_L_ind = inc_L_ind.to(device)
        
        model_output = model(data, inc_V_ind)
        

        pred, rec_v, shared_mu, shared_logvar, pred_views, _,  weights, quality_scores = model_output
        s_loss_u = calculate_loss(loss_fn, pred, label, inc_L_ind, data, rec_v, pred_views, inc_V_ind,
                                     shared_mu, shared_logvar, weights,
                                     quality_scores=quality_scores)

        
        optimizer.zero_grad()
        s_loss_u.backward()
        optimizer.step()
        loss = s_loss_u
        losses.update(loss.item())

    logger.info('Epoch:[{0}]\t'
                'Loss {losses.avg:.3f}\t'.format(
        epoch, losses=losses))
    return losses, model


def main(args, file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset + '_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset + '_six_view_MaskRatios_' + str(
        args.mask_view_ratio) + '_LabelMaskRatio_' +
                              str(args.mask_label_ratio) + '_TraindataRatio_' +
                              str(args.training_sample_ratio) + '.mat')
    folds_num = args.folds_num
    folds_results = [AverageMeter() for _ in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir, args.name + "main_" + args.dataset + '_V_' + str(
            args.mask_view_ratio) + '_L_' +
                           str(args.mask_label_ratio) + '_T_' +
                           str(args.training_sample_ratio) + '_beta_' +
                           str(args.beta) + '_dropout_' +
                           str(args.dropout) + '_d_model_' +
                           str(args.d_emb) + '.txt')
    else:
        logfile = None
    logger = utils.setLogger(logfile)

    for fold_idx in range(folds_num):
        fold_idx = fold_idx
        train_dataloder, train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,
                                                                    training_ratio=args.training_sample_ratio,
                                                                    fold_idx=fold_idx,
                                                                    mode='train',
                                                                    batch_size=args.batch_size,
                                                                    shuffle=False,
                                                                    num_workers=args.workers)
        test_dataloder, test_dataset = MLdataset.getIncDataloader(data_path,
                                                                  fold_data_path,
                                                                  training_ratio=args.training_sample_ratio,
                                                                  val_ratio=0.15,
                                                                  fold_idx=fold_idx,
                                                                  mode='test',
                                                                  batch_size=args.batch_size,
                                                                  num_workers=args.workers)
        val_dataloder, val_dataset = MLdataset.getIncDataloader(data_path,
                                                                fold_data_path,
                                                                training_ratio=args.training_sample_ratio,
                                                                fold_idx=fold_idx,
                                                                mode='val',
                                                                batch_size=args.batch_size,
                                                                num_workers=args.workers)
        d_list = train_dataset.d_list
        n_cls = train_dataset.classes_num
        model = MVAE(d_list, args.d_emb, n_cls, use_quality_net=True, dropout= args.dropout, n_enc_layer=args.n_enc_layer, n_dec_layer=args.n_dec_layer)
        # model = MVAE(d_list, args.d_emb, args.n_enc_layer, args.n_dec_layer, n_cls, args.beta, args.dropout)
        model = model.to(device)
        initialize(model)
        loss_fn = Loss(args.alpha, args.gamma, args.lammda, args.ev_reg, quality_weight=args.beta).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None

        best_result = 0
        best_epoch = 0
        best_model_dict = {"model": model.state_dict(), "epoch": 0}
        tt=time.time()
        for epoch in range(args.epochs):
            train_losses_first, model = train_first(train_dataloder,  model, loss_fn, optimizer, scheduler, epoch, logger)
            with torch.no_grad():
                # _ = _test_first(train_dataloder, model, epoch, logger, "train")
                val_metric = _test_first(val_dataloder, model, epoch, logger, "test")
            
            val_metric = val_metric[0] * 0.2 + val_metric[1] * 0.2 + val_metric[2] * 0.2 + val_metric[3] * 0.4

            if val_metric > best_result:
                best_result = val_metric
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch = epoch

            if epoch > 20 and (epoch - best_epoch > args.patience):
                print('Training stopped: epoch=%d' % (epoch), "Time:",time.time()-tt)
                break

        model.load_state_dict(best_model_dict['model'])
        test_result_first = _test_first(test_dataloder,  model, -1, logger, mode='test')

        logger.info(
            'final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(
                fold_idx, best_epoch, test_result_first[0], test_result_first[1], test_result_first[2],
                test_result_first[3]))

        for i in range(9):
            folds_results[i].update(test_result_first[i])
        # break

    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP 1-HL 1-RL AUCme 1-OE 1-Cov macAUC macro_f1 micro_f1 alpha beta gamma lammda\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg, 4)) + '+' + str(round(res.std, 4)) for res in folds_results]
    res_list.extend([str(args.alpha), str(args.beta), str(args.gamma), str(args.lammda)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write(' ' + os.path.basename(__file__))
    file_handle.write('\n')
    file_handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=True, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'records'))
    parser.add_argument('--root-dir', type=str, metavar='PATH', default='./data/')
    parser.add_argument('--datasets', type=list, default=['corel5k']) #mirflickr iaprtc12
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--weights_dir', type=str, metavar='PATH', default=osp.join(working_dir, 'weights'))
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--name', type=str, default='final_cRE_')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--d_emb', type=int, default=1024)
    parser.add_argument('--n_enc_layer', type=int, default=1)
    parser.add_argument('--n_dec_layer', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=5)
    parser.add_argument('--quality_weight', type=float, default=0.5, help='Quality supervision loss weight')
    parser.add_argument('--ev_reg', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--use_soft_pseudo_labels', action='store_true')

    args = parser.parse_args()
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    assert torch.cuda.is_available()
    device = torch.device('cuda:2')

    lr_list = [0.0005]
    alpha_list = [10] #For all datasets: best alpha, beta, gamma, lambda: [10 0.5 1 0.1]
    beta_list = [0.5]
    gamma_list = [1]
    lammda_list = [1e-1]
    d_emb_list = [768]
    # quality_weight_list = [0.5]

    missing_view_list = [0.5]
    missing_label_list = [0.5]
    training_ratio_list = [0.7]

    args.datasets = ['corel5k']  # corel5k pascal07 espgame mirflickr iaprtc12

    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for lammda in lammda_list:
                        args.lammda = lammda
                        for d_emb in d_emb_list:
                            args.d_emb = d_emb
                            for missing_view in missing_view_list:
                                args.mask_view_ratio = missing_view
                                for missing_label in missing_label_list:
                                    args.mask_label_ratio = missing_label
                                    for training_ratio in training_ratio_list:
                                        args.training_sample_ratio = training_ratio
                                        for dataset in args.datasets:
                                            args.dataset = dataset
                                            file_path = osp.join(args.records_dir,
                                                                    args.name + args.dataset + '_ViewMask_' + str(
                                                                        args.mask_view_ratio) + '_LabelMask_' + str(
                                                                        args.mask_label_ratio) + '_Training_' +
                                                                    str(args.training_sample_ratio) + '.txt')
                                            main(args, file_path)
