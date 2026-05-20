import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import time
from adaptive_utils import compute_gradient_divergence, compute_gradient_divergence_batch, compute_distillation_gain, compute_adaptive_lambdas
import wandb
from scipy.stats import spearmanr, pearsonr
from model import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')  
    # ===== 新增：AdaMOON v2 参数 =====
    parser.add_argument('--lambda_min', type=float, default=0.1, 
                        help='AdaMOON: lambda lower bound')
    parser.add_argument('--lambda_max', type=float, default=2.0, 
                        help='AdaMOON: lambda upper bound')
    parser.add_argument('--alpha_blend', type=float, default=0.5, 
                        help='AdaMOON: weight for gradient divergence vs distillation gain')
    parser.add_argument('--adapt_momentum', type=float, default=0.5, 
                        help='AdaMOON: EMA momentum for lambda smoothing')
    parser.add_argument('--warmup_rounds', type=int, default=3, 
                        help='AdaMOON: number of warmup rounds using center lambda')
    parser.add_argument('--n_eval_batches', type=int, default=3, 
                        help='AdaMOON: number of batches for distillation gain evaluation')
    # ===== 新增结束 =====
   
    # ===== wandb 参数 =====
    parser.add_argument('--use_wandb', type=int, default=0, help='whether to use wandb logging (0 or 1)')
    parser.add_argument('--wandb_project', type=str, default='AdaMOON', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (default: log_file_name)')
    # ===== wandb 参数结束 =====

    args = parser.parse_args()
    return args



def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc,_ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix,_ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch % 10 == 0:
            train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    net = nn.DataParallel(net)
    net.cuda()
    # else:
    #     net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    global_weight_collector = list(global_net.cuda().parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda()
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc

def train_net_adamoon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, 
                      epochs, lr, args_optimizer, lambda_i, temperature, args, round, device="cpu"):
    """
    AdaMOON v2 的 client 端训练函数。
    
    与 train_net_fedcon 的唯一区别：
    - 用 per-client 的 lambda_i 替代全局固定的 mu
    - loss = L_CE + lambda_i * L_contrastive（无 0.1 折扣！）
    
    参考方案 E 修复方案：lambda_i 直接就是最终有效系数。
    """
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    for previous_net in previous_nets:
        previous_net.cuda()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            # ===== 核心区别：用 lambda_i 替代 mu =====
            # 注意：这里直接乘 lambda_i，不再有 MOON 原版的隐含 0.1 折扣
            # MOON 原版：loss2 = mu * criterion(logits, labels)，有效系数 = mu（因为外层没有额外折扣）
            # 但 MOON 论文默认 mu=1~10，实际有效系数就是 mu 本身
            # 我们的 lambda_i ∈ [0.005, 0.05]，直接对标 MOON mu=1 时的量级关系
            loss2 = criterion(logits, labels)
            loss1 = criterion(out, target)
            loss = loss1 + lambda_i * loss2
            # ===== 核心区别结束 =====

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append((lambda_i * loss2).item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss_CE: %f Loss_Con(λ*L): %f λ_i: %f' % 
                    (epoch, epoch_loss, epoch_loss1, epoch_loss2, lambda_i))

    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None, 
                    prev_model_pool=None, server_c=None, clients_c=None, round=None, 
                    device="cpu", lambda_dict=None):  # ← 新增参数 lambda_dict
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        if args.alg == 'fedavg':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device=device)
        # ===== 新增：AdaMOON 分支 =====
        elif args.alg == 'adamoon':
            prev_models = []
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            # 从 lambda_dict 获取该 client 的 λ
            client_lambda = lambda_dict[net_id] if lambda_dict is not None else (args.lambda_min + args.lambda_max) / 2
            trainacc, testacc = train_net_adamoon(net_id, net, global_model, prev_models, train_dl_local, test_dl, 
                                                   n_epoch, args.lr, args.optimizer, client_lambda, 
                                                   args.temperature, args, round, device=device)
        # ===== 新增结束 =====
        elif args.alg == 'local_training':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets


def log_adamoon_wandb(round, train_acc, test_acc, train_loss, 
                      party_list_this_round, lambda_dict, d_values, g_values,
                      lambda_emas, args, nets_this_round, global_model, 
                      net_dataidx_map, old_nets_pool, is_warmup=False):
    """
    AdaMOON 增强版 wandb 日志。
    
    记录内容：
    1. 基础训练指标
    2. λ 分布统计（均值、方差、分位数、范围利用率）
    3. 原始信号统计 + rank 后统计
    4. 信号-λ 相关性（Spearman/Pearson）
    5. Loss 分解（CE vs Contrastive 占比）
    6. Per-client 细粒度数据
    7. 数据异质性 vs λ 的关系
    """
    lambda_center = (args.lambda_min + args.lambda_max) / 2
    
    log_dict = {
        # ===== 1. 基础训练指标 =====
        'perf/test_acc': test_acc,
        'perf/train_acc': train_acc,
        'perf/train_loss': train_loss,
        'perf/generalization_gap': train_acc - test_acc,
    }
    
    if is_warmup:
        log_dict.update({
            'lambda/mean': lambda_center,
            'lambda/std': 0.0,
            'lambda/is_warmup': 1,
        })
        wandb.log(log_dict, step=round)
        return
    
    # ===== 2. λ 分布深度统计 =====
    lambda_values = np.array([lambda_dict[k] for k in party_list_this_round])
    lambda_range = args.lambda_max - args.lambda_min
    
    log_dict.update({
        'lambda/mean': np.mean(lambda_values),
        'lambda/std': np.std(lambda_values),
        'lambda/min': np.min(lambda_values),
        'lambda/max': np.max(lambda_values),
        'lambda/median': np.median(lambda_values),
        'lambda/p10': np.percentile(lambda_values, 10),
        'lambda/p25': np.percentile(lambda_values, 25),
        'lambda/p75': np.percentile(lambda_values, 75),
        'lambda/p90': np.percentile(lambda_values, 90),
        'lambda/iqr': np.percentile(lambda_values, 75) - np.percentile(lambda_values, 25),
        # 范围利用率：实际使用的范围 / 设定范围
        'lambda/range_utilization': (np.max(lambda_values) - np.min(lambda_values)) / lambda_range,
        # 偏度：λ 分布是否对称
        'lambda/skewness': float(((lambda_values - np.mean(lambda_values)) ** 3).mean() / 
                                  (np.std(lambda_values) ** 3 + 1e-8)),
        # 相对于中心值的偏移
        'lambda/center_offset': np.mean(lambda_values) - lambda_center,
        'lambda/is_warmup': 0,
    })
    
    # ===== 3. 原始信号统计 =====
    d_arr = np.array(d_values)
    g_arr = np.array(g_values)
    
    log_dict.update({
        'signal_d/mean': np.mean(d_arr),
        'signal_d/std': np.std(d_arr),
        'signal_d/min': np.min(d_arr),
        'signal_d/max': np.max(d_arr),
        'signal_d/cv': np.std(d_arr) / (np.mean(d_arr) + 1e-8),  # 变异系数
        
        'signal_g/mean': np.mean(g_arr),
        'signal_g/std': np.std(g_arr),
        'signal_g/min': np.min(g_arr),
        'signal_g/max': np.max(g_arr),
        'signal_g/positive_ratio': float(np.mean(g_arr > 0)),  # g>0 的 client 比例
        'signal_g/negative_ratio': float(np.mean(g_arr < 0)),  # g<0 的 client 比例
    })
    
    # ===== 4. 相关性分析 =====
    if len(lambda_values) > 3:  # 至少 4 个点才有意义
        # λ 与 d 的相关性
        spearman_lambda_d, p_lambda_d = spearmanr(lambda_values, d_arr)
        pearson_lambda_d, _ = pearsonr(lambda_values, d_arr)
        
        # λ 与 g 的相关性
        spearman_lambda_g, p_lambda_g = spearmanr(lambda_values, g_arr)
        pearson_lambda_g, _ = pearsonr(lambda_values, g_arr)
        
        # d 与 g 的相关性（看两个信号是否互补）
        spearman_d_g, _ = spearmanr(d_arr, g_arr)
        
        log_dict.update({
            'correlation/spearman_lambda_d': spearman_lambda_d,
            'correlation/spearman_lambda_g': spearman_lambda_g,
            'correlation/pearson_lambda_d': pearson_lambda_d,
            'correlation/pearson_lambda_g': pearson_lambda_g,
            'correlation/spearman_d_g': spearman_d_g,  # 越低说明越互补
            'correlation/p_value_lambda_d': p_lambda_d,
            'correlation/p_value_lambda_g': p_lambda_g,
        })
    
    # ===== 5. EMA 状态分析 =====
    ema_values = np.array([lambda_emas.get(k, lambda_center) for k in party_list_this_round])
    raw_lambdas = lambda_values  # 当前轮的 λ（已经是 EMA 后的）
    
    # EMA 的"惯性"：当前 λ 与上一轮 λ 的差异
    log_dict.update({
        'ema/lambda_change_mean': float(np.mean(np.abs(lambda_values - lambda_center))),
        'ema/lambda_change_max': float(np.max(np.abs(lambda_values - lambda_center))),
    })
    
    # ===== 6. Per-client 细粒度数据 =====
    for idx, net_id in enumerate(party_list_this_round):
        log_dict[f'client_lambda/client_{net_id}'] = lambda_dict[net_id]
        log_dict[f'client_d/client_{net_id}'] = d_values[idx]
        log_dict[f'client_g/client_{net_id}'] = g_values[idx]
    
    # ===== 7. 数据异质性分析（每个 client 的类别数）=====
    client_n_classes = []
    client_data_sizes = []
    for net_id in party_list_this_round:
        dataidxs = net_dataidx_map[net_id]
        client_data_sizes.append(len(dataidxs))
    
    log_dict.update({
        'data/client_size_mean': np.mean(client_data_sizes),
        'data/client_size_std': np.std(client_data_sizes),
        'data/client_size_min': np.min(client_data_sizes),
        'data/client_size_max': np.max(client_data_sizes),
    })
    
    # λ 与数据量的相关性
    if len(lambda_values) > 3:
        spearman_lambda_size, _ = spearmanr(lambda_values, client_data_sizes)
        log_dict['correlation/spearman_lambda_datasize'] = spearman_lambda_size
    
    wandb.log(log_dict, step=round)

def log_adamoon_tables_and_figures(round, party_list_this_round, lambda_dict, 
                                    d_values, g_values, net_dataidx_map, args):
    """
    每隔 N 轮记录一次重量级可视化（表格、直方图、散点图）。
    避免每轮都记录导致 wandb 过慢。
    """
    lambda_values = [lambda_dict[k] for k in party_list_this_round]
    d_arr = np.array(d_values)
    g_arr = np.array(g_values)
    
    # ===== 散点图：d_i vs g_i，颜色 = λ_i =====
    scatter_data = [[d_values[i], g_values[i], lambda_values[i], party_list_this_round[i]] 
                    for i in range(len(party_list_this_round))]
    scatter_table = wandb.Table(
        data=scatter_data, 
        columns=["gradient_divergence", "distillation_gain", "lambda", "client_id"]
    )
    wandb.log({
        f"scatter/d_vs_g_round_{round}": wandb.plot.scatter(
            scatter_table, "gradient_divergence", "distillation_gain",
            title=f"Round {round}: d_i vs g_i"
        )
    }, step=round)
    
    # ===== λ 分布直方图 =====
    wandb.log({
        f"hist/lambda_distribution_round_{round}": wandb.Histogram(lambda_values, num_bins=20)
    }, step=round)
    
    # ===== Client 状态总表 =====
    table_data = []
    for idx, net_id in enumerate(party_list_this_round):
        n_data = len(net_dataidx_map[net_id])
        table_data.append([
            net_id, 
            lambda_dict[net_id], 
            d_values[idx], 
            g_values[idx],
            n_data,
        ])
    
    client_table = wandb.Table(
        data=table_data,
        columns=["client_id", "lambda", "d_i", "g_i", "n_samples"]
    )
    wandb.log({f"table/client_status_round_{round}": client_table}, step=round)

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    # ===== wandb 初始化 =====
    if args.use_wandb:
        wandb_run_name = args.wandb_run_name if args.wandb_run_name else args.log_file_name
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args),
            reinit=True
        )
    # ===== wandb 初始化结束 =====

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            round_start_time = time.time()
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)



            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)

            # ===== 详细日志打印 =====
            round_time = time.time() - round_start_time
            print(f'[Round {round}/{args.comm_round}] '
                f'Test Acc: {test_acc:.4f} | Train Acc: {train_acc:.4f} | '
                f'Loss: {train_loss:.4f} | μ: {args.mu} | '
                f'Time: {round_time:.1f}s')
            remaining = (args.comm_round - round - 1) * round_time
            print(f'    ETA: {remaining/60:.1f} min remaining')

            # ===== wandb 日志（MOON baseline）=====
            if args.use_wandb:
                wandb.log({
                    'perf/test_acc': test_acc,
                    'perf/train_acc': train_acc,
                    'perf/train_loss': train_loss,
                    'perf/generalization_gap': train_acc - test_acc,
                    # MOON 用固定 mu，记录等效值方便与 AdaMOON 对比
                    'lambda/mean': args.mu,
                    'lambda/std': 0.0,
                }, step=round)
            # ===== wandb 日志结束 =====

            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')

    elif args.alg == 'adamoon':
        old_nets_pool = []
        if args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)

        # AdaMOON 状态初始化
        lambda_emas = {}  # {client_id: ema_value}
        lambda_center = (args.lambda_min + args.lambda_max) / 2

        for round in range(n_comm_rounds):
            round_start_time = time.time()
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            # ===== AdaMOON 核心：计算 per-client λ =====
            if round < args.warmup_rounds:
                # Warmup 阶段：所有 client 用中心值
                lambda_dict = {k: lambda_center for k in party_list_this_round}
                logger.info('[AdaMOON] Round %d: WARMUP, λ = %.4f for all clients' % (round, lambda_center))
            else:
                # 计算 g_i（需要 client 数据，在 training 前做）
                g_values = []
                for net_id in party_list_this_round:
                    dataidxs = net_dataidx_map[net_id]
                    train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
                    # g_i: global model vs 上一轮该 client 的 local model（即当前 nets[net_id] 加载 global_w 之前的状态）
                    # 但此时 nets 已经加载了 global_w，所以用 old_nets_pool 中的上一轮 local model
                    if len(old_nets_pool) > 0:
                        prev_local = old_nets_pool[-1][net_id]  # 上一轮的 local model
                    else:
                        prev_local = nets_this_round[net_id]  # 第一轮没有历史，用自身
                    
                    prev_local.cuda()
                    global_model.cuda()
                    g_i = compute_distillation_gain(prev_local, global_model, train_dl_local, device, args.n_eval_batches)
                    prev_local.to('cpu')
                    global_model.to('cpu')
                    g_values.append(g_i)

                # 计算 d_i（深层 Cosine Divergence，批量计算）
                if len(old_nets_pool) > 0:
                    d_values = compute_gradient_divergence_batch(
                        old_nets_pool[-1], global_model, party_list_this_round
                    )
                else:
                    # 第一轮没有历史模型，所有 drift = 0，退化为均匀
                    d_values = [0.0] * len(party_list_this_round)

                # 计算自适应 λ
                lambda_dict = compute_adaptive_lambdas(
                    d_values=d_values,
                    g_values=g_values,
                    lambda_min=args.lambda_min,
                    lambda_max=args.lambda_max,
                    alpha_blend=args.alpha_blend,
                    lambda_emas=lambda_emas,
                    momentum=args.adapt_momentum,
                    client_ids=party_list_this_round
                )

                # 日志记录
                lambda_values = [lambda_dict[k] for k in party_list_this_round]
                logger.info('[AdaMOON] Round %d: λ_mean=%.5f, λ_std=%.5f, λ_min=%.5f, λ_max=%.5f' % 
                            (round, np.mean(lambda_values), np.std(lambda_values), 
                             np.min(lambda_values), np.max(lambda_values)))
                logger.info('[AdaMOON] d_values: mean=%.5f, std=%.5f' % (np.mean(d_values), np.std(d_values)))
                logger.info('[AdaMOON] g_values: mean=%.5f, std=%.5f' % (np.mean(g_values), np.std(g_values)))
                # 记录每个 client 的 λ
                for idx, net_id in enumerate(party_list_this_round):
                    logger.info('[AdaMOON] client %d: λ=%.5f, d=%.5f, g=%.5f' % 
                                (net_id, lambda_dict[net_id], d_values[idx], g_values[idx]))

            # ===== Local Training（传入 lambda_dict）=====
            global_model.cuda()
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, 
                           global_model=global_model, prev_model_pool=old_nets_pool, round=round, 
                           device=device, lambda_dict=lambda_dict)
            global_model.to('cpu')

            # ===== FedAvg 聚合 =====
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)

            # ===== 评估 =====
            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            
            # ===== 详细日志打印 =====
            round_time = time.time() - round_start_time
            if round < args.warmup_rounds:
                print(f'[Round {round}/{args.comm_round}] WARMUP | '
                    f'Test Acc: {test_acc:.4f} | Train Acc: {train_acc:.4f} | '
                    f'Loss: {train_loss:.4f} | λ: {lambda_center:.4f} (fixed) | '
                    f'Time: {round_time:.1f}s')
            else:
                lambda_values = [lambda_dict[k] for k in party_list_this_round]
                print(f'[Round {round}/{args.comm_round}] '
                    f'Test Acc: {test_acc:.4f} | Train Acc: {train_acc:.4f} | '
                    f'Loss: {train_loss:.4f} | Time: {round_time:.1f}s')
                print(f'    λ: mean={np.mean(lambda_values):.5f}, std={np.std(lambda_values):.5f}, '
                    f'min={np.min(lambda_values):.5f}, max={np.max(lambda_values):.5f}')
                print(f'    d: mean={np.mean(d_values):.5f}, std={np.std(d_values):.5f}')
                print(f'    g: mean={np.mean(g_values):.5f}, std={np.std(g_values):.5f}')
                # 预估剩余时间
                remaining = (args.comm_round - round - 1) * round_time
                print(f'    ETA: {remaining/60:.1f} min remaining')
            # ===== wandb 增强日志 =====
            if args.use_wandb:
                if round < args.warmup_rounds:
                    log_adamoon_wandb(
                        round=round, train_acc=train_acc, test_acc=test_acc, train_loss=train_loss,
                        party_list_this_round=party_list_this_round, lambda_dict=lambda_dict,
                        d_values=[], g_values=[], lambda_emas=lambda_emas, args=args,
                        nets_this_round=nets_this_round, global_model=global_model,
                        net_dataidx_map=net_dataidx_map, old_nets_pool=old_nets_pool,
                        is_warmup=True
                    )
                else:
                    log_adamoon_wandb(
                        round=round, train_acc=train_acc, test_acc=test_acc, train_loss=train_loss,
                        party_list_this_round=party_list_this_round, lambda_dict=lambda_dict,
                        d_values=d_values, g_values=g_values, lambda_emas=lambda_emas, args=args,
                        nets_this_round=nets_this_round, global_model=global_model,
                        net_dataidx_map=net_dataidx_map, old_nets_pool=old_nets_pool,
                        is_warmup=False
                    )
                    
                    # 每 10 轮记录一次重量级可视化（散点图、表格、直方图）
                    if round % 10 == 0 or round == n_comm_rounds - 1:
                        log_adamoon_tables_and_figures(
                            round=round, party_list_this_round=party_list_this_round,
                            lambda_dict=lambda_dict, d_values=d_values, g_values=g_values,
                            net_dataidx_map=net_dataidx_map, args=args
                        )
            # ===== wandb 增强日志结束 =====

            # ===== 更新 model pool =====
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i + 1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            # ===== 保存模型 =====
            mkdirs(args.modeldir + 'adamoon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir + 'adamoon/global_model_' + args.log_file_name + '.pth')
    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            # ===== wandb 日志（FedAvg baseline）=====
            if args.use_wandb:
                wandb.log({
                    'perf/test_acc': test_acc,
                    'perf/train_acc': train_acc,
                    'perf/train_loss': train_loss,
                    'perf/generalization_gap': train_acc - test_acc,
                }, step=round)
            # ===== wandb 日志结束 =====
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            # ===== wandb 日志（FedProx baseline）=====
            if args.use_wandb:
                wandb.log({
                    'perf/test_acc': test_acc,
                    'perf/train_acc': train_acc,
                    'perf/train_loss': train_loss,
                    'perf/generalization_gap': train_acc - test_acc,
                    'lambda/mean': args.mu,  # FedProx 的 mu 也记录，方便对比
                    'lambda/std': 0.0,
                }, step=round)
            # ===== wandb 日志结束 =====
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')

    # ===== wandb 结束 + summary =====
    if args.use_wandb:
        try:
            # 不同分支变量名可能不同（test_acc vs testacc）
            final_test = test_acc if 'test_acc' in dir() else (testacc if 'testacc' in dir() else None)
            final_train = train_acc if 'train_acc' in dir() else (trainacc if 'trainacc' in dir() else None)
            final_loss = train_loss if 'train_loss' in dir() else None
            if final_test is not None:
                wandb.run.summary['final_test_acc'] = final_test
            if final_train is not None:
                wandb.run.summary['final_train_acc'] = final_train
            if final_loss is not None:
                wandb.run.summary['final_train_loss'] = final_loss
        except:
            pass
        
        wandb.run.summary['algorithm'] = args.alg
        wandb.run.summary['beta'] = args.beta
        wandb.run.summary['n_parties'] = args.n_parties
        wandb.run.summary['epochs'] = args.epochs
        wandb.run.summary['comm_rounds'] = args.comm_round
        wandb.run.summary['lr'] = args.lr
        wandb.run.summary['seed'] = args.init_seed
        
        if args.alg == 'adamoon':
            wandb.run.summary['lambda_min'] = args.lambda_min
            wandb.run.summary['lambda_max'] = args.lambda_max
            wandb.run.summary['alpha_blend'] = args.alpha_blend
            wandb.run.summary['adapt_momentum'] = args.adapt_momentum
            wandb.run.summary['warmup_rounds'] = args.warmup_rounds
            if lambda_emas:
                final_lambdas = list(lambda_emas.values())
                wandb.run.summary['final_lambda_mean'] = np.mean(final_lambdas)
                wandb.run.summary['final_lambda_std'] = np.std(final_lambdas)
                wandb.run.summary['final_lambda_min'] = np.min(final_lambdas)
                wandb.run.summary['final_lambda_max'] = np.max(final_lambdas)
        elif args.alg in ['moon', 'fedprox']:
            wandb.run.summary['mu'] = args.mu
        
        wandb.finish()
    # ===== wandb 结束 =====