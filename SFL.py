import torch
from torch import nn
import numpy as np
import json
from Config import Config
import random
from torchsummary import summary
import copy
import os
from nets.student_nets import StuNet
from nets.metor_nets import MentorNet
from nets.channel_nets import channel_net
from nets.base_nets import base_net
from data_utils import get_data_loaders,get_data_tiny_loader
from MKDT import MKDT,Train_for_weak_clients
import time
from nets.sc_model import ISCNet
torch.cuda.set_device(0)
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def SFL(clients_loader, train_dataloader, test_dataloader, cfg):
    local_mentor_model_name = ["resnet50","ViT","resnet152","resnet152","resnet152","ViT","resnet152","resnet50","resnet50"]
    Stu_net = ISCNet(cfg.Stu_model_name,cfg.class_num)
    Stu_model = base_net(Stu_net, channel_net(snr=cfg.SNR, rali=cfg.use_Rali, if_RTN=cfg.use_RTN))
    SNR_for_each_client = random.sample(range(15, 25), cfg.n_clients)
    Mean_SNR = np.mean(SNR_for_each_client)
    Power_for_each_client = random.sample(range(0,10),cfg.n_clients)
    Powers = np.sum(Power_for_each_client)/100
    Bandwidth_for_each_client = random.sample(range(0, 10), cfg.n_clients)
    Bandwidthes = np.sum(Bandwidth_for_each_client) / 10

    # student_weights = Stu_model.state_dict()
    # prepruning and broadcast stu_model
    student_weights, last_prune_ratio = ASP(Stu_model.state_dict(),Mean_SNR, Bandwidthes, Powers,cfg,0)
    print("broadcast weights to clients")
    for i in range(cfg.n_clients):
        checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{i}")
        os.makedirs(checkpoint_path, exist_ok=True)
        channel_name = "rali" if cfg.use_Rali else "awgn"
        SNR = cfg.SNR
        stu_weights_path = os.path.join(checkpoint_path, f"student_{cfg.Stu_model_name}_{SNR}_{channel_name}.pth")
        torch.save(student_weights, stu_weights_path)
    for com_round in range(cfg.communication_rounds):
        weights_for_clients = []
        print(f"communication_round {com_round} start! Mean SNR:{Mean_SNR}")
        # local training
        for client_id in range(cfg.n_clients):
            local_Mentor_net = ISCNet(local_mentor_model_name[client_id],cfg.class_num)
            local_Mentor_model = base_net(local_Mentor_net, channel_net(snr=cfg.SNR, rali=cfg.use_Rali, if_RTN=cfg.use_RTN))
            if local_mentor_model_name[client_id] != None:
                # Train clients with MKD
                stu_weights = MKDT(copy.deepcopy(Stu_model),local_Mentor_model,local_mentor_model_name[client_id],clients_loader[client_id],test_dataloader,cfg, cfg.SNR,client_id,com_round)
            else:
                # Train without MKD
                stu_weights = Train_for_weak_clients(copy.deepcopy(Stu_model), clients_loader[client_id], test_dataloader, cfg, cfg.SNR, client_id, com_round)
            weights_for_clients.append(stu_weights)

        # server do:
        print("merge weights")
        w_avg = copy.deepcopy(weights_for_clients[0])
        for k in w_avg.keys():
            if "num_batches_tracked" in k:
                continue
            w_avg[k] = w_avg[k] * clients_loader[0].dataset.__len__()
            for i in range(1,cfg.n_clients):
                w_avg[k] += weights_for_clients[i][k] * clients_loader[i].dataset.__len__()
            w_avg[k] = torch.div(w_avg[k],train_dataloader.dataset.__len__())
        print("broadcast weights to clients")
        for i in range(cfg.n_clients):
            checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{i}")
            os.makedirs(checkpoint_path, exist_ok=True)
            channel_name = "rali" if cfg.use_Rali else "awgn"
            SNR = cfg.SNR
            stu_weights_path = os.path.join(checkpoint_path, f"student_{cfg.Stu_model_name}_{SNR}_{channel_name}.pth")
            torch.save(w_avg, stu_weights_path)
        if com_round+1 % 10 == 0:
            SNR_for_each_client = random.sample(range(15, 25), cfg.n_clients)
            Mean_SNR = np.mean(SNR_for_each_client)
            Power_for_each_client = random.sample(range(0, 10), cfg.n_clients)
            Powers = np.sum(Power_for_each_client) / 100
            Bandwidth_for_each_client = random.sample(range(0, 10), cfg.n_clients)
            Bandwidthes = np.sum(Bandwidth_for_each_client) / 10
            # Pruning for weights when channel changes
            w_avg, last_prune_ratio = ASP(w_avg, Mean_SNR, Bandwidthes, Powers, cfg, last_prune_ratio)


def ASP(weights, Mean_SNR, Bandwidthes, Powers, cfg,last_prune_ratio):
    start = time.time()
    prune_ratio = (((cfg.SNR_MAX-Mean_SNR)/(cfg.SNR_MAX-cfg.SNR_MIN))+Bandwidthes/cfg.Bandwidth_MAX + Powers/(cfg.Power_MAX*cfg.n_clients))*cfg.eta/3
    print("prune ratio:",prune_ratio)
    # prune more
    if prune_ratio > last_prune_ratio:
        print("prune")
        for layer_id,k in enumerate(weights.keys()):
            if "num_batches_tracked" in k or "bias" in k:
                continue
            temp = weights[k].view(-1)
            topk = int(prune_ratio * temp.shape[0])
            _, p_indexes = torch.topk(temp.abs(), k=topk, dim=0, largest=False)
            temp[p_indexes] = 0
            weights[k] = temp.view(weights[k].shape)

    # random add parameters
    elif prune_ratio < last_prune_ratio:
        print("add")
        add_ratio = last_prune_ratio - prune_ratio
        for layer_id,k in enumerate(weights.keys()):
            if "num_batches_tracked" in k or "bias" in k:
                continue
            temp = weights[k].view(-1)
            topk = int(add_ratio * temp.shape[0])
            _, add_indexes = torch.topk(temp.abs(), k=topk, dim=0, largest=False)
            temp[add_indexes] = torch.randn_like(temp)[add_indexes]
            weights[k] = temp.view(weights[k].shape)
    # for layer_id, k in enumerate(weights.keys()):
    #     if "num_batches_tracked" in k or "bias" in k:
    #         continue
    #     mask = weights[k].eq(0)
    #     val = torch.masked_select(weights[k], mask)
    #     print(val.shape)
    #     # print(weights[k])
    print("prune time:",time.time()-start)
    return weights, prune_ratio

if __name__ == '__main__':
    # hyparametes set
    same_seeds(2048)
    cfg = Config()
    # prepare data
    if cfg.dataset == "tiny":
        clients_loader, train_loader, test_loader, stats = get_data_tiny_loader(cfg, True)
    else:
        clients_loader, train_loader, test_loader, stats = get_data_loaders(cfg, True)
    print(stats)
    SFL(clients_loader, train_loader, test_loader, cfg)




