############### test KDHT ##############
from nets.student_nets import StuNet
from nets.metor_nets import MentorNet
from nets.channel_nets import channel_net
from nets.base_nets import base_net
import torch
import random
from torch import nn
from matplotlib import pyplot as plt
import torchvision
import numpy as np
import os
from torch.nn import functional as F
from data_utils import get_data_loaders
from Config import Config
import json
import time
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

# show image and save
def save_images(y, x_rec, save_pth):
    os.makedirs(save_pth,exist_ok=True)
    imgs_sample = y.data.detach()
    filename = os.path.join(save_pth,"raw.jpg")
    plt.figure()
    for i in range(len(imgs_sample)):
        plt.subplot(4, 8, i + 1)
        # plt.tight_layout()
        # 反归一化，将数据重新映射到0-1之间
        img = imgs_sample[i] / 2 + 0.5
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.axis('off')
    plt.savefig(filename)
    os.makedirs(os.path.join(save_pth, "all_imgs_raw"), exist_ok=True)
    plt.figure()
    for i in range(len(imgs_sample)):
        img = imgs_sample[i] / 2 + 0.5
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.savefig(os.path.join(save_pth, f"all_imgs_raw/{i}.jpg"))
    # Show 32 of the images.
    # grid_img = torchvision.utils.make_grid(imgs_sample[:100].detach(), nrow=10)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # print("origin images")
    # plt.show()

    imgs_sample = x_rec.detach()
    filename = os.path.join(save_pth, "rec.jpg")
    plt.figure()
    for i in range(len(imgs_sample)):
        plt.subplot(4, 8, i + 1)
        # plt.tight_layout()
        # 反归一化，将数据重新映射到0-1之间
        img = imgs_sample[i] / 2 + 0.5
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.axis('off')
    plt.savefig(filename)
    os.makedirs(os.path.join(save_pth, "all_imgs_rec"), exist_ok=True)
    plt.figure()
    for i in range(len(imgs_sample)):
        img = imgs_sample[i] / 2 + 0.5
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.savefig(os.path.join(save_pth, f"all_imgs_rec/{i}.jpg"))
    # Show 32 of the images.
    # grid_img = torchvision.utils.make_grid(imgs_sample[:100].detach(), nrow=10)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # plt.title("reconstruct images")
    # plt.show()

# training based on KDHT
def MKDT(stu_model, mentor_model, mentor_model_name, train_dataloader, test_dataloader, cfg, client_snr = None, client_id=0, com_round=0):
    start = time.time()
    checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{client_id}")
    os.makedirs(checkpoint_path, exist_ok=True)
    channel_name = "rali" if cfg.use_Rali else "awgn"
    SNR = client_snr if client_snr != None else cfg.SNR
    # laod weights
    mentor_weights_path = os.path.join(checkpoint_path,f"mentor_{mentor_model_name}_{SNR}_{channel_name}.pth")
    if os.path.exists(mentor_weights_path):
        weights = torch.load(mentor_weights_path,map_location="cpu")
        mentor_model.load_state_dict(weights)
    stu_weights_path = os.path.join(checkpoint_path, f"student_{cfg.Stu_model_name}_{SNR}_{channel_name}.pth")
    if os.path.exists(stu_weights_path):
        weights = torch.load(stu_weights_path, map_location="cpu")
        stu_model.load_state_dict(weights)

    stu_model = stu_model.to(cfg.device)
    mentor_model = mentor_model.to(cfg.device)
    if (com_round+1)%10==0:
        cfg.isc_lr/=10
        cfg.channel_lr/=10

    # define optimizer
    optimizer_stu_encoder = torch.optim.Adam(stu_model.isc_model.encoder.parameters(), lr=cfg.isc_lr/100,
                                             weight_decay=cfg.weight_delay)
    optimizer_stu_decoder = torch.optim.Adam(stu_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_stu_channel = torch.optim.Adam(stu_model.ch_model.parameters(), lr=cfg.channel_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_mentor_encoder = torch.optim.Adam(mentor_model.isc_model.encoder.parameters(), lr=cfg.isc_lr/100,
                                             weight_decay=cfg.weight_delay)
    optimizer_mentor_decoder = torch.optim.Adam(mentor_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_mentor_channel = torch.optim.Adam(mentor_model.ch_model.parameters(), lr=cfg.channel_lr,
                                             weight_decay=cfg.weight_delay)

    # optimizer_stu_encoder = torch.optim.SGD(stu_model.isc_model.encoder.parameters(), lr=cfg.isc_lr / 100,
    #                                          weight_decay=cfg.weight_delay,momentum=0.9)
    # optimizer_stu_decoder = torch.optim.SGD(stu_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
    #                                          weight_decay=cfg.weight_delay,momentum=0.9)
    # optimizer_stu_channel = torch.optim.SGD(stu_model.ch_model.parameters(), lr=cfg.channel_lr,
    #                                          weight_decay=cfg.weight_delay,momentum=0.9)
    # optimizer_mentor_encoder = torch.optim.SGD(mentor_model.isc_model.encoder.parameters(), lr=cfg.isc_lr / 100,
    #                                             weight_decay=cfg.weight_delay,momentum=0.9)
    # optimizer_mentor_decoder = torch.optim.SGD(mentor_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
    #                                             weight_decay=cfg.weight_delay,momentum=0.9)
    # optimizer_mentor_channel = torch.optim.SGD(mentor_model.ch_model.parameters(), lr=cfg.channel_lr,
    #                                             weight_decay=cfg.weight_delay,momentum=0.9)

    scheduler_mentor_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_mentor_decoder, mode='min', factor=0.1, patience=50,
                                                           verbose=True, threshold=1e-8, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    scheduler_mentor_channel = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_mentor_channel, mode='min', factor=0.1,
                                                             patience=50,
                                                             verbose=True, threshold=1e-7, threshold_mode='rel',
                                                             cooldown=0, min_lr=0, eps=1e-08)
    scheduler_stu_encoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stu_encoder, mode='min',
                                                                       factor=0.1, patience=50,
                                                                       verbose=True, threshold=1e-7,
                                                                       threshold_mode='rel',
                                                                       cooldown=0, min_lr=0, eps=1e-08)
    scheduler_stu_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stu_decoder, mode='min',
                                                                          factor=0.1, patience=50,
                                                                          verbose=True, threshold=1e-7,
                                                                          threshold_mode='rel',
                                                                          cooldown=0, min_lr=0, eps=1e-08)
    scheduler_stu_channel = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stu_channel, mode='min',
                                                                          factor=0.1,
                                                                          patience=50,
                                                                          verbose=True, threshold=1e-7,
                                                                          threshold_mode='rel',
                                                                          cooldown=0, min_lr=0, eps=1e-08)
    # define loss function
    mse = nn.MSELoss()
    kl = nn.KLDivLoss()
    crossentropy = nn.CrossEntropyLoss()
    # training
    train_mentor_loss = []
    train_mentor_acc = []
    train_stu_loss = []
    train_stu_acc = []
    for epoch in range(cfg.epochs_for_clients):
        stu_model.train()
        mentor_model.train()
        for x,y in train_dataloader:
            optimizer_stu_encoder.zero_grad()
            optimizer_stu_channel.zero_grad()
            optimizer_stu_decoder.zero_grad()
            optimizer_mentor_encoder.zero_grad()
            optimizer_mentor_channel.zero_grad()
            optimizer_mentor_decoder.zero_grad()
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            mentor_s_encoding, mentor_c_decoding, mentor_x_rec = mentor_model(x)
            stu_s_encoding, stu_c_decoding, stu_x_rec = stu_model(x)
            # compute loss
            l_mentor_task = crossentropy(mentor_x_rec,y)
            mentor_acc = (torch.argmax(mentor_x_rec, dim=-1) == y).float().mean()
            l_stu_task = crossentropy(stu_x_rec,y)
            stu_acc = (torch.argmax(stu_x_rec, dim=-1) == y).float().mean()
            mentor_dist_1 = F.log_softmax(mentor_x_rec)
            mentor_dist_2 = F.softmax(mentor_x_rec)
            stu_dist_1 = F.log_softmax(stu_x_rec)
            stu_dist_2 = F.softmax(stu_x_rec)
            l_mentor_dis = kl(mentor_dist_1,stu_dist_2)/(l_stu_task)
            l_stu_dis = kl(stu_dist_1,mentor_dist_2)/(l_mentor_task)
            l_mentor_hid = l_stu_hid = torch.mean((mse(stu_s_encoding,mentor_s_encoding) + mse(stu_c_decoding,mentor_c_decoding))\
                                       /(l_mentor_task+l_stu_task))
            l_stu = l_stu_task+l_stu_dis+l_stu_hid
            l_mentor = l_mentor_task+l_mentor_dis+l_mentor_hid
            total_loss = (l_stu+l_mentor)/2
            total_loss.backward()
            optimizer_stu_encoder.step()
            optimizer_stu_decoder.step()
            optimizer_stu_channel.step()
            optimizer_mentor_decoder.step()
            optimizer_mentor_encoder.step()
            optimizer_mentor_channel.step()

            # scheduler_mentor_channel.step(l_mentor)
            # scheduler_mentor_decoder.step(l_mentor)
            # scheduler_stu_channel.step(l_stu)
            # scheduler_stu_decoder.step(l_stu)
            # scheduler_stu_encoder.step(l_stu)

            print(f"client{client_id}-epoch {epoch} | student loss:{l_stu} | task_loss:{l_stu_task} | dis_loss:{l_stu_dis} | hid_loss:{l_stu_hid} | acc:{stu_acc}")
            print(f"client{client_id}-epoch {epoch} | mentor loss:{l_mentor} | task_loss:{l_mentor_task} | dis_loss:{l_mentor_dis} | hid_loss:{l_mentor_hid} | acc:{mentor_acc}")

            train_stu_loss.append(l_stu.item())
            train_mentor_loss.append(l_mentor.item())
            train_stu_acc.append(stu_acc.item())
            train_mentor_acc.append(mentor_acc.item())

        # save_images(y, mentor_x_rec, os.path.join(cfg.logs_dir, f"{client_id}","save_imgs",f"round_{com_round}_train_mentor_imgs"))
        # save_images(y, stu_x_rec, os.path.join(cfg.logs_dir, f"{client_id}","save_imgs", f"round_{com_round}_train_student_imgs"))
        # save_weights
        torch.save(mentor_model.state_dict(),mentor_weights_path)
        torch.save(stu_model.state_dict(),stu_weights_path)

    # testing
    # test_mentor_loss, test_stu_loss, test_mentor_acc, test_stu_acc = Test_MKDT_ISC(stu_model, mentor_model, test_dataloader, cfg, client_id, com_round)
    test_stu_loss = test_mentor_loss = test_stu_acc = test_mentor_acc = None
    train_mentor_loss = np.mean(train_mentor_loss)
    train_stu_loss = np.mean(train_stu_loss)
    train_stu_acc = np.mean(train_stu_acc)
    train_mentor_acc = np.mean(train_mentor_acc)
    records = {"stu_train_loss":train_stu_loss,"mentor_train_loss":train_mentor_loss,"stu_test_loss": test_stu_loss,"mentor_test_loss": test_mentor_loss,
                    "stu_train_acc":train_stu_acc,"mentor_train_acc":train_mentor_acc,"stu_test_acc":test_stu_acc,"mentor_test_acc":test_mentor_acc}
    os.makedirs(os.path.join(cfg.logs_dir, f"{client_id}", "loss_acc"),exist_ok=True)
    with open(os.path.join(cfg.logs_dir, f"{client_id}", "loss_acc", f"round_{com_round}_{SNR}.json"), "w",
              encoding="utf-8")as f:
        f.write(json.dumps(records, ensure_ascii=False, indent=4))
    print("waste time:",time.time()-start)
    return stu_model.state_dict()

# test student and mentor models
def Test_MKDT_ISC(stu_model, mentor_model, test_dataloader, cfg, client_id, com_round):
    stu_model.to(cfg.device)
    mentor_model.to(cfg.device)
    stu_model.eval()
    mentor_model.eval()
    # define loss function
    mse = nn.MSELoss()
    crossentropy = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss()
    test_stu_loss = []
    test_mentor_loss = []
    test_mentor_acc = []
    test_stu_acc = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            mentor_s_encoding, mentor_c_decoding, mentor_x_rec = mentor_model(x)
            stu_s_encoding, stu_c_decoding, stu_x_rec = stu_model(x)
            # compute loss
            l_mentor_task = crossentropy(mentor_x_rec, y)
            l_stu_task = crossentropy(stu_x_rec, y)
            mentor_acc = (torch.argmax(mentor_x_rec, dim=-1) == y).float().mean()
            stu_acc = (torch.argmax(stu_x_rec, dim=-1) == y).float().mean()
            mentor_dist_1 = F.log_softmax(mentor_x_rec)
            mentor_dist_2 = F.softmax(mentor_x_rec)
            stu_dist_1 = F.log_softmax(stu_x_rec)
            stu_dist_2 = F.softmax(stu_x_rec)
            l_mentor_dis = kl(mentor_dist_1, stu_dist_2) / (l_stu_task)
            l_stu_dis = kl(stu_dist_1, mentor_dist_2) / (l_mentor_task)
            l_mentor_hid = l_stu_hid = torch.mean(
                (mse(stu_s_encoding, mentor_s_encoding) + mse(stu_c_decoding, mentor_c_decoding)) \
                / (l_mentor_task + l_stu_task))
            l_stu = l_stu_task + l_stu_dis + l_stu_hid
            l_mentor = l_mentor_task + l_mentor_dis + l_mentor_hid
            print(
                f"client_id{client_id}-test | student loss:{l_stu} | task_loss:{l_stu_task} | dis_loss:{l_stu_dis} | hid_loss:{l_stu_hid} | acc:{stu_acc}")
            print(
                f"client_id{client_id}-test | mentor loss:{l_mentor} | task_loss:{l_mentor_task} | dis_loss:{l_mentor_dis} | hid_loss:{l_mentor_hid} | acc:{mentor_acc}")

            test_stu_loss.append(l_stu.item())
            test_mentor_loss.append(l_mentor.item())
            test_mentor_acc.append(mentor_acc.item())
            test_stu_acc.append(stu_acc.item())
        # save_images(y,mentor_x_rec,os.path.join(cfg.logs_dir,f"{client_id}","save_imgs", f"round_{com_round}_test_mentor_imgs"))
        # save_images(y,stu_x_rec,os.path.join(cfg.logs_dir,f"{client_id}","test_student_imgs"))
    test_stu_loss = np.mean(test_stu_loss)
    test_mentor_loss = np.mean(test_mentor_loss)
    test_mentor_acc = np.mean(test_mentor_acc)
    test_stu_acc = np.mean(test_stu_acc)
    return test_mentor_loss, test_stu_loss, test_mentor_acc, test_stu_acc

def Train_for_weak_clients(stu_model, train_dataloader, test_dataloader, cfg, client_snr, client_id, com_round):
    start = time.time()
    checkpoint_path = os.path.join(cfg.checkpoints_dir, f"{client_id}")
    os.makedirs(checkpoint_path, exist_ok=True)
    channel_name = "rali" if cfg.use_Rali else "awgn"
    SNR = client_snr
    # laod weights
    stu_weights_path = os.path.join(checkpoint_path, f"student_{cfg.Stu_model_name}_{SNR}_{channel_name}.pth")
    print(stu_weights_path)
    if os.path.exists(stu_weights_path):
        print("loading weights")
        weights = torch.load(stu_weights_path, map_location="cpu")
        stu_model.load_state_dict(weights)

    stu_model.to(cfg.device)
    # define optimizer
    optimizer_stu_encoder = torch.optim.Adam(stu_model.isc_model.encoder.parameters(), lr=cfg.isc_lr/100,
                                             weight_decay=cfg.weight_delay)
    optimizer_stu_decoder = torch.optim.Adam(stu_model.isc_model.decoder.parameters(), lr=cfg.isc_lr,
                                             weight_decay=cfg.weight_delay)
    optimizer_stu_channel = torch.optim.Adam(stu_model.ch_model.parameters(), lr=cfg.channel_lr,
                                             weight_decay=cfg.weight_delay)

    scheduler_stu_decoder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stu_decoder, mode='min',
                                                                       factor=0.1, patience=250,
                                                                       verbose=True, threshold=0.0001,
                                                                       threshold_mode='rel',
                                                                       cooldown=0, min_lr=0, eps=1e-08)
    scheduler_stu_channel = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stu_channel, mode='min',
                                                                       factor=0.1,
                                                                       patience=250,
                                                                       verbose=True, threshold=0.0001,
                                                                       threshold_mode='rel',
                                                                       cooldown=0, min_lr=0, eps=1e-08)

    # define loss function
    mse = nn.MSELoss()
    crossentropy = nn.CrossEntropyLoss()
    train_stu_loss = []
    # training
    for epoch in range(cfg.epochs_for_clients):
        train_stu_loss = []
        train_stu_acc = []
        stu_model.train()
        for x,y in train_dataloader:
            optimizer_stu_encoder.zero_grad()
            optimizer_stu_channel.zero_grad()
            optimizer_stu_decoder.zero_grad()
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            stu_s_encoding,stu_c_decoding,stu_x_rec = stu_model(x)
            # compute loss
            # l_stu_task = F.l1_loss(stu_x_rec,y)*100
            l_stu_task = crossentropy(stu_x_rec, y)
            acc = (torch.argmax(stu_x_rec, dim=-1) == y).float().mean()
            l_stu_coding = mse(stu_s_encoding,stu_c_decoding)
            l_stu = l_stu_task
            l_stu.backward()
            optimizer_stu_encoder.step()
            optimizer_stu_decoder.step()
            optimizer_stu_channel.step()

            scheduler_stu_decoder.step(l_stu)
            scheduler_stu_channel.step(l_stu)
            print(f"client{client_id}-epoch {epoch} | student loss:{l_stu} | task_loss:{l_stu_task} | code_loss:{l_stu_coding} | acc:{acc}")
            train_stu_loss.append(l_stu.item())
            train_stu_acc.append(acc.item())
        # save_images(y, stu_x_rec, os.path.join(cfg.logs_dir, f"{client_id}","save_imgs",f"round_{com_round}_train_student_imgs"))
        # save_weights
        torch.save(stu_model.state_dict(), stu_weights_path)
    # testing
    test_stu_loss, test_stu_acc = Test_Stu_ISC(stu_model,test_dataloader, cfg,client_id, com_round)
    train_stu_loss = np.mean(train_stu_loss)
    train_stu_acc = np.mean(train_stu_acc)
    records = {"stu_train_loss":train_stu_loss,"stu_test_loss":test_stu_loss,"stu_train_acc":train_stu_acc,"stu_test_acc":test_stu_acc}
    os.makedirs(os.path.join(cfg.logs_dir,f"{client_id}","loss_acc"),exist_ok=True)
    with open(os.path.join(cfg.logs_dir,f"{client_id}","loss_acc",f"round_{com_round}_{SNR}.json"),"w",encoding="utf-8")as f:
        f.write(json.dumps(records,ensure_ascii=False,indent=4))
    print("waste time:", time.time() - start)
    return stu_model.state_dict()

def Test_Stu_ISC(stu_model, test_dataloader, cfg, client_id, com_round):
    stu_model.to(cfg.device)
    stu_model.eval()
    # define loss function
    mse = nn.MSELoss()
    crossentropy = nn.CrossEntropyLoss()
    test_stu_loss = []
    test_stu_acc = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            stu_s_encoding, stu_c_decoding, stu_x_rec = stu_model(x)
            # compute loss
            l_stu_task = crossentropy(stu_x_rec, y)
            acc = (torch.argmax(stu_x_rec, dim=-1) == y).float().mean()
            l_stu_coding = mse(stu_s_encoding,stu_c_decoding)
            l_stu = l_stu_task + l_stu_coding

            print(
                f"client_id{client_id}-test | student loss:{l_stu} | task_loss:{l_stu_task} | code_loss:{l_stu_coding} | acc:{acc}")

            test_stu_loss.append(l_stu.item())
            test_stu_acc.append(acc.item())
        # save_images(y,stu_x_rec,os.path.join(cfg.logs_dir,f"{client_id}","save_imgs",f"round_{com_round}_test_student_imgs"))
    test_stu_loss = np.mean(test_stu_loss)
    test_stu_acc = np.mean(test_stu_acc)
    return test_stu_loss, test_stu_acc

if __name__ == '__main__':
    # hyparametes set
    same_seeds(2048)
    config = Config()
    # prepare data
    _, train_loader, test_loader, stats = get_data_loaders(config, True)
    print(stats)

    # KDHT(Stu_model,Mentor_model,train_loader,test_loader,config)
    # for snr in [0,5,10,15,20,25]:
    #     # prepare model
    #     Stu_net = StuNet(config.Stu_model_name)
    #     Stu_model = base_net(Stu_net, channel_net(snr=snr, rali=config.use_Rali, if_RTN=config.use_RTN))
    #     Mentor_net = MentorNet(config.Mentor_model_name)
    #     Mentor_model = base_net(Mentor_net, channel_net(snr=config.SNR, rali=config.use_Rali, if_RTN=config.use_RTN))
    #     KDHT(Stu_model, Mentor_model, train_loader, test_loader, config,client_snr = snr, client_id = 4)
    #     # Train_for_weak_clients(Stu_model, train_loader, test_loader, config, client_snr=snr, client_id=3, com_round=0)

    ## CL
    Stu_net = StuNet(config.Stu_model_name)
    Stu_model = base_net(Stu_net, channel_net(snr=config.SNR, rali=config.use_Rali, if_RTN=config.use_RTN))
    # Mentor_net = MentorNet(config.Mentor_model_name)
    # Mentor_model = base_net(Mentor_net, channel_net(snr=config.SNR, rali=config.use_Rali, if_RTN=config.use_RTN))
    # KDHT(Stu_model, Mentor_model, train_loader, test_loader, config, client_snr=snr, client_id=4)
    Train_for_weak_clients(Stu_model, train_loader, test_loader, config, client_snr=config.SNR, client_id=12, com_round=0)














