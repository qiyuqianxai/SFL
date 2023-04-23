from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import pickle

def message_gen(k, mb_size):
    tot_message_num = pow(2,k)
    m = torch.zeros(mb_size, tot_message_num)
    label = torch.zeros(mb_size)
    for ind_mb in range(mb_size):
        if ind_mb % tot_message_num == 0:
            rand_lst = torch.randperm(tot_message_num)
        ind_one_rand_lst = ind_mb % tot_message_num
        ind_one = rand_lst[ind_one_rand_lst]
        m[ind_mb, ind_one] = 1
        label[ind_mb] = ind_one
    return m, label

def channel_set_gen(num_channels, tap_num, if_toy):
    channel_list = []
    for ind_channels in range(num_channels):
        if if_toy:
            assert tap_num == 1
            if ind_channels % 2 == 0:
                h_toy = torch.zeros(2 * tap_num)
                h_toy[0] = 1 * np.cos(np.pi/4)
                h_toy[1] = 1 * np.sin(np.pi/4)
            else:
                h_toy = torch.zeros(2 * tap_num)
                h_toy[0] = 1 * np.cos((3*np.pi) / 4)
                h_toy[1] = 1 * np.sin((3*np.pi) / 4)
            channel_list.append(h_toy)
        else:
            chan_var = 1 / (2 * tap_num)  # since we are generating real and im. part indep. so 1/2 and we are considering complex, -> 2L generated
            Chan = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2 * tap_num),
                                                                              chan_var * torch.eye(2 * tap_num))
            h = Chan.sample()
            channel_list.append(h)
    return channel_list

def complex_mul(h, x): # h fixed on batch, x has multiple batch
    if len(h.shape) == 1:
        # h is same over all messages (if estimated h, it is averaged)
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[0] - x[:, 1] * h[1]
        y[:, 1] = x[:, 0] * h[1] + x[:, 1] * h[0]
    elif len(h.shape) == 2:
        # h_estimated is not averaged
        assert x.shape[0] == h.shape[0]
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[:, 0] - x[:, 1] * h[:, 1]
        y[:, 1] = x[:, 0] * h[:, 1] + x[:, 1] * h[:, 0]
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError       #尚未实现的方法
    return y

def complex_mul_taps(h, x_tensor):
    if len(h.shape) == 1:
        L = h.shape[0] // 2  # length/2 of channel vector means number of taps  6/2=3,确实是tap的个数
    elif len(h.shape) == 2:
        L = h.shape[1] // 2  # length/2 of channel vector means number of taps
    else:
        print('h shape length need to be either 1 or 2')
        raise NotImplementedError
    y = torch.zeros(x_tensor.shape[0], x_tensor.shape[1], dtype=torch.float)
    assert x_tensor.shape[1] % 2 == 0
    for ind_channel_use in range(x_tensor.shape[1]//2):     # "//" : 取整除 - 返回商的整数部分（向下取整）
        for ind_conv in range(min(L, ind_channel_use+1)):
            if len(h.shape) == 1:
                y[:, (ind_channel_use) * 2:(ind_channel_use + 1) * 2] += complex_mul(h[2*ind_conv:2*(ind_conv+1)], x_tensor[:, (ind_channel_use-ind_conv)*2:(ind_channel_use-ind_conv+1)*2])
            else:
                y[:, (ind_channel_use) * 2:(ind_channel_use + 1) * 2] += complex_mul(
                    h[:, 2 * ind_conv:2 * (ind_conv + 1)],
                    x_tensor[:, (ind_channel_use - ind_conv) * 2:(ind_channel_use - ind_conv + 1) * 2])

    return y

def complex_conv_transpose(h_trans, y_tensor): # takes the role of inverse filtering  起到反滤波的作用？？？
    assert len(y_tensor.shape) == 2 # batch
    assert y_tensor.shape[1] % 2 == 0
    assert h_trans.shape[0] % 2 == 0
    if len(h_trans.shape) == 1:
        L = h_trans.shape[0] // 2
    elif len(h_trans.shape) == 2:
        L = h_trans.shape[1] // 2
    else:
        print('h shape length need to be either 1 or 2')

    deconv_y = torch.zeros(y_tensor.shape[0], y_tensor.shape[1] + 2*(L-1), dtype=torch.float)
    for ind_y in range(y_tensor.shape[1]//2):
        ind_y_deconv = ind_y + (L-1)
        for ind_conv in range(L):
            if len(h_trans.shape) == 1:
                deconv_y[:, 2*(ind_y_deconv - ind_conv):2*(ind_y_deconv - ind_conv+1)] += complex_mul(h_trans[2*ind_conv:2*(ind_conv+1)] , y_tensor[:,2*ind_y:2*(ind_y+1)])
            else:
                deconv_y[:, 2 * (ind_y_deconv - ind_conv):2 * (ind_y_deconv - ind_conv + 1)] += complex_mul(
                    h_trans[:, 2 * ind_conv:2 * (ind_conv + 1)], y_tensor[:, 2 * ind_y:2 * (ind_y + 1)])
    return deconv_y[:, 2*(L-1):]

# 产生h
# f_meta_channels = open("training_channels.pckl", 'rb')      #以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。一般用于非文本文件如图片等。
# h_list = pickle.load(f_meta_channels)          #把f_meta_channels中的值赋给h_list_meta
# f_meta_channels.close()
# print(h_list)

h_list = channel_set_gen(1, 1, True)

class channel_net(nn.Module):
    def __init__(self, M=512, num_neurons_encoder=128, n=32, n_inv_filter=2, num_neurons_decoder=128, if_bias=True, if_RTN=False, snr=25, rali=True):
        super(channel_net, self).__init__()
        self.enc_fc1 = nn.Linear(M, num_neurons_encoder, bias=if_bias)
        self.enc_fc2 = nn.Linear(num_neurons_encoder, n, bias=if_bias)

        ### norm, nothing to train
        ### channel, nothing to train

        num_inv_filter = 2 * n_inv_filter       #P8最后一行末尾
        self.RTN = if_RTN
        if self.RTN:
            self.rtn_1 = nn.Linear(n, n, bias=if_bias)
            self.rtn_2 = nn.Linear(n, n, bias=if_bias)
            self.rtn_3 = nn.Linear(n, num_inv_filter, bias=if_bias)

        self.dec_fc1 = nn.Linear(n, num_neurons_decoder, bias=if_bias)
        self.dec_fc2 = nn.Linear(num_neurons_decoder, M, bias=if_bias)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.norm = nn.LayerNorm(M)

        # 产生噪音
        Eb_over_N = pow(10, (snr / 10))
        noise_var = 1 / (2 * Eb_over_N)
        self.Noise = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n),
                                                                           noise_var * torch.eye(n))
        self.rali = rali
        self.h = h_list[0]

    def forward(self, x, device="cpu"):
        self.h.to(device)
        x = self.enc_fc1(x)
        x = self.relu(x)
        x = self.enc_fc2(x)

        if self.rali:
            # normalize
            x_norm = torch.norm(x, dim=1)       #指定dim = 1，即去掉其dim=1，所以是横向求值，计算Frobenius范数  返回给定张量的矩阵范数或向量范数  计算指定维度的范数

            x_norm = x_norm.unsqueeze(1)        #Returns a new tensor with a dimension of size one inserted at the specified position.

            x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm  # since each has ^2 norm as 0.5 -> complex 1

            # channel
            x = complex_mul_taps(self.h, x).to(device)

        # noise
        n = torch.zeros(x.shape[0], x.shape[1])
        for noise_batch_ind in range(x.shape[0]):
            n[noise_batch_ind] = self.Noise.sample()        #sample()作用：和channel_set_gen中else下类似
        n = n.type(torch.FloatTensor).to(device)
        # n.to(device)
        x = x + n # noise insertion

        # RTN
        if self.RTN:
            h_inv = self.rtn_1(x)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_2(h_inv)
            h_inv = self.tanh(h_inv)
            h_inv = self.rtn_3(h_inv) # no activation for the final rtn (linear activation without weights)
            x = complex_conv_transpose(h_inv, x)

        x = self.dec_fc1(x)
        x = self.relu(x)
        x = self.dec_fc2(x) # softmax taken at loss function
        # x = self.norm(x)
        return x

from torchsummary import summary

if __name__ == '__main__':
    net = channel_net(rali=True)
    summary(net,(512,),batch_size=1,device="cpu")
