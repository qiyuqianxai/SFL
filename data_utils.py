import os
import torch, torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from Config import Config
import random
from torchvision import datasets
from matplotlib import pyplot as plt
from Config import Config
# -------------------------------------------------------------------------------------------------------
# DATASETS
# -------------------------------------------------------------------------------------------------------

DATA_PATH = r"D:/datasets"
np.random.seed(2048)


def get_mnist():
    '''Return MNIST train/test data and labels as numpy arrays'''
    data_train = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=True, download=True)
    data_test = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=False, download=True)

    x_train, y_train = data_train.train_data.view(-1, 1, 28, 28).numpy(), np.array(
        data_train.train_labels)
    x_test, y_test = data_test.test_data.view(-1, 1, 28, 28).numpy(), np.array(
        data_test.test_labels)

    return x_train, y_train, x_test, y_test

def get_fashionmnist():
    '''Return MNIST train/test data and labels as numpy arrays'''
    data_train = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=True,
                                                   download=True)
    data_test = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=False,
                                                  download=True)

    x_train, y_train = data_train.train_data.view(-1, 1, 28, 28).numpy(), np.array(
        data_train.train_labels)
    x_test, y_test = data_test.test_data.view(-1, 1, 28, 28).numpy(), np.array(
        data_test.test_labels)

    return x_train, y_train, x_test, y_test

def get_cifar10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=False, download=True)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test

def split_noniid(train_data, train_labels, n_clients, alpha):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n\_clients个子集
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    client_data = [[train_data[ids],train_labels[ids]] for ids in client_idcs]
    plt.figure(figsize=(24,5))
    plt.hist([train_labels[idc] for idc in client_idcs], stacked=True,
            bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
            label=["Client {}".format(i) for i in range(n_clients)], rwidth=0.5)
    plt.xticks(np.arange(10), list(set(train_labels)),fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(-0.5,10)
    plt.xlabel("Label",fontsize=24)
    plt.ylabel("Data volume",fontsize=24)
    plt.legend(loc="upper right", fontsize=12)
    # plt.legend(ncol=10,loc="upper center")		# ncol参数表示图例中的元素列数，如果有4个元素，分为4列，就是一行，默认是1，即1列，为竖列展示图例，可以根据元素修改参数值，进行展示图例。
    plt.savefig("datadist.png",bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return client_data

class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]


def get_default_data_transforms(name, train=True, verbose=True):
    transforms_train = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fashionmnist': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            # transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        # (0.24703223, 0.24348513, 0.26158784)
        'kws': None
    }
    transforms_eval = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fashionmnist': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),  #
        'kws': None
    }

    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train[name].transforms:
            print(' -', transformation)
        print()

    return (transforms_train[name], transforms_eval[name])


def get_data_loaders(cfg, verbose=True):
    x_train, y_train, x_test, y_test = globals()['get_' + cfg.dataset]()
    # x_test = np.array([sample for i,sample in enumerate(x_test) if i%(10000/cfg.label_data_size)==0])
    # y_test = np.array([sample for i,sample in enumerate(y_test) if i%(10000/cfg.label_data_size)==0])
    # print(x_test.shape)
    data_transforms, label_transforms = get_default_data_transforms(cfg.dataset, verbose=False)

    split = split_noniid(x_train,y_train,cfg.n_clients,cfg.distribution_alpha)

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, data_transforms),
                                                  batch_size=cfg.batch_size_for_clients, shuffle=True) for x, y in split]

    train_loader = torch.utils.data.DataLoader(CustomImageDataset(x_train, y_train, data_transforms), batch_size=32,
                                               shuffle=True)
    # test作为服务器上的数据
    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, label_transforms),
                                              batch_size=cfg.test_batch_size, shuffle=True)

    # examples = enumerate(test_loader)  # img&label
    # batch_idx, (imgs, labels) = next(examples)  # 读取数据,batch_idx从0开始
    # # 显示6张图片
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.tight_layout()
    #     # 反归一化，将数据重新映射到0-1之间
    #     img = imgs[i] / 2 + 0.5
    #     plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    # plt.savefig("res.jpg")
    # # plt.show()
    # plt.figure()
    # for i in range(6):
    #     img = imgs[i] / 2 + 0.5
    #     plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    #     plt.axis('off')
    #     # plt.tight_layout()
    #     plt.savefig(f"res{i}.jpg",bbox_inches='tight', pad_inches=0.1)
    stats = {"split": [x.shape[0] for x, y in split]}
    print(stats)
    # client_loaders = sorted(client_loaders, key=lambda x: x.dataset.__len__())
    return client_loaders, train_loader, test_loader

if __name__ == "__main__":
    cfg = Config()
    cfg.dataset = "fashionmnist"
    print(cfg.dataset)
    N_CLIENTS = 10
    DIRICHLET_ALPHA = 0.01
    client_loaders, train_loader, test_loader = get_data_loaders(cfg, verbose=True)

