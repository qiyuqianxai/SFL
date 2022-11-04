import os
import torch, torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from Config import Config
import random
#-------------------------------------------------------------------------------------------------------
# DATASETS
#-------------------------------------------------------------------------------------------------------

DATA_PATH = "datasets"
np.random.seed(2048)

def get_mnist():
  '''Return MNIST train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=True, download=True) 
  data_test = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=False, download=True) 
  
  x_train, y_train = data_train.train_data.view(-1,1,28,28).expand(-1, 3, -1, -1).numpy()/255, np.array(data_train.train_labels)
  x_test, y_test = data_test.test_data.view(-1,1,28,28).expand(-1, 3, -1, -1).numpy()/255, np.array(data_test.test_labels)

  return x_train, y_train, x_test, y_test


def get_fashionmnist():
  '''Return MNIST train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=True, download=True) 
  data_test = torchvision.datasets.FashionMNIST(root=os.path.join(DATA_PATH, "FashionMNIST"), train=False, download=True) 
  
  x_train, y_train = data_train.train_data.view(-1,1,28,28).expand(-1, 3, -1, -1).numpy()/255, np.array(data_train.train_labels)
  x_test, y_test = data_test.test_data.view(-1,1,28,28).expand(-1, 3, -1, -1).numpy()/255, np.array(data_test.test_labels)

  return x_train, y_train, x_test, y_test


def get_cifar10():
  '''Return CIFAR10 train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=True, download=True) 
  data_test = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=False, download=True) 
  
  x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
  x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)
  
  return x_train, y_train, x_test, y_test


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
  print("\nData: ")
  print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
      np.min(labels_train), np.max(labels_train)))
  print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
      np.min(labels_test), np.max(labels_test)))


#-------------------------------------------------------------------------------------------------------
# SPLIT DATA AMONG CLIENTS
#-------------------------------------------------------------------------------------------------------

def split_image_data(data,labels, n_clients=10, avg_alloc=False):
    clients_data = []
    avg_size = data.shape[0] // n_clients
    for i in range(n_clients - 1):
        if avg_alloc:
            data_indexes = random.sample(range(data.shape[0]), avg_size)
        else:
            data_size = random.randint(int(data.shape[0] * 0.09), int(data.shape[0] * 0.3))
            data_indexes = random.sample(range(data.shape[0]), data_size)
        client_data = data[data_indexes]
        client_label = labels[data_indexes]
        clients_data.append([client_data,client_label])
        labels = np.delete(labels, data_indexes,axis=0)
        data = np.delete(data, data_indexes,axis=0)
    clients_data.append([data,labels])
    for i,client_data in enumerate(clients_data):
        print(f"client {i} data size:",client_data[0].shape[0])
        print(f"client {i} label:", set(client_data[1]))
    return clients_data
#-------------------------------------------------------------------------------------------------------
# IMAGE DATASET CLASS
#-------------------------------------------------------------------------------------------------------
# class CustomImageDataset(Dataset):
#   '''
#   A custom Dataset class for images
#   inputs : numpy array [n_data x shape]
#   labels : numpy array [n_data (x 1)]
#   '''
#   def __init__(self, inputs, labels, data_transforms=None,label_transforms=None):
#       self.inputs = torch.Tensor(inputs)
#       self.data_transforms = data_transforms
#       self.label_transforms = label_transforms
#
#   def __getitem__(self, index):
#       o_img = self.inputs[index]
#       if self.data_transforms is not None:
#         img = self.data_transforms(o_img)
#
#       if self.label_transforms is not None:
#         label = self.label_transforms(o_img)
#
#       return (img, label)
#
#   def __len__(self):
#       return self.inputs.shape[0]
#
#
# def get_default_data_transforms(name, verbose=True):
#   data_transforms = {
#   'mnist': transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     #transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.06078,),(0.1957,))
#     ]),
#   'fashionmnist': transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     #transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
#     ]),
#   'cifar10' : transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     # transforms.RandomCrop(32, padding=4),
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
#   }
#
#   label_transforms = {
#     'mnist': transforms.Compose([
#       transforms.ToPILImage(),
#       transforms.Resize((39, 39)),
#       # transforms.RandomCrop(32, padding=4),
#       transforms.ToTensor(),
#       transforms.Normalize((0.06078,), (0.1957,))
#     ]),
#     'fashionmnist': transforms.Compose([
#       transforms.ToPILImage(),
#       transforms.Resize((39, 39)),
#       # transforms.RandomCrop(32, padding=4),
#       transforms.ToTensor(),
#       transforms.Normalize((0.1307,), (0.3081,))
#     ]),
#     'cifar10': transforms.Compose([
#       transforms.ToPILImage(),
#       transforms.Resize((39, 39)),
#       # transforms.RandomCrop(32, padding=4),
#       # transforms.RandomHorizontalFlip(),
#       transforms.ToTensor(),
#       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
#     # (0.24703223, 0.24348513, 0.26158784)
#   }
#   if verbose:
#     print("\nData preprocessing: ")
#     for transformation in data_transforms[name].transforms:
#       print(' -', transformation)
#     print()
#
#   return (data_transforms[name],label_transforms[name])

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
            transforms.Resize((224, 224)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fashionmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fashionmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
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
  
  x_train, y_train, x_test, y_test = globals()['get_'+cfg.dataset]()

  if verbose:
    print_image_data_stats(x_train, y_train, x_test, y_test)

  data_transforms,label_transforms = get_default_data_transforms(cfg.dataset, verbose=False)

  split = split_image_data(x_train, y_train, n_clients=cfg.n_clients, avg_alloc=cfg.data_diribution_balancedness_for_clents)

  client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x,y, data_transforms),
                                                                batch_size=cfg.batch_size_for_clients, shuffle=True) for x,y in split]
  train_loader = torch.utils.data.DataLoader(CustomImageDataset(x_train, y_train, data_transforms), batch_size=32, shuffle=True)
  test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, label_transforms), batch_size=cfg.test_batch_size, shuffle=False)

  examples = enumerate(test_loader)  # img&label
  batch_idx, (imgs, labels) = next(examples)  # 读取数据,batch_idx从0开始
  # 显示6张图片
  import matplotlib.pyplot as plt
  plt.figure()
  for i in range(6):
      plt.subplot(2, 3, i + 1)
      plt.tight_layout()
      # 反归一化，将数据重新映射到0-1之间
      img = imgs[i] / 2 + 0.5
      plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
  plt.savefig("res.jpg")
  # plt.show()
  plt.figure()
  for i in range(6):
      img = imgs[i] / 2 + 0.5
      plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
      plt.axis('off')
      # plt.tight_layout()
      plt.savefig(f"res{i}.jpg",bbox_inches='tight', pad_inches=0.1)
  stats = {"split": [x.shape[0] for x,y in split]}

  return client_loaders, train_loader, test_loader, stats

if __name__ == '__main__':
    cfg = Config()
    client_loaders, train_loader, test_loader, stats = get_data_loaders(cfg, True)
