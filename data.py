import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

from data_loader import MNISTNoisyLoader


def set_up_data(H):
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    assert H.dataset == 'cifar10'
    H.image_size = 32
    H.image_channels = 3
    shift = -120.63838
    scale = 1. / 64.16736

    do_low_bit = H.dataset in ['ffhq_256']

    # if H.test_eval:
    #     print('DOING TEST')
    #     eval_dataset = teX
    # else:
    #     eval_dataset = vaX

    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)
    DATA_DIR = '/tmp2/ashesh/ashesh/VAE_based/data/MNIST/noisy/'
    H.fpath_dict = {
        # 10: DATA_DIR + 'train_10.npy',
        20: DATA_DIR + 'train_20.npy',
        30: DATA_DIR + 'train_30.npy',
        # 40: DATA_DIR + 'train_40.npy',
        # 50: DATA_DIR + 'train_50.npy',
        # 60: DATA_DIR + 'train_60.npy'
    }
    train_data = MNISTNoisyLoader(H.fpath_dict)
    valid_data = MNISTNoisyLoader(H.fpath_dict)
    untranspose = False

    # if H.dataset == 'ffhq_1024':
    #     train_data = ImageFolder(trX, transforms.ToTensor())
    #     valid_data = ImageFolder(eval_dataset, transforms.ToTensor())
    #     untranspose = True
    # else:
    #     train_data = TensorDataset(torch.as_tensor(trX))
    #     valid_data = TensorDataset(torch.as_tensor(eval_dataset))
    #     untranspose = False

    def preprocess_func(x):
        # NOTE: I've changed x. Now, x contains just input. It does not contain the target.
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        nonlocal do_low_bit
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x = x.permute(0, 2, 3, 1)
        inp = x.cuda(non_blocking=True).float()
        out = inp.clone()
        # import pdb
        # pdb.set_trace()
        inp.add_(shift).mul_(scale)
        if do_low_bit:
            # 5 bits of precision
            out.mul_(1. / 8.).floor_().mul_(8.)
        out.add_(shift_loss).mul_(scale_loss)
        return inp, out

    return H, train_data, valid_data, preprocess_func


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def unpickle_cifar10(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    data = dict(zip([k.decode() for k in data.keys()], data.values()))
    return data


def imagenet32(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet32-train.npy'), mmap_mode='r')
    np.random.seed(42)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet32-valid.npy'), mmap_mode='r')
    return train, valid, test


def imagenet64(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet64-train.npy'), mmap_mode='r')
    np.random.seed(42)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet64-valid.npy'), mmap_mode='r')  # this is test.
    return train, valid, test


def ffhq1024(data_root):
    # we did not significantly tune hyperparameters on ffhq-1024, and so simply evaluate on the test set
    return os.path.join(data_root,
                        'ffhq1024/train'), os.path.join(data_root,
                                                        'ffhq1024/valid'), os.path.join(data_root, 'ffhq1024/valid')


def ffhq256(data_root):
    trX = np.load(os.path.join(data_root, 'ffhq-256.npy'), mmap_mode='r')
    np.random.seed(5)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    # we did not significantly tune hyperparameters on ffhq-256, and so simply evaluate on the test set
    return train, valid, valid


def cifar10(data_root, one_hot=True):
    tr_data = [
        unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'data_batch_%d' % i)) for i in range(1, 6)
    ]
    trX = np.vstack(data['data'] for data in tr_data)
    trY = np.asarray(flatten([data['labels'] for data in tr_data]))
    te_data = unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'test_batch'))
    teX = np.asarray(te_data['data'])
    teY = np.asarray(te_data['labels'])
    trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    trX, vaX, trY, vaY = train_test_split(trX, trY, test_size=5000, random_state=11172018)
    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
        vaY = np.eye(10, dtype=np.float32)[vaY]
        teY = np.eye(10, dtype=np.float32)[teY]
    else:
        trY = np.reshape(trY, [-1, 1])
        vaY = np.reshape(vaY, [-1, 1])
        teY = np.reshape(teY, [-1, 1])
    return (trX, trY), (vaX, vaY), (teX, teY)
