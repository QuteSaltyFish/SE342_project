'''
used to read the data from the data folder return 32*32*32
'''
import torch as t
import torchvision as tv
from torchvision import transforms
import os
from PIL import Image
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torchvision.transforms.functional as tf
import random


class data_set(t.utils.data.Dataset):
    def __init__(self, idx,  train):
        self.idx = idx
        self.train = train
        self.config = json.load(open('config.json'))
        self.data_root = self.config["Taining_Dir"]
        self.names = np.array(os.listdir(self.data_root))
        self.sort()
        self.names = self.names[idx]
        self.label_root = self.config['Label_Path']
        self.label_names = np.array(os.listdir(self.label_root))
        self.read_label()
        self.label_names = self.label_names[idx]
        self.init_transform()
        self.read_data()

    def read_data(self):
        self.data = []
        self.label = []
        for index in range(len(self.names)):
            data = Image.open(os.path.join(self.data_root, self.names[index]))
            label = Image.open(os.path.join(
                self.label_root, self.label_names[index]))
            if self.train:
                self.data_augrmentation(data, label)
            data, label = self.transform(
                data), (self.transform(label) * 255)
            self.data.append(data)
            self.label.append(label)

    def data_augrmentation(self, image, mask):
        # 拿到角度的随机数。angle是一个-180到180之间的一个数
        angle = transforms.RandomRotation.get_params([-180, 180])
        # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
        tmp_image = tf.rotate(image, angle, resample=Image.NEAREST)
        tmp_mask = tf.rotate(mask, angle, resample=Image.NEAREST)
        tmp_image, tmp_mask = self.transform(
            tmp_image), (self.transform(tmp_mask) * 255)
        self.data.append(tmp_image)
        self.label.append(tmp_mask)

        # 自己写随机部分，50%的概率应用垂直，水平翻转。
        if random.random() >= 0:
            tmp_image = tf.hflip(image)
            tmp_mask = tf.hflip(mask)
            tmp_image, tmp_mask = self.transform(
                tmp_image), (self.transform(tmp_mask) * 255)
            self.data.append(tmp_image)
            self.label.append(tmp_mask)

        if random.random() >= 0:
            tmp_image = tf.vflip(image)
            tmp_mask = tf.vflip(mask)
            tmp_image, tmp_mask = self.transform(
                tmp_image), (self.transform(tmp_mask) * 255)
            self.data.append(tmp_image)
            self.label.append(tmp_mask)

        # 也可以实现一些复杂的操作
        # 50%的概率对图像放大后裁剪固定大小
        # 50%的概率对图像缩小后周边补0，并维持固定大小
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.25, 1.0), ratio=(1, 1))
        tmp_image = tf.resized_crop(image, i, j, h, w, 256)
        tmp_mask = tf.resized_crop(mask, i, j, h, w, 256)
        tmp_image, tmp_mask = self.transform(
            tmp_image), (self.transform(tmp_mask) * 255)
        self.data.append(tmp_image)
        self.label.append(tmp_mask)

        pad = random.randint(0, 192)
        tmp_image = tf.pad(image, pad)
        tmp_image = tf.resize(image, 256)
        tmp_mask = tf.pad(mask, pad)
        tmp_mask = tf.resize(mask, 256)
        tmp_image, tmp_mask = self.transform(
            tmp_image), (self.transform(tmp_mask) * 255)
        self.data.append(tmp_image)
        self.label.append(tmp_mask)

    def sort(self):
        d = self.names
        sorted_key_list = sorted(d, key=lambda x: (int)(
            os.path.splitext(x)[0].strip('img')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.names = np.array(sorted_key_list)
        # print(self.data_names)

    def read_label(self):
        d = self.label_names
        sorted_key_list = sorted(d, key=lambda x: (int)(
            os.path.splitext(x)[0].strip('label')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.label_names = np.array(sorted_key_list)

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.Resize(
                [4032//self.config['k'], 3024//self.config['k']]),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index].to(t.long)
        return [data, label]

    def __len__(self):
        return len(self.data)


class MyDataSet():
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.data_root = self.config["Taining_Dir"]
        self.data_names = np.array(os.listdir(self.data_root))
        self.DEVICE = t.device(self.config["DEVICE"])
        self.gray = self.config["gray"]
        self.sort()

    def sort(self):
        d = self.data_names
        sorted_key_list = sorted(d, key=lambda x: (int)(
            os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.data_names = np.array(sorted_key_list)
        # print(self.data_names)

    def test_trian_split(self, p=0.8):
        '''
        p is the portation of the training set
        '''
        length = len(self.data_names)

        # create a random array idx
        idx = np.array(range(length))
        np.random.shuffle(idx)
        self.train_idx = idx[:(int)(length*p)]
        self.test_idx = idx[(int)(length*p):]

        self.train_set = data_set(self.train_idx)
        self.test_set = data_set(self.test_idx)
        return self.train_set, self.test_set

    def __len__(self):
        return len(self.data_names)


class In_the_wild_set(t.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.test_root = self.config["Test_Dir"]
        self.test_names = os.listdir(self.test_root)
        self.DEVICE = t.device(self.config["DEVICE"])
        self.gray = self.config["gray"]
        self.init_transform()

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def sort(self):
        d = self.test_names
        sorted_key_list = sorted(d, key=lambda x: (int)(
            os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.test_names = sorted_key_list

        # sorted_dict = map(lambda x:{x:(int)(os.path.splitext(x)[0].strip('candidate'))}, d)
        # print(sorted_dict)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.test_root, self.test_names[index]))
        voxel = self.transform(data['voxel'].astype(np.float32))/255
        seg = self.transform(data['seg'].astype(np.float32))
        data = (voxel*seg).unsqueeze(0)
        name = os.path.basename(self.test_names[index])
        name = os.path.splitext(name)[0]
        return data, name

    def __len__(self):
        return len(self.test_names)


if __name__ == "__main__":
    dataset = data_set(np.arange(11))
    print(len(dataset), dataset[1])
