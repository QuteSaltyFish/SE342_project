import json
import os
import numpy as np
import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

from model import dataloader
from PIL import Image
from model.unet_model import UNet
from model.func import output

if __name__ == "__main__":
    config = json.load(open('config.json'))
    data_root = config["Taining_Dir"]
    names = np.array(os.listdir(data_root))
    # dirs = ['data/image/img{}.jpg'.format(i) for i in range(1,25)]
    for i in range(len(names)):
        output_numpy = output(os.path.join(data_root, names[i]))


