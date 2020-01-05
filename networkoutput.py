import json
import os
import numpy as np
import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

from model import dataloader
from PIL import Image
from model.unet_model import UNet
from model.func import paint


def output(dir):
    config = json.load(open("config.json"))
    DEVICE = t.device('cpu')
    name = dir.split('/')[-1]
    name = os.path.splitext(name)[0]
    index = int(name.strip('img'))
    transform = tv.transforms.Compose([
        tv.transforms.Resize(
            [4032//config['k'], 3024//config['k']]),
        tv.transforms.ToTensor()
    ])
    data = Image.open(dir)
    if data.height < data.width:
        data = data.transpose(Image.ROTATE_90)
    data = transform(data)
    model = UNet(3, 4).to(DEVICE)
    # model.load_state_dict(t.load("saved_model/all_v3/7990.pkl"))
    # Test the train_loader
    if (1 <= index <= 11):
        model.load_state_dict(
            t.load("model/model1.pkl"))
    elif(12 <= index <= 24):
        model.load_state_dict(
            t.load("model/model2.pkl"))
    model.load_state_dict(t.load("saved_model/all_v5.2/400.pkl"))
    model.eval()

    with t.no_grad():
        data = data.to(DEVICE).unsqueeze(0)
        out = model(data).squeeze()

        out = out.permute(1, 2, 0)
        out = out.view(-1, 4)

        pred = out.max(1, keepdim=True)[1]
        pred = pred.view(4032//config['k'], 3024//config['k'])
        print(t.max(pred))
        pred = pred.cpu().to(t.uint8).squeeze()
        print(t.max(pred))
        pred = paint(pred)
        if not os.path.exists('final_output'):
            os.makedirs('final_output')
        tv.transforms.ToPILImage()(pred).save('final_output/{}.png'.format(name))


if __name__ == "__main__":
    config = json.load(open('config.json'))
    data_root = config["Taining_Dir"]
    # data_root = '/mnt/ff3bee5c-da50-4d90-848f-2a69bb4db3c8/HomeWork/SE342_project/data/image'
    names = np.array(os.listdir(data_root))
    # dirs = ['data/image/img{}.jpg'.format(i) for i in range(1,25)]
    for i in range(len(names)):
        output_numpy = output(os.path.join(data_root, names[i]))
