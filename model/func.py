import json
import os

import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

from model import dataloader
from PIL import Image
from model.unet_model import UNet
def paint(data):
    r = t.tensor([1, 0, 0])
    g = t.tensor([0, 1, 0])
    b = t.tensor([0, 0, 1])

    output = t.zeros([data.shape[0], data.shape[1], 3])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 1:
                output[i, j] = r
            elif data[i, j] == 2:
                output[i, j] = g
            elif data[i, j] == 3:
                output[i, j] = b
    output = output.permute(2, 0, 1)
    return output


def save_model(model, epoch, dir):
    DIR = 'saved_model/'+dir
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    t.save(model.state_dict(), DIR + '/{}.pkl'.format(epoch))


def load_model(model, epoch):
    path = 'saved_model/''{}.pkl'.format(epoch)
    model.load_state_dict(t.load(path))
    return model


def eval_model_new_thread(epoch, gpu):
    config = json.load(open("config.json"))
    path_train = 'result/train_result'
    path_test = 'result/test_result'
    # if not os.path.exists(path_train):
    #     os.makedirs(path_train)
    if not os.path.exists(path_test):
        os.makedirs(path_test)
    python_path = config['python_path']
    os.system('nohup {} -u test_eval.py --epoch={} --gpu={} > {} 2>&1 &'.format(python_path, epoch, gpu,
                                                                                path_test + '/{}.out'.format(epoch)))
    # os.system('nohup {} -u train_eval.py --epoch={} --gpu={} > {} 2>&1 &'.format(python_path, epoch, gpu[1],
    #  path_train + '/{}.out'.format(epoch)))

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
   
    data = transform(data)
    model = UNet(3, 4).to(DEVICE)
    # Test the train_loader
    if (1<=index<=11):
        model.load_state_dict(
            t.load("model/model1.pkl"))
    elif(12<=index<=24):
        model.load_state_dict(
            t.load("model/model2.pkl"))
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
        # print('DEBUG')


if __name__ == '__main__':
    eval_model_new_thread(0, 1)
