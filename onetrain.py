from tensorboard.backend.event_processing import event_accumulator
import os
from model.unet_model import UNet
import torch as t
import torchvision as tv
# %%
import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model.dataloader import *
from model.func import save_model, eval_model_new_thread, eval_model, load_model
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

# %%
t.backends.cudnn.benchmark = True
time_start = time.time()
config = json.load(open("config.json"))
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DEVICE = t.device(config["DEVICE"])
LR = config['lr']
LR = 1e-4
EPOCH = 8000
BATCH_SIZE = config["batch_size"]
WD = config['Weight_Decay']
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
parser.add_argument("--epoch", default=0, type=int,
                    help="The epoch to start from")
parser.add_argument("--name", default='all_v2', type=str,
                    help="Whether to test after training")
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# using K-fold
np.random.seed(1998)
idx = np.arange(11, len(np.array(os.listdir(config["Taining_Dir"]))))
# shuffle the data before the
writer = SummaryWriter('runs/{}'.format(args.name))

train_data = data_set(idx, train=False)
# train_data.data_argumentation()

train_loader = DataLoader.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=config["num_workers"])

model = UNet(3, 4).to(DEVICE)
if args.epoch != 0:
    path = 'saved_model/{}/{}.pkl'.format(args.name, args.epoch)
    model.load_state_dict(t.load(path))
optimizer = t.optim.SGD(model.parameters(), lr=LR)
# optimizer = t.optim.Adam(model.parameters())
print(optimizer.param_groups[0]['lr'])

criterian = t.nn.CrossEntropyLoss().to(DEVICE)

# Test the train_loader
for epoch in range(args.epoch, EPOCH):
    model = model.train()
    train_loss = 0
    correct = 0
    for batch_idx, [data, label] in enumerate(train_loader):
        data, label = data.to(DEVICE), label.to(DEVICE)
        # print(t.max(data), t.max(label))
        out = model(data).squeeze()
        out, label = out.permute(1, 2, 0), label.squeeze()
        out, label = out.view(-1, 4), label.view(-1)
        loss = criterian(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(label.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / \
        len(train_loader.dataset)/(4032*3024//config['k']//config['k'])

    # train_l.append(train_loss)
    # train_a.append(train_acc)

    print('\nEpoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, train_loss, correct, len(train_loader.dataset)*(4032 * 3024 // config['k'] // config['k']), train_acc))
    if epoch % 10 == 0:
        save_model(model, epoch, '{}'.format(args.name))
