import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model.dataloader import *
from model.unet_model import UNet
from model.func import save_model, eval_model_new_thread, eval_model, load_model, paint
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
import pandas as pd
if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    parser.add_argument("--epoch", default=28, type=int,
                        help="The epoch to be tested")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_set = data_set(np.arange(11))
    test_loader = DataLoader.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    model = UNet(3, 4).to(DEVICE)
    # Test the train_loader
    model.load_state_dict(
        t.load("saved_model/all/2420.pkl"))
    model.eval()

    with t.no_grad():
        # Test the test_loader
        test_loss = 0
        correct = 0
        idx = []
        Name = []
        Score = []
        for batch_idx, [data, label] in enumerate(test_loader):
            data = data.to(DEVICE)
            out = model(data).squeeze()

            out = out.permute(1, 2, 0)
            out = out.view(-1, 4)

            pred = out.max(1, keepdim=True)[1]
            pred = pred.view(4032//config['k'], 3024//config['k'])
            print(t.max(pred))
            pred = pred.cpu().to(t.uint8).squeeze()
            print(t.max(pred))
            pred = paint(pred)
            tv.transforms.ToPILImage()(pred).save('test.png')
            print('DEBUG')
