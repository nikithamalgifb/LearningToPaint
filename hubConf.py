import cv2
import torch
import numpy as np
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from baseline.utils.tensorboard import TensorBoard
from baseline.Renderer.model import FCN
from baseline.Renderer.stroke_gen import *

from argparse import Namespace

torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cuda = torch.cuda.is_available()

class Model:
    def __init__(self, device=None, jit=False):
        self.device = device
        self.jit = jit
        self.criterion = nn.MSELoss()
        net = FCN()
        self.step = 0
        self.opt = Namespace(**{
            'batch_size': 64,
            'debug': '',
            'script': False,
            })

        if self.jit:
            net = torch.jit.script(net)
        self.module = net


    def get_module(self):
        return self.module

    def train(self, niter=1):
        self.module.train()
        self.train_batch = []
        self.ground_truth = []
        for i in range(self.opt.batch_size):
            f = np.random.uniform(0, 1, 10)
            self.train_batch.append(f)
            self.ground_truth.append(draw(f))

        self.train_batch = torch.tensor(self.train_batch).float()
        self.ground_truth = torch.tensor(self.ground_truth).float()

        if use_cuda:
            self.module = self.module.cuda()
            self.train_batch = self.train_batch.cuda()
            self.ground_truth = self.ground_truth.cuda()
        self.gen = self.module(self.train_batch)
        optimizer = optim.Adam(self.module.parameters(), lr=3e-6)

        for _ in range(niter):
            optimizer.zero_grad()
            loss = self.criterion(self.gen, self.ground_truth)
            loss.backward()
            optimizer.step()
            print(self.step, loss.item())
            if self.step < 200000:
                lr = 1e-4
            elif self.step < 400000:
                lr = 1e-5
            else:
                lr = 1e-6
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def eval(self, niter=1):
        self.module.eval()
        for _ in range(niter):
            self.gen = self.module(self.train_batch)
            loss = self.criterion(self.gen, self.ground_truth)
            for i in range(32):
                G = self.gen[i].cpu().data.numpy()
                GT = self.ground_truth[i].cpu().data.numpy()

if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    module = m.get_module()
    while m.step < 100:
        m.train(niter=1)
        if m.step%100 == 0:
            m.eval(niter=1)
        m.step += 1

