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

        train_batch = []
        ground_truth = []
        for i in range(self.opt.batch_size):
            f = np.random.uniform(0, 1, 10)
            train_batch.append(f)
            ground_truth.append(draw(f))

        train_batch = torch.tensor(train_batch).float()
        ground_truth = torch.tensor(ground_truth).float()

        if self.jit:
            net = torch.jit.script(net)

        net = net.to(self.device)
        train_batch = train_batch.to(self.device)
        ground_truth = ground_truth.to(self.device)

        self.module = net
        self.example_inputs = (train_batch,ground_truth)
        self.optimizer = optim.Adam(self.module.parameters(), lr=3e-6)

    def get_module(self):
        return self.module,self.example_inputs

    def train(self, niter=1):
        self.module.train()

        for _ in range(niter):
            gen = self.module(self.example_inputs[0])
            self.optimizer.zero_grad()
            loss = self.criterion(gen, self.example_inputs[1])
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1):
        self.module.eval()
        for _ in range(niter):
            self.module(self.example_inputs[0])

if __name__ == '__main__':
    m = Model(device='cpu', jit=False)
    module,example_inputs = m.get_module()
    while m.step < 100:
        m.train(niter=1)
        if m.step%100 == 0:
            m.eval(niter=1)
        m.step += 1

