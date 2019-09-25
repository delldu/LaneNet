#! /usr/bin/env python
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    File Author: Dell, Thu Sep 26 00:43:48 CST 2019
# ***
# ************************************************************************************/
# 

import time
from tqdm import tqdm
import torch

def SpeedTest(model, input, with_gpu=True):
    epochs = 100
    model.eval()

    if with_gpu:
        model = model.cuda()
        input = input.cuda()
    else:
        model = model.cpu()
        input = input.cpu()

    start = time.time()
    for i in tqdm(range(epochs)):
        with torch.no_grad():
           output = model(input)
    spends = (time.time() - start)*1000/epochs

    if with_gpu:
        print("GPU average spend {} ms".format(spends))
    else:
        print("CPU average spend {} ms".format(spends))

