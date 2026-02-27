#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import math
from parameters import *


def trainModel(trainCorpus, lm, optimizer, epochs, batchSize, stress_model):
    idx = np.arange(len(trainCorpus), dtype='int32')
    lm.train()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    step = 0
    for epoch in range(epochs):
        np.random.shuffle(idx)
        for b in range(0, len(idx), batchSize):
            batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            loss, H, sed_loss = lm(batch, stress_model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if H is not None and sed_loss is not None:
                print("Epoch:",epoch,'/',epochs,", Batch:",b // batchSize, '/', len(idx) // batchSize, ", loss: ",loss.item(), ", H: ",H.item(), ", sed_loss: ",sed_loss.item())
            else:
                print("Epoch:",epoch,'/',epochs,", Batch:",b // batchSize, '/', len(idx) // batchSize, ", loss: ",loss.item())
            step += 1
            if step % 100 == 0:
                lm.save(modelFileName + f".step{step}")
        scheduler.step()
        lm.save(modelFileName + f".{epoch}")


def perplexity(lm, testCorpus, batchSize):
    lm.eval()
    H = 0.
    c = 0
    for b in range(0,len(testCorpus),batchSize):
        batch = testCorpus[b:min(b+batchSize, len(testCorpus))]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            h = lm.get_H(batch)
            H += l * h
    return math.exp(H/c)
