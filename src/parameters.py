import torch

corpusFileName = './data/corpusPoems'
modelFileName = '../models/fineTunedDP_3'
trainDataFileName = '/workspace/Neural Poet/src/data/train_dataset_cu.txt'
testDataFileName = './data/testCorpus'
char2idFileName = './data/char2id'
auth2idFileName = './data/auth2id'

device = torch.device("cuda")
# device = torch.device("cpu")

batchSize = 64
char_emb_size = 128


hid_size = 512
lstm_layers = 3
dropout = 0.4

epochs = 20
learning_rate = 0.002 # намалява с тренирането

defaultTemperature = 0.2

INS_COST = 1.0
GAMMA = 0.5