import os
import sys
import numpy as np
import random
import time
import logging
import matplotlib.pyplot as plt
import sklearn
import csv
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import seaborn as sns
import soundfile as sf
import argparse
 
sys.path.append("../PANN/audioset_tagging_cnn/pytorch/")
from models import *
sys.path.append("utils/")
from dataset import *
from filtragem import noise_reduction
from modelo import *




# Arguments & parameters
sample_rate = 32000
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 0
fmax = 32000
model_type = "Transfer_Cnn10"
pretrained_checkpoint_path = "Cnn10_mAP=0.380.pth"
freeze_base = False
device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
classes_num = 2 # saudavel ou nao
pretrain = True if pretrained_checkpoint_path else False

# Model
Model = eval(model_type)
model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)

# Load pretrained model
if pretrain:
    logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
    model.load_from_pretrain(pretrained_checkpoint_path)
    print('Load pretrained model successfully!')

if 'cuda' in device:
    model.to(device)
    print("Utilizando: ",device)

# Otmizador
model_opt = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)

# Variaveis e estatísticas
best_val_acc = 0
best_val_f1_score = 0

min_frequency = 0.0
max_frequency = None

train_accs = []
train_f1s = []
val_accs = []
val_f1s = []

# Argumentos do filtro

parser = argparse.ArgumentParser()
parser.add_argument("--freq", type=int, default=3) # shape do filtro aplicado sobre o ruído
parser.add_argument("--time", type=int, default=3) # shape do filtro aplicado sobre o ruído
parser.add_argument("--thresh", type=float, default=2) # limiar em multiplos de STD para o ruído
parser.add_argument("--propdec", type=float, default=1.0) # intensidade da supressão do ruído
parser.add_argument("--param", type=str, default="None") # parametro atual testado
parser.add_argument("--it", type=int, default=0) # iteracao atual

args = parser.parse_args()

# SEED = 42

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

model_path = f'testes/checkpoints/model_filtro{args.param}{args.it}.ckpt'

# Arquivos de audios
audio_target_dictionary = {}
train_files = Load_Train_dataset(audio_target_dictionary)
eval_files = Load_Eval_dataset(audio_target_dictionary)

# treino para audios com filtro
train_files_filtered = noise_reduction(train_files, args.freq, args.time, args.thresh, args.propdec)
eval_files_filtered = noise_reduction(eval_files, args.freq, args.time, args.thresh, args.propdec)

# treino para audios sem filtro
# train_files_filtered = Load_Normal_audios(train_files) 
# eval_files_filtered = Load_Normal_audios(eval_files)

for epoch in range(150):
    model.train()
    train_acc, train_f1, _ = run_epoch(model, 
                  LossCompute(model, model_opt),
                  train_files_filtered, train_files, audio_target_dictionary, device, training=True)
    
    train_accs.append(train_acc)
    train_f1s.append(train_f1)

    model.eval()
    with torch.no_grad():
        val_acc, val_f1_score, true_val_f1_score = run_epoch(model, 
                    LossCompute(model, None),
                    eval_files_filtered, eval_files, audio_target_dictionary, device, training=False)
        
        val_accs.append(val_acc)
        val_f1s.append(val_f1_score)

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        print('Saving model')
        torch.save({'model_state_dict': model.state_dict()}, model_path)


# epochs = list(range(1, len(train_accs) + 1))

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(epochs, [x.cpu().item() if torch.is_tensor(x) else x for x in train_accs], label='Train Acc')
# plt.plot(epochs, [x.cpu().item() if torch.is_tensor(x) else x for x in val_accs], label='Val Acc')
# plt.title('Acurácia por Época')
# plt.xlabel('Época')
# plt.ylabel('Acurácia')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_f1s, label='Train F1')
# plt.plot(epochs, val_f1s, label='Val F1')
# plt.title('F1-score por Época')
# plt.xlabel('Época')
# plt.ylabel('F1-score')
# plt.legend()

# plt.tight_layout()
# plt.savefig('treino.png')