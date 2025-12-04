#import os
import sys
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import argparse
import torchaudio
 
sys.path.append("../PANN/audioset_tagging_cnn/pytorch/")
from models import *
sys.path.append("utils/")
from dataset import *
from filtragem import noise_reduction
from modelo import *

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Arguments & parameters
sample_rate = 16000
window_size = 1024
hop_size = 320
mel_bins = 128 #64
fmin = 0
fmax = 8000 #16000/2
model_type = "Transfer_AudioMAE"
pretrained_checkpoint_path = "../AudioMAE/checkpoints/pretrained.pth"
freeze_base = False
device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
classes_num = 2 # saudavel ou nao
pretrain = True if pretrained_checkpoint_path else False

mel_spectrogrammer = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=window_size,
    hop_length=hop_size,
    n_mels=mel_bins,
    f_min=fmin,
    f_max=fmax,
)

# Model
Model = eval(model_type)
model = Model(classes_num=classes_num,
              freeze_base=freeze_base,
              pretrained_checkpoint=pretrained_checkpoint_path)

if 'cuda' in device:
    model.to(device)
    print("Utilizando: ",device)


# Otmizador
head_params = list(model.head.parameters())
base_params = list(model.base.parameters()) 

# Configuração do otimizador com grupos de parâmetros
model_opt = torch.optim.AdamW([
    { 'params': head_params, 'lr': 1e-3, 'weight_decay': 0.0 }, # LR maior para a cabeça, sem weight decay
    { 'params': base_params, 'lr': 5e-5, 'weight_decay': 0.05 }  # LR bem menor para a base
], betas=(0.9, 0.95), eps=1e-8)


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
parser.add_argument("--param", type=str, default="Filtragem") # parametro atual testado
parser.add_argument("--it", type=int, default=0) # iteracao atual  
parser.add_argument("--filter", type=int, default=1) # filtrar ou nao
parser.add_argument("--noise", type=int, default=0) # ruido ou nao
parser.add_argument("--masking", type=float, default=0.0) # porcentagem de mascara para o treino
parser.add_argument("--threshold", type=float, default=0.34)
parser.add_argument("--threshold_db", type=float,default=0)

args = parser.parse_args()
filter_train = True if args.filter==1 else False # filtrar ou nao os dados de treino
noise_train = True if args.noise==1 else False
reduction_method = "Db" if args.threshold_db>0 else "Pct"

model_path = f'testes/checkpoints/model_filtro{args.param}{args.it}.ckpt'

# Arquivos de audios
audio_target_dictionary = {} # Guarda a relação path-label
train_files = Load_Train_dataset(audio_target_dictionary, noise_train)
eval_files = Load_Eval_dataset(audio_target_dictionary, noise_train)
print("Arquivos de treino/eval carregados com sucesso")

if(filter_train): # treino para audios com filtro
    train_files_filtered, new_train_files = noise_reduction(train_files, model_type, args.freq, args.time, args.thresh, args.propdec, args.threshold, args.threshold_db, windows=True , training=True)
    eval_files_filtered, eval_files = noise_reduction(eval_files, model_type, args.freq, args.time, args.thresh, args.propdec, args.threshold, args.threshold_db, windows=True, training=False)

else: # treino para audios sem filtro
    train_files_filtered = Load_Normal_audios(train_files) 
    eval_files_filtered = Load_Normal_audios(eval_files)
    new_train_files = train_files

#norm_mean, norm_std = calculate_norm_stats(train_files_filtered)

for epoch in range(50):
    model.train()
    train_acc, train_f1, true_train_f1 = run_epoch(epoch,model,model_type, 
                  LossCompute(model, model_opt),
                  train_files_filtered, new_train_files, audio_target_dictionary, device, training=True, masking=args.masking)
    
    train_accs.append(train_acc)
    train_f1s.append(true_train_f1)

    model.eval()
    with torch.no_grad():
        val_acc, val_f1_score, true_val_f1_score = run_epoch(epoch, model, model_type, 
                    LossCompute(model, None),
                    eval_files_filtered, eval_files, audio_target_dictionary, device, training=False, masking=args.masking)
        
        val_accs.append(val_acc)
        val_f1s.append(true_val_f1_score)

    if best_val_acc < val_acc:
        best_val_acc = val_acc
    #if best_val_f1_score < true_val_f1_score:
    #    best_val_f1_score = true_val_f1_score
        print('Saving model')
        torch.save({
            'seed':  SEED,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model_opt.state_dict(),
            'metrics': {
                'val_acc': val_acc,
                'val_f1_true': true_val_f1_score,
                'train_acc': train_acc,
                'train_f1': true_train_f1
            },
            'filter_args': {
                'n_grad_freq': args.freq,
                'n_grad_time':   args.time,
                'n_std_thresh':   args.thresh,
                'prop_decrease':   args.propdec,
                'reduction_method': reduction_method,
                'reduction_value': args.threshold_db if args.threshold_db>0 else args.threshold
            }
        }, model_path)

    #train_files_filtered, new_train_files = noise_reduction(train_files, args.freq, args.time, args.thresh, args.propdec, windows=True, training=True)


epochs = list(range(1, len(train_accs) + 1))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, [x.cpu().item() if torch.is_tensor(x) else x for x in train_accs], label='Train Acc')
plt.plot(epochs, [x.cpu().item() if torch.is_tensor(x) else x for x in val_accs], label='Val Acc')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_f1s, label='Train F1')
plt.plot(epochs, val_f1s, label='Val F1')
plt.title('F1-score por Época')
plt.xlabel('Época')
plt.ylabel('F1-score')
plt.legend()

plt.tight_layout()
plt.savefig('treino2.png')