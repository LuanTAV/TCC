import sys
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import soundfile as sf
import pandas
import argparse

sys.path.append("../PANN/audioset_tagging_cnn/pytorch/")
from models import *
sys.path.append("utils/")
from dataset import *
from filtragem import noise_reduction
from modelo import *

# SEED = 42

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# Arguments & parameters
sample_rate = 32000
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 0
fmax = 32000
model_type = "Transfer_Cnn10"
freeze_base = False
device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
classes_num = 2 # saudavel ou nao

filter_test = True # filtrar ou nao os dados de teste
test_files_filtered = []



Model = eval(model_type)
model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)

# Argumentos do filtro
parser = argparse.ArgumentParser()
parser.add_argument("--freq", type=int, default=8) # shape do filtro aplicado sobre o ruído 3
parser.add_argument("--time", type=int, default=8) # shape do filtro aplicado sobre o ruído 3
parser.add_argument("--thresh", type=float, default=3.0) # limiar em multiplos de STD para o ruído 2
parser.add_argument("--propdec", type=float, default=0.5) # intensidade da supressão do ruído 1.0
parser.add_argument("--param", type=str, default="Melhor") # parametro atual testado
parser.add_argument("--it", type=int, default=0) # iteracao atual

args = parser.parse_args()

acc_mean = 0
f1_mean = 0

#for i in range(args.qtd):

model_path = f'testes/checkpoints/model_filtro{args.param}{args.it}.ckpt'

# Load trained model
logging.info('Load pretrained model from {}'.format(model_path))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

if 'cuda' in device:
    model.to(device)
    print("Utilizando: ",device)

model.eval()


audio_target_dictionary = {}

test_files = Load_Test_dataset(audio_target_dictionary)

# Testes com ou sem filtro
if(filter_test):
    test_files_filtered = noise_reduction(test_files, args.freq, args.time, args.thresh, args.propdec)
else:
    test_files_filtered = Load_Normal_audios(test_files)

csv_filepath = 'test.csv'

with torch.no_grad():
    print(write_to_csv(model, csv_filepath, test_files_filtered, test_files, device, filter_test))

prediction_labels = pandas.read_csv(csv_filepath)
test_folder = '../../SPIRA/SPIRA_Dataset_V2/'


prediction_labels_list = []
true_labels_list = []

for index in range(len(prediction_labels['file_path'])):

    file = prediction_labels['file_path'][index]
    prediction_label = prediction_labels['label'][index]
    prediction_labels_list.append(prediction_label)

    file_path = file
    true_label = audio_target_dictionary[file_path]

    if true_label is None:
        print(f"Aviso: arquivo {file_path} não encontrado no dicionário!")
        continue

    true_labels_list.append(true_label)

prediction_labels_list = np.array(prediction_labels_list)
true_labels_list = np.array(true_labels_list)

cm = sklearn.metrics.confusion_matrix(true_labels_list, prediction_labels_list, labels=[0, 1])
print(cm)


acc = accuracy_score(true_labels_list, prediction_labels_list)
f1 = f1_score(true_labels_list, prediction_labels_list)

acc_mean = acc_mean + acc
f1_mean = f1_mean + f1

# plt.figure(figsize=(6, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
# plt.title(f'Matriz de Confusão\nAcurácia: {acc:.4f} | F1-score: {f1:.4f}')
# plt.xlabel('Previsto')
# plt.ylabel('Real')

# plt.tight_layout()
# plt.savefig(f'testes/results{args.param}{args.it}.png', dpi=300)
# plt.close()

if(args.param == "optuna"):
    print(f"Acurácia: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
# else:
#     with open(f"testes/results_full_media_{args.param}.csv", "a") as f:
#         f.write(f"{args.param},{args.it},{args.thresh},{args.propdec},{args.freq},{args.time},{acc:.4f},{f1:.4f}\n")


# acc_mean = acc_mean / args.qtd
# f1_mean = f1_mean / args.qtd

# with open(f"testes/results_resumo_media_{args.param}.csv", "a") as f:
#     f.write(f"{args.param},{args.it},{args.thresh},{args.propdec},{args.freq},{args.time},{acc_mean:.4f},{f1_mean:.4f}\n")
