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
import librosa
import os

sys.path.append("../PANN/audioset_tagging_cnn/pytorch/")
from models import *
sys.path.append("utils/")
from dataset import *
from filtragem import noise_reduction#, verify_windows_labels
from modelo import *


# Arguments & parameters
sample_rate = 16000
window_size = 1024
hop_size = 160
mel_bins = 128
fmin = 0
fmax = 8000 #16000/2
model_type = "Transfer_AudioMAE"
freeze_base = False
device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
classes_num = 2 # saudavel ou nao


test_files_filtered = []


# Argumentos do filtro
parser = argparse.ArgumentParser()
parser.add_argument("--freq", type=int, default=3) # shape do filtro aplicado sobre o ruído 3
parser.add_argument("--time", type=int, default=3) # shape do filtro aplicado sobre o ruído 3
parser.add_argument("--thresh", type=float, default=2.0) # limiar em multiplos de STD para o ruído 2
parser.add_argument("--propdec", type=float, default=1.0) # intensidade da supressão do ruído 1.0
parser.add_argument("--param", type=str, default="Melhor") # parametro atual testado
parser.add_argument("--it", type=int, default=0) # iteracao atual
parser.add_argument("--filter", type=int, default=1) # filtrar ou nao
parser.add_argument("--noise", type=int, default=0)
parser.add_argument("--masking", type=float, default=0.0)
parser.add_argument("--threshold", type=float, default=0.34)
parser.add_argument("--threshold_db", type=float,default=0)

args = parser.parse_args()

filter_test = True if args.filter==1 else False # filtrar ou nao os dados de teste
noise_test = True if args.noise==1 else False # adicionar ruido ou nao os dados de teste

acc_mean = 0
f1_mean = 0

model_path = f'testes/checkpoints/model_filtro{args.param}{args.it}.ckpt'

Model = eval(model_type)
model = Model(classes_num=classes_num,
              freeze_base=freeze_base,
              pretrained_checkpoint=model_path,
              training = False)

# Load trained model
logging.info('Load pretrained model from {}'.format(model_path))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

if 'cuda' in device:
    model.to(device)
    print("Utilizando: ",device)

model.eval()

audio_target_dictionary = {}

test_files = Load_Test_dataset(audio_target_dictionary, noise_test)

# Testes com ou sem filtro
if(filter_test):
    test_files_filtered, test_files = noise_reduction(test_files, model_type, args.freq, args.time, args.thresh, args.propdec, args.threshold, args.threshold_db, windows=True, training=False)
else:
    test_files_filtered = Load_Normal_audios(test_files)

print(f"Quantidade de audios: {len(test_files_filtered)}/{len(test_files)}")
csv_filepath = 'test.csv'

with torch.no_grad():
    print(write_to_csv(model, csv_filepath, test_files_filtered, test_files, device, transformer = True, masking=args.masking))


prediction_labels = pandas.read_csv(csv_filepath)

df_agg = prediction_labels.groupby("file_path").agg({
    "label": majority_vote_test             # aplica voto de maioria
}).reset_index()

print(df_agg.head())
prediction_labels = df_agg

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

TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP) if (TN+FP) > 0 else 0.0 # dos que chutou negativo, quantos estavam certo ?

acc = accuracy_score(true_labels_list, prediction_labels_list)
f1 = f1_score(true_labels_list, prediction_labels_list)
precision = precision_score(true_labels_list, prediction_labels_list)
recall = recall_score(true_labels_list, prediction_labels_list)

print(acc)


out_path = f"resultados/resultado_{args.param}.csv"
file_exists = os.path.exists(out_path)

with open(out_path, "a") as f:
    if not file_exists:
        f.write("param,it,thresh,propdec,freq,time,masking,threshold,acc,f1,precision,recall,specificity\n")
    f.write(f"{args.param},{args.it},{args.thresh},{args.propdec},{args.freq},{args.time},{args.masking},{args.threshold},"
            f"{acc:.4f},{f1:.4f},{precision:.4f},{recall:.4f},{specificity:.4f}\n")