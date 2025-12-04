import sys, os
import numpy as np
import time
import sklearn
import csv
import torch
import torchaudio
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

from dataset import majority_vote_eval

sys.path.append("../../PANN/audioset_tagging_cnn/pytorch/")
from models import *

# Modificado de 
# https://github.com/marcelomatheusgauy/Pretrained_audio_neural_networks_emotion_recognition/blob/main/Pretrained_audio_neural_networks/train_utils.py

sys.path.append(os.path.abspath("../../TCC/AudioMAE"))

from models_mae import *  # vem do repo AudioMAE

#import torch.nn.functional as F
#import torchvision.transforms as T


class Transfer_AudioMAE(nn.Module):
    def __init__(self, classes_num=2, freeze_base=False, pretrained_checkpoint=None, training = True):
        """
        Wrapper para usar o AudioMAE em nova tarefa.
        """
        super().__init__()
        
        # cria modelo ViT-base (o encoder do AudioMAE)
        self.base = mae_vit_base_patch16_dec512d8b(
            audio_exp=True,
            img_size=(1024, 128),  # tamanho do spectrograma
            in_chans=1
        )

        # se tiver checkpoint pré-treinado
        if pretrained_checkpoint is not None:
            if training:
                checkpoint = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)

                state_dict = checkpoint.get('model', checkpoint) 
                msg = self.base.load_state_dict(state_dict, strict=False)
                
                print("Checkpoint pré-treinado carregado com sucesso:", msg)
            else:
                checkpoint = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)

                state_dict = checkpoint.get('model', checkpoint) 
                msg = self.load_state_dict(state_dict, strict=False)
        
        # opção de congelar encoder
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        # nova camada para classificaçao
        self.head = nn.Linear(self.base.embed_dim, classes_num)
     

    def forward(self, x, masking):
        features_encoder, _, _, _ = self.base.forward_encoder(x, mask_ratio=masking)
        
        # [CLS] token (o primeiro token) para a classificação, como é padrão no ViT
        cls_token = features_encoder[:, 0]
        
        # Passa o token [CLS] pela cabeça de classificação
        output = self.head(cls_token)

        return output

# Classe utilizando o modelo PANN (CNN10) para fine-tuning
class Transfer_Cnn10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn10 as a sub module.
        """
        super(Transfer_Cnn10, self).__init__()
        audioset_classes_num = 527 # numero de classes originais da CNN10
        
        self.base = Cnn10(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        self.fc_transfer = nn.Linear(512, classes_num, bias=True) # adiciona uma camada nova para nossa classificaçao

        if freeze_base: # congela as camadas anteriores se necessário
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu')
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, masking=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, None)
        embedding = output_dict['embedding']
        logits = self.fc_transfer(embedding)  # logits crus
        #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1) # talvez mudar para sigmoid para classificaçao binaria (classes_num = 2)
        #output_dict['clipwise_output'] = clipwise_output
        output_dict['clipwise_output'] = logits # na verdade sao logits e nao probabilidades aqui
 
        return output_dict
    
def process_batches(filtered_audios, files, audio_target_dictionary, batch_size, file_index, device, model_type):

    data_batch = []
    
    audio_target_list = []
    #print(data_batch, batch_size, file_index, len(filtered_audios))

    while len(data_batch) < batch_size and file_index < len(filtered_audios):

        data_path = files[file_index]
        data_elem = filtered_audios[file_index]
        #print(f"AQUI: {data_path}, {data_elem}")
        if isinstance(data_elem, np.ndarray):
            # Se for, aplica a transformação
            data_elem = torch.from_numpy(data_elem).float()
        
        data_batch.append(data_elem)

        audio_target_list.append(audio_target_dictionary[data_path])
        
        file_index +=1
        
    #convert list to torch tensor (pads different audio lengths to same size)
    #CNN10
    if model_type == "Transfer_Cnn10":
        data_batch = nn.utils.rnn.pad_sequence(data_batch, batch_first=True) 
    #AudioMAE
    elif model_type == "Transfer_AudioMAE":
        data_batch = torch.stack(data_batch, dim=0)

    data_batch = data_batch.to(device)
    
    audio_target_list = torch.LongTensor(audio_target_list)
    audio_target_list = audio_target_list.to(device)
    
    return data_batch, audio_target_list, file_index

#function to train model
def run_epoch(epoch, model, model_type, loss_compute, filtered_audios, files, audio_target_dictionary, device, training=True, batch_size=16, masking = 0.0):
    
    train_acc_avg = 0.0
    f1_score_avg = 0.0
    
    number_elements = len(files)
    
    outputs=[] # preds por batch
    targets=[] # y_true por batch
    
    file_index = 0
    step_index = 0

    while file_index < number_elements:
        step_index +=1
        
        data_batch, audio_target_list, file_index = process_batches(filtered_audios, files, audio_target_dictionary, batch_size, file_index, device, model_type)
        
        output_dict = model.forward(data_batch, masking)

        _, train_acc, f1_score_step, output, target = loss_compute(output_dict, audio_target_list, training, model_type)
        
        outputs.append(output)
        targets.append(target)

        train_acc_avg = (train_acc_avg*(step_index-1)+train_acc)/(step_index)
        f1_score_avg = (f1_score_avg*(step_index-1)+f1_score_step)/(step_index)
        
        print(f"[Epoch {epoch+1}] Step {step_index}/{(len(files)+batch_size-1)//batch_size} "
            f"| Batch size: {data_batch.size(0)} "
            f"| Train Acc: {train_acc_avg:.4f} "
            f"| F1: {f1_score_avg:.4f}")

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

    true_f1_score = sklearn.metrics.f1_score(targets, outputs, average='macro', zero_division=0)
    print('Final F1_score=', true_f1_score)

    if training == False:
        print("VAL: shape preds/targets:", len(outputs), len(targets))
        print(confusion_matrix(targets, outputs))
        print(classification_report(targets, outputs, digits=4))

        majority_vote_eval(outputs, targets, files, audio_target_dictionary)
    
    return train_acc_avg, f1_score_avg, true_f1_score


class LossCompute:
    "A simple loss compute and train function."
    def __init__(self, model, opt=None):
        self.model = model
        self.opt = opt
        
    def __call__(self, output_dict, y, training, model_type):
        train_acc = 0
        f1_score=0

        cross_entropy_loss = nn.CrossEntropyLoss()

        #CNN10
        if model_type == "Transfer_Cnn10":
            loss = cross_entropy_loss(output_dict['clipwise_output'], y)
            _, predicted = torch.max(output_dict['clipwise_output'], 1)
        #AudioMAE
        elif model_type == "Transfer_AudioMAE":
            loss = cross_entropy_loss(output_dict, y)
            _, predicted = torch.max(output_dict, 1)
        
        train_acc = torch.sum(predicted==y)/y.shape[0]
        preds = predicted.detach().cpu().clone()
        y_true = y.detach().cpu().clone()
        f1_score = sklearn.metrics.f1_score(y_true, preds, average='macro', zero_division=0) # average='macro'
            
        if training == True:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()
                
        return loss.data.item(), train_acc, f1_score, preds, y_true
    
#function to write model test outputs to a csv file
## Utilizado apenas para os testes
def write_to_csv(model, csv_filepath, filtered_audios, files, device, transformer = False, masking=0.0):
    header = ['file_path', 'label']

    with open(csv_filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for index in range(len(files)):
            data_batch = []

            data_path = files[index]
            data_elem = filtered_audios[index]

            data_elem = torch.from_numpy(data_elem).float()

            data_batch.append(data_elem)

            data_batch = nn.utils.rnn.pad_sequence(data_batch, batch_first=True)
            data_batch = data_batch.to(device)
            
            output_dict = model.forward(data_batch, masking)
            if transformer:
                _, predicted = torch.max(output_dict, 1)
            else:
                _, predicted = torch.max(output_dict['clipwise_output'], 1)

            preds = predicted.detach().cpu().clone()

            pred_string = preds[0].item()
            data_row = [data_path, pred_string]

            writer.writerow(data_row)
            
    return True