import sys
import numpy as np
import time
import sklearn
import csv
import torch
import torchaudio
import torch.nn as nn
 
sys.path.append("../../PANN/audioset_tagging_cnn/pytorch/")
from models import *

# Modificado de 
# https://github.com/marcelomatheusgauy/Pretrained_audio_neural_networks_emotion_recognition/blob/main/Pretrained_audio_neural_networks/train_utils.py

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

    def forward(self, input):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, None)
        embedding = output_dict['embedding']
        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1) # talvez mudar para sigmoid para classificaçao binaria (classes_num = 2)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict
    
def process_batches(filtered_audios, files, audio_target_dictionary, batch_size, file_index, device):

    new_sample_rate = 32000
    sample_rate = 22050 # sr apos filtragem
    data_batch = []
    
    audio_target_list = []
    #print(data_batch, batch_size, file_index, len(filtered_audios))

    while len(data_batch) < batch_size and file_index < len(filtered_audios):

        data_path = files[file_index]
        data_elem = filtered_audios[file_index]

        data_elem = torch.from_numpy(data_elem).float()

        data_elem = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(data_elem)
        sample_rate = new_sample_rate
        
        data_batch.append(data_elem)
        
        #if audio_target_dictionary[data_path] == 0:
        #    audio_target_list.append(0)
        #elif audio_target_dictionary[data_path] == 1:
        #    audio_target_list.append(1)
        #else: #this should not happen
        #    audio_target_list.append(2)

        audio_target_list.append(audio_target_dictionary[data_path])
        
        file_index +=1
        
    #convert list to torch tensor (pads different audio lengths to same size)
    data_batch = nn.utils.rnn.pad_sequence(data_batch, batch_first=True)
    
    data_batch = data_batch.to(device)
    
    audio_target_list = torch.LongTensor(audio_target_list)
    audio_target_list = audio_target_list.to(device)
    
    return data_batch, audio_target_list, file_index

#function to train model
def run_epoch(model, loss_compute, filtered_audios, files, audio_target_dictionary, device, training=True, batch_size=16):
    "Standard Training and Logging Function"
    train_acc_avg = 0
    f1_score_avg = 0
    
    number_elements = len(files)
    
    outputs=[]
    targets=[]
    
    file_index = 0
    step_index = 0
    while file_index < number_elements:
        step_index +=1
        
        data_batch, audio_target_list, file_index = process_batches(filtered_audios, files, audio_target_dictionary, batch_size, file_index, device)
        
        output_dict = model.forward(data_batch)

        _, train_acc, f1_score, output, target = loss_compute(output_dict, audio_target_list, training)
        
        outputs.append(output)
        targets.append(target)

        train_acc_avg = (train_acc_avg*(step_index-1)+train_acc)/(step_index)
        f1_score_avg = (f1_score_avg*(step_index-1)+f1_score)/(step_index)
        
        if step_index % 5 == 1:
            print("Epoch Step: %d  Train_acc: %f F1_score: %f" %
                    (step_index, train_acc_avg, f1_score_avg))

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    true_f1_score = sklearn.metrics.f1_score(targets, outputs, labels=[0,1], average='macro')
    print('Final F1_score=', true_f1_score)
    
    return train_acc_avg, f1_score_avg, true_f1_score


class LossCompute:
    "A simple loss compute and train function."
    def __init__(self, model, opt=None):
        self.model = model
        self.opt = opt
        
    def __call__(self, output_dict, y, training):
        train_acc = 0
        f1_score=0

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(output_dict['clipwise_output'], y)
        _, predicted = torch.max(output_dict['clipwise_output'], 1)
        train_acc = torch.sum(predicted==y)/y.shape[0]
        preds = predicted.detach().cpu().clone()
        y_true = y.detach().cpu().clone()
        f1_score = sklearn.metrics.f1_score(y_true, preds, labels=[0,1], average='macro')
            
        if training == True:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()
        return loss.data.item(), train_acc, f1_score, preds, y_true
    
#function to write model test outputs to a csv file
## Utilizado apenas para os testes
def write_to_csv(model, csv_filepath, filtered_audios, files, device, filtrado = True):
    header = ['file_path', 'label']

    with open(csv_filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        new_sample_rate = 32000
        sample_rate = 22050

        for index in range(len(files)):
            data_batch = []

            data_path = files[index]
            data_elem = filtered_audios[index]
            #print("Elem: ", data_elem)

            if(filtrado):
                data_elem = torch.from_numpy(data_elem).float()
                data_elem = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(data_elem)
                sample_rate = new_sample_rate

            data_batch.append(data_elem)

            data_batch = nn.utils.rnn.pad_sequence(data_batch, batch_first=True)
            data_batch = data_batch.to(device)
            
            output_dict = model.forward(data_batch)

            _, predicted = torch.max(output_dict['clipwise_output'], 1)

            preds = predicted.detach().cpu().clone()

            pred_string = preds[0].item()
            data_row = [data_path, pred_string]
            #print(data_row)
            writer.writerow(data_row)
            
    return True