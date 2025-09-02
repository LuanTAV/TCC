import sys
import numpy as np
import time
import sklearn
import csv
import torch
import torchaudio
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

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
        logits = self.fc_transfer(embedding)  # logits crus
        #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1) # talvez mudar para sigmoid para classificaçao binaria (classes_num = 2)
        #output_dict['clipwise_output'] = clipwise_output
        output_dict['clipwise_output'] = logits # na verdade sao logits e nao probabilidades aqui
 
        return output_dict
    
def process_batches(filtered_audios, files, audio_target_dictionary, batch_size, file_index, device):

    new_sample_rate = 32000
    sample_rate = 32000 #22050 # sr apos filtragem
    data_batch = []
    
    audio_target_list = []
    #print(data_batch, batch_size, file_index, len(filtered_audios))

    while len(data_batch) < batch_size and file_index < len(filtered_audios):

        data_path = files[file_index]
        data_elem = filtered_audios[file_index]
        #print(f"AQUI: {data_path}, {data_elem}")
        data_elem = torch.from_numpy(data_elem).float()

        # data_elem = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(data_elem)
        # sample_rate = new_sample_rate
        
        data_batch.append(data_elem)

        audio_target_list.append(audio_target_dictionary[data_path])
        
        file_index +=1
        
    #convert list to torch tensor (pads different audio lengths to same size)
    data_batch = nn.utils.rnn.pad_sequence(data_batch, batch_first=True)
    
    data_batch = data_batch.to(device)
    
    audio_target_list = torch.LongTensor(audio_target_list)
    audio_target_list = audio_target_list.to(device)
    
    return data_batch, audio_target_list, file_index

#function to train model
def run_epoch(epoch, model, loss_compute, filtered_audios, files, audio_target_dictionary, device, training=True, batch_size=16):
    "Standard Training and Logging Function"
    train_acc_avg = 0.0
    f1_score_avg = 0.0
    
    number_elements = len(files)
    
    outputs=[] # preds por batch
    targets=[] # y_true por batch
    
    file_index = 0
    step_index = 0
    while file_index < number_elements:
        step_index +=1
        
        data_batch, audio_target_list, file_index = process_batches(filtered_audios, files, audio_target_dictionary, batch_size, file_index, device)
        
        output_dict = model.forward(data_batch)

        _, train_acc, f1_score_step, output, target = loss_compute(output_dict, audio_target_list, training)
        
        outputs.append(output)
        targets.append(target)

        train_acc_avg = (train_acc_avg*(step_index-1)+train_acc)/(step_index)
        f1_score_avg = (f1_score_avg*(step_index-1)+f1_score_step)/(step_index)
        
        #if step_index % 5 == 1:
        print(f"[Epoch {epoch+1}] Step {step_index}/{(len(files)+batch_size-1)//batch_size} "
            f"| Batch size: {data_batch.size(0)} "
            f"| Train Acc: {train_acc_avg:.4f} "
            f"| F1: {f1_score_avg:.4f}")

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    true_f1_score = sklearn.metrics.f1_score(targets, outputs, average='macro', zero_division=0)
    print('Final F1_score=', true_f1_score)

    if training == False:
        # print("Matriz de confusão:\n",
        #       confusion_matrix(targets, outputs))
        # perm = np.random.permutation(len(targets))
        # f1_shuffle = f1_score(targets[perm], outputs, zero_division=0)
        # print("F1_macro com y_true embaralhado (deve cair MUITO):", f1_shuffle)
        print("VAL: shape preds/targets:", len(outputs), len(targets))
        print(confusion_matrix(targets, outputs))
        print(classification_report(targets, outputs, digits=4))
    
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
        f1_score = sklearn.metrics.f1_score(y_true, preds, average='macro', zero_division=0) # average='macro'
            
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

            data_elem = torch.from_numpy(data_elem).float()

            #if(filtrado):
                #data_elem = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(data_elem)
                #sample_rate = new_sample_rate

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


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        #return self.factor * \ (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return 0.0001
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))