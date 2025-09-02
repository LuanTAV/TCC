import pandas as pd
import random
import torchaudio
import librosa
import numpy as np

folder = '../SPIRA/SPIRA_Dataset_V2/'

def Load_Train_dataset(file_label):
    
    train_csv = '../SPIRA/SPIRA_Dataset_V2/metadata_train.csv'
    
    df = pd.read_csv(train_csv)

    data_paths_train = []

    for row in range(len(df)):
        file = df['file_path'][row]
        label = df['class'][row]
        file_path = folder+file
        data_path = file_path
        data_paths_train.append(data_path)
        
        file_label[file_path] = label

    random.shuffle(data_paths_train)

    return data_paths_train 

def Load_Eval_dataset(file_label):

    eval_csv = '../SPIRA/SPIRA_Dataset_V2/metadata_eval.csv'

    df = pd.read_csv(eval_csv)

    data_paths_eval = []

    for row in range(len(df)):
        file = df['file_path'][row]
        label = df['class'][row]
        file_path = folder+file
        data_path = file_path
        data_paths_eval.append(data_path)
        
        file_label[file_path] = label

    random.shuffle(data_paths_eval)

    return data_paths_eval 
    

def Load_Test_dataset(file_label):


    test_csv = '../SPIRA/SPIRA_Dataset_V2/metadata_test.csv'

    df = pd.read_csv(test_csv)

    data_paths_test = []

    for row in range(len(df)):
        file = df['file_path'][row]
        label = df['class'][row]
        file_path = folder+file
        data_path = file_path
        data_paths_test.append(data_path)
        
        file_label[file_path] = label

    #random.shuffle(data_paths_test)

    return data_paths_test

def Load_Normal_audios(file_paths):

    normal_audios = []
    new_sample_rate = 32000

    for filename in file_paths:
        #sample_rate = torchaudio.info(filename).sample_rate
        #data_elem, sample_rate = load_mono_32k_ta(filename)
        #resample para 32kHz
        #data_elem = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(data_elem)
        #sample_rate = new_sample_rate
        #data_elem = data_elem[0]
        data_elem, sample_rate = librosa.core.load(filename, sr=32000, mono=True)
        normal_audios.append(data_elem)

    return normal_audios

def load_mono_32k_ta(path):
    wav, sr = torchaudio.load(path)            # [C, T], float32 em [-1,1]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)    # mono ANTES do resample
    if sr != 32000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)(wav)
        sr = 32000                             # <-- atualiza o sr!
    y = wav.squeeze(0).numpy().astype(np.float32)
    return y, sr


