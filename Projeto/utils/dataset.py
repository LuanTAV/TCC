import pandas as pd
import random
import torchaudio
import librosa
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

folder = '../SPIRA/SPIRA_Dataset_V2/'
folder_noise = '../SPIRA/SPIRA_Dataset_V2/com_ruido/'

def Load_Train_dataset(file_label, noise=False):
    
    train_csv = '../SPIRA/SPIRA_Dataset_V2/metadata_train.csv'
    
    df = pd.read_csv(train_csv)

    data_paths_train = []

    for row in range(len(df)):
        file = df['file_path'][row]
        label = df['class'][row]
        file_path = folder+file
        if(noise):
            if(label == 1):
                file = file[:10] + "pacientes_" + file[10:]
            else:
                file = file[:9] + "controle_" + file[9:]
            file_path = folder_noise + file
        data_path = file_path
        data_paths_train.append(data_path)
        
        file_label[file_path] = label

    random.shuffle(data_paths_train)

    return data_paths_train 

def Load_Eval_dataset(file_label, noise=False):

    eval_csv = '../SPIRA/SPIRA_Dataset_V2/metadata_eval.csv'

    df = pd.read_csv(eval_csv)

    data_paths_eval = []

    for row in range(len(df)):
        file = df['file_path'][row]
        label = df['class'][row]
        file_path = folder+file
        if(noise):
            if(label == 1):
                file = file[:10] + "pacientes_" + file[10:]
            else:
                file = file[:9] + "controle_" + file[9:]
            file_path = folder_noise + file
        data_path = file_path
        data_paths_eval.append(data_path)
        
        file_label[file_path] = label

    random.shuffle(data_paths_eval)

    return data_paths_eval 
    

def Load_Test_dataset(file_label, noise = False):


    test_csv = '../SPIRA/SPIRA_Dataset_V2/metadata_test.csv'

    df = pd.read_csv(test_csv)

    data_paths_test = []

    for row in range(len(df)):
        file = df['file_path'][row]
        label = df['class'][row]
        file_path = folder + file

        if(noise):
            if(label == 1):
                file = file[:10] + "pacientes_" + file[10:]
            else:
                file = file[:9] + "controle_" + file[9:]
            file_path = folder_noise + file
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
        print(f"Carregado:{filename}")

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

def majority_vote_test(preds):
    return Counter(preds).most_common(1)[0][0]

def majority_vote_eval(outputs, targets, file_refs, audio_target_dictionary):
    """
    Aplica voto de maioria por arquivo a partir das predições de janelas.
    """
    # cria DataFrame com todas as infos
    df = pd.DataFrame({
        "file": file_refs,
        "y_true": targets,
        "y_pred": outputs
    })

    # voto de maioria por arquivo
    majority_preds = (
        df.groupby("file")["y_pred"]
          .agg(lambda x: x.value_counts().idxmax())   # pega classe mais frequente
    )

    # rótulos verdadeiros por arquivo (via dicionário de labels)
    y_true_file = [audio_target_dictionary[f] for f in majority_preds.index]
    y_pred_file = majority_preds.values

    # métricas por arquivo
    acc_file = accuracy_score(y_true_file, y_pred_file)
    f1_file = f1_score(y_true_file, y_pred_file, average="macro", zero_division=0)

    print("===== Avaliação por ARQUIVO (voto de maioria) =====")
    print("Accuracy:", acc_file)
    print("F1 Score:", f1_file)
    print("Matriz de confusão:")
    print(confusion_matrix(y_true_file, y_pred_file))
    print("Relatório de classificação:")
    print(classification_report(y_true_file, y_pred_file, digits=4))

    return acc_file, f1_file, y_true_file, y_pred_file
