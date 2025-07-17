import numpy as np
import scipy.io.wavfile as wavfile
import sys
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import os

sys.path.append("../noise-reduce-tool/common/")
from noisereduce import reduce_noise

from wav2f0stats import wav2f0stats

def noise_reduction(file_paths, n_grad_freq=3, n_grad_time=3, n_std_thresh=2, prop_decrease=1.0):

    filtrados = []
    cont = 0

    for filename in file_paths:
        sample_rate, audio, noise = wav2f0stats(filename,6)
        #print(type(audio),len(audio))
        y, eps = reduce_noise(audio_clip=audio, noise_clip=noise, 
                              n_grad_freq=n_grad_freq, n_grad_time=n_grad_time, n_std_thresh=n_std_thresh, prop_decrease=prop_decrease, verbose=False) # valores para tunar
        #print(type(y),len(y))
        filtrados.append(y)
        cont+=1
        print(f"Filtrado ({cont}/{len(file_paths)}) SR: {sample_rate}")

        # base_name = os.path.basename(filename)
        # name_only = os.path.splitext(base_name)[0]

        # if(cont == 2):
        #     sf.write(f"sinal_original_{name_only}.wav", audio, samplerate=sample_rate)
        #     sf.write(f"sinal_filtrado_{name_only}.wav", y, samplerate=sample_rate)
        #     plota_filtragem(audio, y, eps, sample_rate)

    return filtrados

def plota_filtragem(audio, y, eps,sample_rate):

    sig = [audio, y, eps]
    sig_names = ["Sinal original", "Sinal filtrado", "Ruído extraído"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for i in range(3):
        t = np.arange(len(sig[i])) / sample_rate
        ax[i].plot(t, sig[i])
        ax[i].set_ylim([-1, 1])
        ax[i].set_title(sig_names[i])
        ax[i].set_xlabel("Tempo (s)")
        ax[i].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("waveform_plot7.png")
    plt.close()