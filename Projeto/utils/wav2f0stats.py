# Codigo modificado de:
#https://github.com/SPIRA-COVID19/noise-reduce-tool/blob/master/common/wav2f0stats.py

#!/usr/bin/python3
# coding: utf-8
# # Projeto SPIRA
# ## Pós-processamento dos sinais de áudio
# ### Segmentação de trechos de elocução e estatísticas relacionadas a F0
# #### Marcelo Queiroz - Reunião do projeto SPIRA em 08/10/2020
#

import sys
import os
import math as m
import numpy as np
from scipy import stats
import scipy.io.wavfile as wavfile
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


def wav2f0stats(file_path, threshold = 6):
    NOISE_THRESHOLD = float(threshold)

    def opusfile_read(filename):
        x, rate = librosa.load(filename)
        return rate, x
    
    # abre sinal gravado em arquivo
    rate, x = opusfile_read(file_path)

    #print(str(sys.argv[1]),end=',')

    # pré-processamentos: elimina dc e ajusta faixa de amplitudes a [-1,+1]
    # (obs: isso não é uma normalização, mas é necessário para permitir
    #       o uso da opção normalize=False no widget ipd.Audio)

    sample_depth = 8*x[0].itemsize # número de bits por amostra
    x = x[:]-np.mean(x) # elimina dc
    #x = x/(2**(sample_depth-1)) # ajusta amplitudes para [-1,+1]

    #plt.plot(np.arange(len(x))/rate,x);plt.title(f"Arquivo {file_path} (sinal original)")
    #plt.xlabel("tempo");plt.ylabel("amplitude");plt.ylim([-1,1]);
    #plt.savefig("waveform_plot.png");plt.close()

    # calcula a energia média (em dB) do sinal sobre janelas deslizantes
    # devolve sinal edB e seu valor mínimo (noise floor)
    def window_pow(sig, window_size=4096):
        sig2 = np.power(sig,2)
        window = np.ones(window_size)/float(window_size)
        edB = 10*np.log10(np.convolve(sig2, window))[window_size-1:]
        # corrige aberrações nas extremidades (trechos artificialmente silenciosos)
        imin = int(0.5*rate) # isola 500ms iniciais e finais 
        edBmin = min(edB[imin:-imin])
        edB = np.maximum(edB,edBmin)
        return edB, edBmin

    # calcula envoltória de energia em dB do sinal
    edB, edBmin = window_pow(x)

    # fig, ax1 = plt.subplots()
    # fig1 = ax1.plot(np.arange(len(x))/rate, x, label="Sinal")
    # ax1.set_ylim([-1,1])
    # ax1.set_xlabel(r"tempo")
    # ax1.set_ylabel(r"amplitude")
    # ax2 = ax1.twinx()
    # fig2 = ax2.plot(np.arange(len(edB))/rate,edB,"tab:orange",label="Energia (dB)")
    # m = min(edB);M=max(edB)-m
    # ax2.set_ylim([m-1.1*M,m+1.1*M])
    # ax2.set_ylabel(r"energia (dB)")
    # figs = fig1+fig2;labels = [l.get_label() for l in figs]
    # plt.legend(figs,labels,loc="lower right")
    #plt.savefig("waveform_plot2.png");plt.close()

    # Filtro da mediana 1D para vetores booleanos, sobre janelas de 2*N+1 elementos
    def boolean_majority_filter(sig_in,N):
        # cria uma cópia do vetor de entrada
        sig_out = sig_in.copy()
        # coloca N valores True de sentinela antes e depois do vetor sig_in
        # (True força maior chance das bordas serem consideradas ruído, o que é + comum)
        sig_pad = np.concatenate((np.ones(N),sig_in,np.ones(N+1)))
        # contadores de "votos"
        nTrue = 0
        nFalse = 0
        # inicialização da contagem (corresponderá à contagem da situação em sig[0])
        for i in range(2*N+1):
            if sig_pad[i]:
                nTrue += 1
            else:
                nFalse += 1
        # aplica filtro da maioria nos índices de sig_in/out: a cada índice i, o resultado
        # é o voto da maioria na janela sig_in[i-N:i+N+1] (contendo 2*N+1 elementos)
        # que corresponde aos índices sig_pad[i:i+2*N+1] no vetor com sentinelas
        for i in range(len(sig_out)):
            sig_out[i] = nTrue>nFalse
            # subtrai o voto retirado (primeiro da janela deslizante atual)
            # se possível, tira um voto do sinal já *filtrado*
            # (aproveita estabilidade à esquerda)
            if i>=N:
                nout = sig_out[i-N]
            else: # se não for possível, tira um "True" do vetor com sentinelas
                nout = sig_pad[i]
            if nout:
                nTrue -=1
            else:
                nFalse -= 1
            # inclui o voto novo, que é o último da janela deslizante
            # referente ao próximo índice (i+1) em sig
            if sig_pad[i+1+2*N]:
                nTrue += 1
            else:
                nFalse += 1
        return sig_out


    # seleção de trechos do sinal sig contendo ruído, devolve sinal booleano
    def noise_sel(sig,edB,edBmin,noise_threshold=NOISE_THRESHOLD):
        # seleciona frames com rms próxima do nível mínimo
        inoise_pre = edB<edBmin+noise_threshold
        # aplica filtro da mediana (voto de maioria) para eliminar
        # trechos menores do que 0.2s
        inoise = boolean_majority_filter(inoise_pre,int(0.2*rate))
        return inoise, inoise_pre    


    # aplica seleção de frames ao sinal de entrada
    xnoise, xnoisep = noise_sel(x,edB,edBmin)


    # recorta trechos do sinal identificados como ruído
    xx = x.copy()
    n = 1
    while n<len(x)-1:
        if (xnoise[n] and not xnoise[n-1]) or (xnoise[n-1] and not xnoise[n]):
            for i in range(-100,101):
                if (n+i) in range(len(xx)): xx[n+i] = xx[n+i]*abs(i)/100
            n = n+99
        n = n+1
    noise = x[xnoise]
    #plt.plot(np.arange(len(noise))/rate,noise)
    #plt.title("Trechos de ruído do sinal")
    #plt.xlabel("tempo");plt.ylabel("amplitude");plt.ylim([-1,1]);
    #plt.savefig("waveform_plot4.png");plt.close()
    
    # recorta trechos de áudio do sinal identificados como locução
    xloc = np.logical_not(xnoise)
    loc = xx[xloc]

    # ## Extração de F0 com YIN

    # The standard range is 75–600 Hertz (https://www.fon.hum.uva.nl/praat/manual/Voice.html)
    #f0 = librosa.yin(loc,fmin=75,fmax=600,sr=rate)
    #plt.plot(f0);plt.title("Curva de F0 instantânea");plt.show()
    # Small data decidiu em 5/11/2020 usar 50-600
    (f0, pf0, ppf0) = librosa.pyin(loc,sr=rate,fmin=50,fmax=600)


    f0final=f0[~np.isnan(f0)]

    # print(np.median(f0final),end=',')
    # print(np.mean(f0final),end=',')
    # print(np.std(f0final),end=',')
    # print(np.min(f0final),end=',')
    # print(np.max(f0final))
    
    

    #fig = plt.plot(np.arange(len(xnoise))/rate,xnoisep,"c:")
    #plt.plot(np.arange(len(xnoise))/rate,xnoise)
    #plt.xlabel("tempo");plt.yticks([],"")
    #plt.ylabel(r"não é ruído $\longleftarrow\quad\quad\quad\quad\quad\quad\longrightarrow$ é ruído")
    #plt.title("Seleção de ruído e filtro da maioria");
    #plt.savefig("waveform_plot3.png");plt.close()

    return rate, x, noise
    # return {
    #     'median': np.median(f0final),
    #     'mean': np.mean(f0final),
    #     'std': np.std(f0final),
    #     'min': np.min(f0final),
    #     'max': np.max(f0final),
    # }