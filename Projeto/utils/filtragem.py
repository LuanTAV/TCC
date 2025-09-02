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
    novos_arquivos = []
    cont = 0

    for filename in file_paths:
        sample_rate, audio, noise = wav2f0stats(filename,6)
        tipo = filename.split('/')[3]
        print(f"Duração do áudio completo: {len(audio) / sample_rate:.3f} s")
        print(f"Duração do ruído extraído: {len(noise) / sample_rate:.3f} s")

        y, eps = reduce_noise(audio_clip=audio, noise_clip=noise, 
                           n_grad_freq=n_grad_freq, n_grad_time=n_grad_time, n_std_thresh=n_std_thresh, prop_decrease=prop_decrease, verbose=False) # valores para tunar
        
        windowed_audios,_,_ = slice_windows(y, sample_rate, 4, 1, True, 0.0, False)
        for i in range(len(windowed_audios)):
            filtrados.append(windowed_audios[i])
            novos_arquivos.append(filename)
        #filtrados.append(y)
        cont+=1
        
        print(f"Filtrado ({cont}/{len(file_paths)}) SR: {sample_rate} Tipo: {tipo} Nome: {filename}")

        # base_name = os.path.basename(filename)
        # name_only = os.path.splitext(base_name)[0]

        # if(cont == 2):
        #     #sf.write(f"sinal_eps.wav", eps, samplerate=sample_rate)
        #     #sf.write(f"sinal_noise.wav", noise, samplerate=sample_rate)
        #     sf.write(f"sinal_filtrado_{name_only}.wav", windowed_audio[4], samplerate=sample_rate)
        #     sf.write(f"sinal_original_{name_only}.wav", y, samplerate=sample_rate)
        #     #plota_filtragem(audio, y, eps, sample_rate)

    return filtrados, novos_arquivos

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
    plt.savefig("waveform_plot_84.png")
    plt.close()

    import numpy as np

def slice_windows(y, sr, win_s, hop_s=None, pad_last=True, pad_value=0.0, keep_channels=False):
    """
    Corta o áudio em janelas de tamanho fixo.

    Parâmetros
    ----------
    y : np.ndarray
        Áudio mono [N] ou multi-canal [C, N] em float32/float64.
    sr : int
        Taxa de amostragem (Hz).
    win_s : float
        Duração da janela (segundos).
    hop_s : float | None
        Passo entre janelas (segundos). Se None, usa hop = win_s (janelas contíguas).
    pad_last : bool
        Se True, preenche a última janela com zeros (pad_value) quando não couber inteira.
        Se False, descarta o resto final.
    pad_value : float
        Valor usado no padding.
    keep_channels : bool
        Se True e y for [C, N], retorna janelas como [num, C, L].
        Se False, mistura canais por média e retorna [num, L].

    Retorna
    -------
    windows : np.ndarray
        Array de janelas: [num, L] (mono) ou [num, C, L] (multi-canal).
    starts : np.ndarray
        Amostras de início de cada janela (inteiros).
    times  : np.ndarray
        Tempos (segundos) correspondentes a cada início.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y_mono = y
        C = 1
    elif y.ndim == 2:
        C = y.shape[0]
        y_mono = y.mean(axis=0) if not keep_channels else y
    else:
        raise ValueError("y deve ser 1D (mono) ou 2D (C, N).")

    L = int(round(win_s * sr))
    H = int(round((hop_s if hop_s is not None else win_s) * sr))
    if L <= 0 or H <= 0:
        raise ValueError("win_s e hop_s devem ser > 0.")

    N = y.shape[-1]
    starts = list(range(0, max(N - L + 1, 0), H))
    if pad_last and (len(starts) == 0 or starts[-1] + L < N):
        starts.append(max(0, N - (N % H)))

    wins = []
    for s in starts:
        e = s + L
        if keep_channels and C > 1:
            if e <= N:
                seg = y[:, s:e]
            else:
                seg = np.full((C, L), pad_value, dtype=y.dtype)
                seg[:, :max(N - s, 0)] = y[:, s:N]
        else:
            src = y_mono
            if e <= N:
                seg = src[s:e]
            else:
                seg = np.full(L, pad_value, dtype=src.dtype)
                seg[:max(N - s, 0)] = src[s:N]
        wins.append(seg)

    windows = np.stack(wins) if wins else (np.empty((0, C, L)) if (keep_channels and C>1) else np.empty((0, L)))
    starts = np.array(starts, dtype=int)
    times  = starts / float(sr)
    return windows, starts, times

from collections import defaultdict
def verify_windows_labels(filtrados, novos_arquivos, label_dict, sr=None, print_n=5):
    # 0) tamanhos
    assert len(filtrados) == len(novos_arquivos), \
        f"Desalinhado: {len(filtrados)} janelas vs {len(novos_arquivos)} paths"

    # 1) todos os paths existem no dicionário de rótulos (ou no pai se usar sufixo '#seg')
    missing = []
    for p in set(novos_arquivos):
        if p in label_dict:
            continue
        if "#seg" in p and p.split("#")[0] in label_dict:
            continue
        missing.append(p)
    assert not missing, f"Paths sem rótulo no label_dict: {missing[:5]} ... total={len(missing)}"

    # 2) se usar '#seg', garanta que todas as janelas de um mesmo pai têm o MESMO rótulo
    groups = defaultdict(list)
    for i, p in enumerate(novos_arquivos):
        parent = p.split("#")[0] if "#seg" in p else p
        groups[parent].append(i)

    for parent, idxs in groups.items():
        # todos com mesmo label do pai
        lbls = []
        for i in idxs:
            p = novos_arquivos[i]
            lbl = label_dict.get(p, label_dict.get(parent))
            lbls.append(lbl)
        assert len(set(lbls)) == 1, f"Labels divergentes para {parent}: {set(lbls)}"

    # 3) amostra de inspeção
    print("— Amostra de (path -> label -> len):")
    for i in range(min(print_n, len(novos_arquivos))):
        p = novos_arquivos[i]
        parent = p.split("#")[0] if "#seg" in p else p
        lbl = label_dict.get(p, label_dict[parent])
        dur_s = (len(filtrados[i]) / sr) if (sr is not None) else None
        print(f"[{i}] {p}  →  y={lbl}  len={len(filtrados[i])}{' ('+str(round(dur_s,3))+'s)' if dur_s else ''}")

    print("✔ Verificação básica OK! (tamanhos, presença no dicionário e consistência por pai)")
