import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# df = pd.read_csv('results_full_media_thresh.csv', header=None)

# df.columns = ['name', 'config_id', 'thresh', 'propdec', 'freq', 'time', 'acc', 'f1']

# group_size = 10
# num_groups = len(df) // group_size

# x_labels = []
# avg_accs = []
# avg_f1s = []

# for i in range(num_groups):
#     group = df.iloc[i * group_size : (i + 1) * group_size]
#     config_name = f"{group.iloc[0]['name']} {group.iloc[i]['thresh']}"
    
#     mean_acc = group['acc'].mean()
#     mean_f1 = group['f1'].mean()

#     x_labels.append(config_name)
#     avg_accs.append(mean_acc)
#     avg_f1s.append(mean_f1)

# # Plotar os pontos
# plt.figure(figsize=(10, 6))
# plt.plot(x_labels, avg_accs, marker='o', label='Accuracy (média)')
# plt.plot(x_labels, avg_f1s, marker='s', label='F1-score (média)')
# plt.xlabel('Configuração')
# plt.ylabel('Média (10 testes)')
# plt.title('Média de Accuracy e F1-score por configuração')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('params.png')

# ########################################################################################################################
# x_labels = []
# avg_accs = []
# avg_f1s = []

# df = pd.read_csv('results_full_media_propdec.csv', header=None)
# df.columns = ['name', 'config_id', 'thresh', 'propdec', 'freq', 'time', 'acc', 'f1']
# num_groups = len(df) // group_size

# for i in range(num_groups):
#     group = df.iloc[i * group_size : (i + 1) * group_size]
#     config_name = f"{group.iloc[0]['name']} {group.iloc[0]['propdec']}"
    
#     mean_acc = group['acc'].mean()
#     mean_f1 = group['f1'].mean()

#     x_labels.append(config_name)
#     avg_accs.append(mean_acc)
#     avg_f1s.append(mean_f1)

# # Plotar os pontos
# plt.figure(figsize=(10, 6))
# plt.plot(x_labels, avg_accs, marker='o', label='Accuracy (média)')
# plt.plot(x_labels, avg_f1s, marker='s', label='F1-score (média)')
# plt.xlabel('Configuração')
# plt.ylabel('Média (10 testes)')
# plt.title('Média de Accuracy e F1-score por configuração')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('params2.png')
# ########################################################################################################################

# x_labels = []
# avg_accs = []
# avg_f1s = []

# df = pd.read_csv('results_full_media_freqtime.csv', header=None)
# df.columns = ['name', 'config_id', 'thresh', 'propdec', 'freq', 'time', 'acc', 'f1']
# num_groups = len(df) // group_size

# for i in range(num_groups):
#     group = df.iloc[i * group_size : (i + 1) * group_size]
#     config_name = f"{group.iloc[0]['name']} {group.iloc[0]['freq']} {group.iloc[0]['time']}"
    
#     mean_acc = group['acc'].mean()
#     mean_f1 = group['f1'].mean()

#     x_labels.append(config_name)
#     avg_accs.append(mean_acc)
#     avg_f1s.append(mean_f1)

# # Plotar os pontos
# plt.figure(figsize=(15, 6))
# plt.plot(x_labels, avg_accs, marker='o', label='Accuracy (média)')
# plt.plot(x_labels, avg_f1s, marker='s', label='F1-score (média)')
# plt.xlabel('Configuração')
# plt.ylabel('Média (10 testes)')
# plt.title('Média de Accuracy e F1-score por configuração')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('params3.png')


###################################################################################################


df = pd.read_csv("testes/sensibilidade100.csv", header=None)
df.columns = ["Tipo", "Iteracao", "Thresh", "PropDecrease", "Freq", "Time", "Acuracia", "F1"]
df = df.sort_values(by="PropDecrease")
ticks_x = np.arange(0.0, 1.05, 0.1)
ticks_y = np.arange(0.0, 1.05, 0.1)

plt.style.use("ggplot")  

plt.figure(figsize=(12, 6))
plt.plot(df["PropDecrease"], df["Acuracia"], label="Acurácia", marker="o", linewidth=2)
plt.plot(df["PropDecrease"], df["F1"], label="F1-Score", marker="*", linewidth=2)

plt.title("Impacto da Supressão de Ruído na Performance do Modelo", fontsize=14, weight='bold')
plt.xlabel("Intensidade de supressão (prop_decrease)", fontsize=12)
plt.ylabel("Métrica", fontsize=12)
plt.xticks(ticks_x, [f"{x:.1f}" for x in ticks_x], fontsize=10)
plt.yticks(ticks_x, [f"{x:.1f}" for x in ticks_x], fontsize=10)
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("testes/resultsSensibilidade.png", dpi=300)
plt.close()
