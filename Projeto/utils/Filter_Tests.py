import subprocess

freq_list = [1, 2, 3, 4, 5, 6 ,7]
time_list = [1, 2, 3, 4, 5, 6, 7]
thresh_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4]
propdec_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

#params = ["thresh", "freqtime", "propdec"]
params = ["sensibilidade"]
train = True
cont = 0
qtd = 10 # numero de vezes para o calculo da media

# Testes variando thresh
if "thresh" in params:
    for i in thresh_list:
        print(f"Executando para thresh: {i}")
        for q in range(qtd):
            if train == True:
                subprocess.run([
                    "python3", "SpiraFilter.py",
                    "--thresh", str(i),
                    "--param", "thresh",
                    "--it", str(cont),
                ])

            subprocess.run([
                "python3", "SpiraTest.py",
                "--thresh", str(i),
                "--param", "thresh",
                "--it", str(cont),
            ])
        cont = cont + 1

cont =  0
# Testes variando janela frequency/time
if "freqtime" in params:
    for i in freq_list:
        for j in time_list:
            print(f"Executando para freq/time: {i}")
            for q in range(qtd):
                if train == True:
                    subprocess.run([
                        "python3", "SpiraFilter.py",
                        "--freq", str(i),
                        "--time", str(j),
                        "--param", "freqtime",
                        "--it", str(cont),
                    ])

                subprocess.run([
                    "python3", "SpiraTest.py",
                    "--freq", str(i),
                    "--time", str(j),
                    "--param", "freqtime",
                    "--it", str(cont),
                ])
            cont = cont + 1

cont =  0
# Testes variando prop_decrease
if "propdec" in params:
    for i in propdec_list:
        print(f"Executando para propdec: {i}")
        for q in range(qtd):
            if train == True:
                subprocess.run([
                    "python3", "SpiraFilter.py",
                    "--propdec", str(i),
                    "--param", "propdec",
                    "--it", str(cont),
                ])
            subprocess.run([
                "python3", "SpiraTest.py",
                "--propdec", str(i),
                "--param", "propdec",
                "--it", str(cont),
            ])
        cont = cont + 1

# uma das melhore configurações encontradas
if "melhor" in params:
    print(f"Executando para Melhor")
    for q in range(qtd):
        if train == True:
            subprocess.run([
                "python3", "SpiraFilter.py",
                "--freq", str(8),
                "--time", str(8),
                "--thresh", str(3.0),
                "--propdec", str(0.5),
                "--param", "Melhor",
                "--it", str(q),
            ])
        subprocess.run([
            "python3", "SpiraTest.py",
            "--freq", str(8),
            "--time", str(8),
            "--thresh", str(3.0),
            "--propdec", str(0.5),
            "--param", "Melhor",
            "--it", str(q),
        ])

# configuraçao base passada
if "primeiro" in params:
    print(f"Executando para Primeiro")
    for q in range(qtd):
        if train == True:
            subprocess.run([
                "python3", "SpiraFilter.py",
                "--freq", str(3),
                "--time", str(3),
                "--thresh", str(2.0),
                "--propdec", str(1.0),
                "--param", "Primeiro",
                "--it", str(q),
            ])
        subprocess.run([
            "python3", "SpiraTest.py",
            "--freq", str(3),
            "--time", str(3),
            "--thresh", str(2.0),
            "--propdec", str(1.0),
            "--param", "Primeiro",
            "--it", str(q),
        ])

qtd = 100
if "sensibilidade" in params:
    print(f"Executando para Sensibilidade")
    for q in range(qtd):
        freq = max(1, int(8 - (q * (8 / qtd))))          
        time = max(1, int(8 - (q * (8 / qtd))))           
        thresh = max(0.1, 3.0 - (q * (3.0 / qtd)))        
        propdec = max(0.1, 1 - (q * (1 / qtd)))       

        subprocess.run([
            "python3", "SpiraTest.py",
            "--freq", str(freq),
            "--time", str(time),
            "--thresh", str(round(thresh, 2)),
            "--propdec", str(round(propdec, 2)),
            "--param", "Sensibilidade",
            "--it", str(q),
        ])
