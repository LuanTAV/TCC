import optuna
import subprocess
import re

def objective(trial):
    freq = trial.suggest_int("freq", 5, 10)
    time = trial.suggest_int("time", 5, 10)
    thresh = trial.suggest_float("thresh", 2.0, 3.0, step=0.1)
    propdec = trial.suggest_float("propdec", 0.1, 0.5, step=0.1)

    it = 0
    param_tag = "optuna"

    try:
        subprocess.run([
            "python3", "SpiraFilter.py",
            "--freq", str(freq),
            "--time", str(time),
            "--thresh", str(thresh),
            "--propdec", str(propdec),
            "--param", param_tag,
            "--it", str(it)
        ], check=True)

        result = subprocess.run([
            "python3", "SpiraTest.py",
            "--freq", str(freq),
            "--time", str(time),
            "--thresh", str(thresh),
            "--propdec", str(propdec),
            "--param", param_tag,
            "--it", str(it)
        ], capture_output=True, text=True, check=True)

        # Extrair métricas
        stdout = result.stdout
        f1_match = re.search(r"F1[-_ ]score:?\s*([0-9.]+)", stdout, re.IGNORECASE)
        acc_match = re.search(r"Acurácia:?\s*([0-9.]+)", stdout, re.IGNORECASE)

        f1 = float(f1_match.group(1)) if f1_match else 0.0
        acc = float(acc_match.group(1)) if acc_match else 0.0

        trial.set_user_attr("accuracy", acc)
        return f1  # Otimizar com base no F1-score

    except subprocess.CalledProcessError as e:
        print("Erro na execução dos scripts:", e)
        return 0.0


# Criação do estudo
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100) 

# Resultados
print("Melhor F1-score:", study.best_value)
print("Melhores parâmetros:", study.best_params)

# Salvar estudo
df = study.trials_dataframe(attrs=("params","user_attrs"))
df.to_csv("optuna_resultados3.csv", index=False)
print(df[["value", "params_freq", "params_time", "params_thresh", "params_propdec", "user_attrs_accuracy"]])

