"""
Treinamento do modelo de classificação (Random Forest) descrito na
seção 3.3 do TCC:

"Para o módulo de análise clínica, será utilizado um algoritmo de
classificação baseado em Random Forest [...] O desempenho do modelo será
avaliado por meio de métricas como acurácia, precisão, recall e F1-score
[...] Serão realizados testes de validação cruzada (cross-validation)
para garantir a robustez dos resultados obtidos."

Treina DOIS classificadores Random Forest a partir do mesmo conjunto de
variáveis de entrada:
  1) prioridade_model  -> nível de prioridade de atendimento (1 a 5)
  2) condicao_model    -> sugestão de possível condição clínica

Salva os modelos treinados e os encoders em model/*.pkl
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from generate_dataset import gerar_dataset, SINTOMAS, NIVEIS_CONSCIENCIA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def carregar_dados():
    caminho = os.path.join(BASE_DIR, "dataset_simulado.csv")
    if not os.path.exists(caminho):
        gerar_dataset(n=6000, caminho=caminho)
    return pd.read_csv(caminho)


def treinar():
    df = carregar_dados()

    # Codifica nível de consciência (variável categórica) em número
    consciencia_encoder = LabelEncoder()
    consciencia_encoder.fit(NIVEIS_CONSCIENCIA)
    df["consciencia_cod"] = consciencia_encoder.transform(df["consciencia"])

    features = ["temperatura", "freq_cardiaca", "pas", "pad", "spo2", "consciencia_cod"] + SINTOMAS
    X = df[features]

    resultados = {}

    # ---------- Modelo 1: Prioridade de atendimento ----------
    y_prioridade = df["prioridade"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_prioridade, test_size=0.2, random_state=42, stratify=y_prioridade
    )

    modelo_prioridade = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, class_weight="balanced"
    )
    modelo_prioridade.fit(X_train, y_train)

    y_pred = modelo_prioridade.predict(X_test)
    cv_scores = cross_val_score(modelo_prioridade, X, y_prioridade, cv=5)

    resultados["prioridade"] = {
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "cv_media": cv_scores.mean(),
        "cv_desvio": cv_scores.std(),
    }

    # ---------- Modelo 2: Condição clínica sugerida ----------
    condicao_encoder = LabelEncoder()
    y_condicao = condicao_encoder.fit_transform(df["condicao"])

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y_condicao, test_size=0.2, random_state=42, stratify=y_condicao
    )

    modelo_condicao = RandomForestClassifier(
        n_estimators=200, max_depth=14, random_state=42, class_weight="balanced"
    )
    modelo_condicao.fit(X_train2, y_train2)

    y_pred2 = modelo_condicao.predict(X_test2)
    cv_scores2 = cross_val_score(modelo_condicao, X, y_condicao, cv=5)

    resultados["condicao"] = {
        "acuracia": accuracy_score(y_test2, y_pred2),
        "precisao": precision_score(y_test2, y_pred2, average="weighted", zero_division=0),
        "recall": recall_score(y_test2, y_pred2, average="weighted", zero_division=0),
        "f1": f1_score(y_test2, y_pred2, average="weighted", zero_division=0),
        "cv_media": cv_scores2.mean(),
        "cv_desvio": cv_scores2.std(),
    }

    # ---------- Salvar artefatos ----------
    joblib.dump(modelo_prioridade, os.path.join(BASE_DIR, "modelo_prioridade.pkl"))
    joblib.dump(modelo_condicao, os.path.join(BASE_DIR, "modelo_condicao.pkl"))
    joblib.dump(consciencia_encoder, os.path.join(BASE_DIR, "encoder_consciencia.pkl"))
    joblib.dump(condicao_encoder, os.path.join(BASE_DIR, "encoder_condicao.pkl"))
    joblib.dump(features, os.path.join(BASE_DIR, "features.pkl"))

    print("\n=== Resultados do treinamento (dados simulados) ===")
    for nome, m in resultados.items():
        print(f"\nModelo: {nome}")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

    print("\nModelos salvos em:", BASE_DIR)
    return resultados


if __name__ == "__main__":
    treinar()
