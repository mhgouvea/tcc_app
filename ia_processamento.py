"""
Camada de processamento (IA) - seção 3.2 e 3.3 do TCC.

Carrega os modelos Random Forest treinados e aplica a classificação sobre
os dados inseridos pelo profissional de enfermagem durante a triagem,
retornando:
  - o nível de prioridade de atendimento sugerido (1 a 5, escala Manchester)
  - a possível condição clínica associada aos sintomas informados
  - o grau de confiança (probabilidade) de cada predição

O sistema é uma ferramenta de APOIO à decisão: a sugestão nunca substitui
o julgamento clínico do profissional de saúde, que mantém a autonomia e a
responsabilidade pela decisão final (ver seção 2.1 do TCC, BERNER, 2009).
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

SINTOMAS = [
    "febre",
    "dor_no_peito",
    "falta_de_ar",
    "tosse",
    "dor_cabeca_intensa",
    "tontura",
    "nausea_vomito",
    "dor_abdominal",
    "sangramento_ativo",
    "confusao_mental",
    "convulsao",
    "fraqueza_subita_um_lado",
    "dificuldade_falar",
    "dor_nas_costas",
    "erupcao_cutanea",
]

SINTOMAS_LABELS = {
    "febre": "Febre",
    "dor_no_peito": "Dor no peito",
    "falta_de_ar": "Falta de ar / dificuldade respiratória",
    "tosse": "Tosse",
    "dor_cabeca_intensa": "Dor de cabeça intensa",
    "tontura": "Tontura",
    "nausea_vomito": "Náusea / vômito",
    "dor_abdominal": "Dor abdominal",
    "sangramento_ativo": "Sangramento ativo",
    "confusao_mental": "Confusão mental",
    "convulsao": "Convulsão",
    "fraqueza_subita_um_lado": "Fraqueza súbita de um lado do corpo",
    "dificuldade_falar": "Dificuldade para falar",
    "dor_nas_costas": "Dor nas costas",
    "erupcao_cutanea": "Erupção cutânea",
}

PRIORIDADE_LABELS = {
    1: ("Emergência", "Vermelho", "Atendimento imediato"),
    2: ("Muito Urgente", "Laranja", "Até 10 minutos"),
    3: ("Urgente", "Amarelo", "Até 60 minutos"),
    4: ("Pouco Urgente", "Verde", "Até 120 minutos"),
    5: ("Não Urgente", "Azul", "Até 240 minutos"),
}

CONDICAO_LABELS = {
    "Possivel_Sindrome_Coronariana_Aguda": "Possível Síndrome Coronariana Aguda",
    "Possivel_AVC": "Possível Acidente Vascular Cerebral (AVC)",
    "Possivel_Sepse_Infeccao_Grave": "Possível Sepse / Infecção Grave",
    "Possivel_Insuficiencia_Respiratoria": "Possível Insuficiência Respiratória",
    "Possivel_Crise_Convulsiva": "Possível Crise Convulsiva",
    "Possivel_Trauma_Hemorragia": "Possível Trauma / Hemorragia",
    "Possivel_Quadro_Gastrointestinal": "Possível Quadro Gastrointestinal",
    "Possivel_Quadro_Infeccioso_Leve": "Possível Quadro Infeccioso Leve",
    "Quadro_Leve_Observacao": "Quadro Leve / Observação",
}


class MotorTriagem:
    """Encapsula os modelos treinados e expõe o método de classificação."""

    def __init__(self):
        self.modelo_prioridade = joblib.load(os.path.join(MODEL_DIR, "modelo_prioridade.pkl"))
        self.modelo_condicao = joblib.load(os.path.join(MODEL_DIR, "modelo_condicao.pkl"))
        self.encoder_consciencia = joblib.load(os.path.join(MODEL_DIR, "encoder_consciencia.pkl"))
        self.encoder_condicao = joblib.load(os.path.join(MODEL_DIR, "encoder_condicao.pkl"))
        self.features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))

    def classificar(self, vitais: dict, sintomas_selecionados: list):
        """
        vitais: dict com temperatura, freq_cardiaca, pas, pad, spo2, consciencia
        sintomas_selecionados: lista de chaves presentes em SINTOMAS
        """
        linha = {
            "temperatura": vitais["temperatura"],
            "freq_cardiaca": vitais["freq_cardiaca"],
            "pas": vitais["pas"],
            "pad": vitais["pad"],
            "spo2": vitais["spo2"],
            "consciencia_cod": self.encoder_consciencia.transform([vitais["consciencia"]])[0],
        }
        for s in SINTOMAS:
            linha[s] = 1 if s in sintomas_selecionados else 0

        X = pd.DataFrame([linha])[self.features]

        pred_prioridade = int(self.modelo_prioridade.predict(X)[0])
        proba_prioridade = self.modelo_prioridade.predict_proba(X)[0]
        confianca_prioridade = float(max(proba_prioridade))

        pred_condicao_idx = self.modelo_condicao.predict(X)[0]
        proba_condicao = self.modelo_condicao.predict_proba(X)[0]
        confianca_condicao = float(max(proba_condicao))
        condicao_key = self.encoder_condicao.inverse_transform([pred_condicao_idx])[0]

        label, cor, tempo = PRIORIDADE_LABELS[pred_prioridade]

        return {
            "prioridade": pred_prioridade,
            "prioridade_label": label,
            "prioridade_cor": cor,
            "prioridade_tempo": tempo,
            "confianca_prioridade": round(confianca_prioridade * 100, 1),
            "condicao_key": condicao_key,
            "condicao_label": CONDICAO_LABELS.get(condicao_key, condicao_key),
            "confianca_condicao": round(confianca_condicao * 100, 1),
        }


# Instância única (carregada uma vez quando o Flask sobe)
_motor = None


def get_motor():
    global _motor
    if _motor is None:
        _motor = MotorTriagem()
    return _motor
