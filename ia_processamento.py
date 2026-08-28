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


PRESSAO_CATEGORIAS = [
    # (nome, nivel de severidade 0-5, cor, faixa mmHg exibida, recomendação)
    {
        "chave": "hipotensao",
        "nome": "Hipotensão",
        "nivel": 2,
        "cor": "#2b7de9",
        "faixa": "PAS < 90 ou PAD < 60",
        "recomendacao": "Avaliar sinais de hipoperfusão (tontura, palidez, sudorese). Investigar causa.",
    },
    {
        "chave": "normal",
        "nome": "Normal",
        "nivel": 0,
        "cor": "#2f9e44",
        "faixa": "PAS < 120 e PAD < 80",
        "recomendacao": "Manter hábitos saudáveis. Reavaliação de rotina.",
    },
    {
        "chave": "elevada",
        "nome": "Elevada",
        "nivel": 1,
        "cor": "#e6b800",
        "faixa": "PAS 120–129 e PAD < 80",
        "recomendacao": "Reforçar orientações sobre dieta, atividade física e reduzir o sal. Reavaliar em breve.",
    },
    {
        "chave": "hipertensao_1",
        "nome": "Hipertensão Estágio 1",
        "nivel": 3,
        "cor": "#f4772e",
        "faixa": "PAS 130–139 ou PAD 80–89",
        "recomendacao": "Encaminhar para avaliação médica. Considerar mudança de estilo de vida e acompanhamento.",
    },
    {
        "chave": "hipertensao_2",
        "nome": "Hipertensão Estágio 2",
        "nivel": 4,
        "cor": "#d7263d",
        "faixa": "PAS ≥ 140 ou PAD ≥ 90",
        "recomendacao": "Encaminhar para avaliação médica prioritária. Provável necessidade de tratamento medicamentoso.",
    },
    {
        "chave": "crise_hipertensiva",
        "nome": "Crise Hipertensiva",
        "nivel": 5,
        "cor": "#7f0f20",
        "faixa": "PAS > 180 e/ou PAD > 120",
        "recomendacao": "URGÊNCIA: encaminhar imediatamente para atendimento médico. Verificar sinais de lesão de órgão-alvo.",
    },
]


def classificar_pressao_arterial(pas: int, pad: int) -> dict:
    """
    Classifica a pressão arterial a partir dos valores de PAS (sistólica) e
    PAD (diastólica), com base nas faixas de referência amplamente adotadas
    por diretrizes de cardiologia (ex.: American Heart Association / SBC).

    Esta é uma classificação por REGRAS CLÍNICAS (determinística), não um
    modelo de machine learning - a categorização de pressão arterial segue
    limiares médicos bem estabelecidos, então regras são o método correto
    e mais transparente para essa tarefa (diferente da priorização de
    triagem, que usa o modelo Random Forest treinado).

    Retorna um dicionário com a categoria, nível de severidade (0-5),
    cor de referência, faixa de valores e uma recomendação textual.

    IMPORTANTE: aviso educacional apenas - não substitui avaliação médica.
    """
    pas = int(pas)
    pad = int(pad)

    if pas > 180 or pad > 120:
        cat = PRESSAO_CATEGORIAS[5]
    elif pas >= 140 or pad >= 90:
        cat = PRESSAO_CATEGORIAS[4]
    elif pas >= 130 or pad >= 80:
        cat = PRESSAO_CATEGORIAS[3]
    elif pas >= 120 and pad < 80:
        cat = PRESSAO_CATEGORIAS[2]
    elif pas < 90 or pad < 60:
        cat = PRESSAO_CATEGORIAS[0]
    else:
        cat = PRESSAO_CATEGORIAS[1]

    is_hipertenso = cat["chave"] in ("hipertensao_1", "hipertensao_2", "crise_hipertensiva")

    # Posição (0-100%) numa régua visual de 60 a 200 mmHg de PAS, para a barra de gauge
    gauge_pct = max(0, min(100, round((pas - 60) / (200 - 60) * 100)))

    return {
        "categoria": cat["nome"],
        "categoria_chave": cat["chave"],
        "nivel": cat["nivel"],
        "cor": cat["cor"],
        "faixa_referencia": cat["faixa"],
        "recomendacao": cat["recomendacao"],
        "is_hipertenso": is_hipertenso,
        "gauge_pct": gauge_pct,
        "pas": pas,
        "pad": pad,
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
