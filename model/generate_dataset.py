"""
Geração de dados clínicos SIMULADOS para treinamento do modelo de IA.

Conforme descrito no TCC (seção 3.3 - Modelo de Inteligência Artificial):
"O modelo será treinado com um conjunto de dados simulados, contendo
combinações de sintomas e seus respectivos diagnósticos, baseados nos
protocolos de triagem utilizados em serviços de urgência e emergência."

IMPORTANTE: Estes dados são inteiramente sintéticos/fictícios, gerados por
regras que aproximam o raciocínio clínico do Protocolo de Manchester
(MACKWAY-JONES; MARSDEN; WINDLE, 2017). Não devem ser usados para decisões
clínicas reais. Servem apenas para demonstrar o funcionamento do protótipo
acadêmico.
"""

import random
import csv
import os

random.seed(42)

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

NIVEIS_CONSCIENCIA = ["Alerta", "Resposta_a_voz", "Resposta_a_dor", "Nao_responsivo"]

PRIORIDADES = {
    1: "Emergencia",       # Vermelho - atendimento imediato
    2: "Muito_Urgente",    # Laranja - até 10 min
    3: "Urgente",          # Amarelo - até 60 min
    4: "Pouco_Urgente",    # Verde - até 120 min
    5: "Nao_Urgente",      # Azul - até 240 min
}

CONDICOES = [
    "Possivel_Sindrome_Coronariana_Aguda",
    "Possivel_AVC",
    "Possivel_Sepse_Infeccao_Grave",
    "Possivel_Insuficiencia_Respiratoria",
    "Possivel_Crise_Convulsiva",
    "Possivel_Trauma_Hemorragia",
    "Possivel_Quadro_Gastrointestinal",
    "Possivel_Quadro_Infeccioso_Leve",
    "Quadro_Leve_Observacao",
]


def gerar_caso():
    """Gera um caso clínico simulado com regras que aproximam gravidade."""
    sintomas_ativos = set()
    n_sintomas = random.choices([1, 2, 3, 4], weights=[0.35, 0.35, 0.2, 0.1])[0]
    sintomas_ativos.update(random.sample(SINTOMAS, n_sintomas))

    # Vitais base "normais" com variação
    temperatura = round(random.gauss(36.6, 0.6), 1)
    freq_cardiaca = int(random.gauss(80, 12))
    pas = int(random.gauss(120, 15))   # pressão sistólica
    pad = int(random.gauss(78, 10))    # pressão diastólica
    spo2 = int(random.gauss(97, 2))
    consciencia = "Alerta"

    gravidade_score = 0

    if "febre" in sintomas_ativos:
        temperatura = round(random.uniform(37.8, 40.5), 1)
        gravidade_score += 1 if temperatura < 39 else 2

    if "dor_no_peito" in sintomas_ativos:
        freq_cardiaca = int(random.uniform(90, 140))
        gravidade_score += 3

    if "falta_de_ar" in sintomas_ativos:
        spo2 = int(random.uniform(80, 93))
        freq_cardiaca = int(random.uniform(95, 130))
        gravidade_score += 3

    if "sangramento_ativo" in sintomas_ativos:
        pas = int(random.uniform(70, 100))
        freq_cardiaca = int(random.uniform(100, 140))
        gravidade_score += 3

    if "confusao_mental" in sintomas_ativos:
        consciencia = random.choice(["Resposta_a_voz", "Resposta_a_dor"])
        gravidade_score += 3

    if "convulsao" in sintomas_ativos:
        consciencia = random.choice(["Resposta_a_dor", "Nao_responsivo"])
        gravidade_score += 4

    if "fraqueza_subita_um_lado" in sintomas_ativos or "dificuldade_falar" in sintomas_ativos:
        gravidade_score += 4

    if "dor_abdominal" in sintomas_ativos:
        gravidade_score += 1

    if "nausea_vomito" in sintomas_ativos:
        gravidade_score += 1

    if "tontura" in sintomas_ativos:
        gravidade_score += 1

    if "dor_cabeca_intensa" in sintomas_ativos:
        gravidade_score += 1

    if "tosse" in sintomas_ativos:
        gravidade_score += 0

    if "dor_nas_costas" in sintomas_ativos:
        gravidade_score += 0

    if "erupcao_cutanea" in sintomas_ativos:
        gravidade_score += 0

    # Ajuste de gravidade por vitais fora da normalidade
    if spo2 < 90:
        gravidade_score += 3
    elif spo2 < 94:
        gravidade_score += 1

    if freq_cardiaca > 130 or freq_cardiaca < 45:
        gravidade_score += 2
    elif freq_cardiaca > 110:
        gravidade_score += 1

    if pas < 90:
        gravidade_score += 2

    if consciencia == "Nao_responsivo":
        gravidade_score += 5
    elif consciencia == "Resposta_a_dor":
        gravidade_score += 3
    elif consciencia == "Resposta_a_voz":
        gravidade_score += 2

    # Mapeia score -> prioridade (1 = mais grave)
    if gravidade_score >= 9:
        prioridade = 1
    elif gravidade_score >= 6:
        prioridade = 2
    elif gravidade_score >= 4:
        prioridade = 3
    elif gravidade_score >= 2:
        prioridade = 4
    else:
        prioridade = 5

    # Define condição sugerida com base no padrão de sintomas predominante
    if "fraqueza_subita_um_lado" in sintomas_ativos or "dificuldade_falar" in sintomas_ativos:
        condicao = "Possivel_AVC"
    elif "dor_no_peito" in sintomas_ativos:
        condicao = "Possivel_Sindrome_Coronariana_Aguda"
    elif "falta_de_ar" in sintomas_ativos:
        condicao = "Possivel_Insuficiencia_Respiratoria"
    elif "convulsao" in sintomas_ativos:
        condicao = "Possivel_Crise_Convulsiva"
    elif "sangramento_ativo" in sintomas_ativos:
        condicao = "Possivel_Trauma_Hemorragia"
    elif "febre" in sintomas_ativos and ("confusao_mental" in sintomas_ativos or temperatura >= 39.5):
        condicao = "Possivel_Sepse_Infeccao_Grave"
    elif "dor_abdominal" in sintomas_ativos or "nausea_vomito" in sintomas_ativos:
        condicao = "Possivel_Quadro_Gastrointestinal"
    elif "febre" in sintomas_ativos or "tosse" in sintomas_ativos:
        condicao = "Possivel_Quadro_Infeccioso_Leve"
    else:
        condicao = "Quadro_Leve_Observacao"

    linha = {
        "temperatura": temperatura,
        "freq_cardiaca": freq_cardiaca,
        "pas": pas,
        "pad": pad,
        "spo2": spo2,
        "consciencia": consciencia,
    }
    for s in SINTOMAS:
        linha[s] = 1 if s in sintomas_ativos else 0

    linha["prioridade"] = prioridade
    linha["condicao"] = condicao

    return linha


def gerar_dataset(n=6000, caminho="model/dataset_simulado.csv"):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    casos = [gerar_caso() for _ in range(n)]
    colunas = list(casos[0].keys())
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(casos)
    print(f"Dataset simulado gerado com {n} casos em: {caminho}")
    return caminho


if __name__ == "__main__":
    gerar_dataset()
