# Sistema de Apoio à Decisão Clínica baseado em IA para Triagem de Pacientes

Protótipo funcional desenvolvido a partir do TCC de **Matheus Henrique Gouvêa Nunes**
(Curso de Sistemas de Informação — UNIARA), orientação do Prof. **André Luiz da Silva**.

Implementa a arquitetura em três camadas descrita no trabalho (seção 3.2):

1. **Interface web** (Flask + HTML/CSS) — cadastro de pacientes e triagem.
2. **Processamento (IA)** — dois modelos `RandomForestClassifier` (Scikit-learn):
   um para o **nível de prioridade** (escala de 1 a 5, inspirada no Protocolo de
   Manchester) e outro para a **condição clínica sugerida**.
3. **Armazenamento** — banco de dados SQLite com pacientes e histórico de triagens.

⚠️ **Aviso importante**: os modelos foram treinados com dados **simulados**
(`model/generate_dataset.py`), gerados por regras que aproximam a lógica do
Protocolo de Manchester. Este é um protótipo acadêmico — **não deve ser usado
para decisões clínicas reais** sem validação clínica formal, conforme discutido
na seção 2.5 do TCC (ética, LGPD e Resolução CNS 466/2012).

## Estrutura do projeto

```
tcc_app/
├── app.py                     # Aplicação Flask (rotas / interface web)
├── database.py                # Camada de armazenamento (SQLite)
├── ia_processamento.py        # Camada de processamento (carrega e aplica os modelos)
├── requirements.txt
├── model/
│   ├── generate_dataset.py    # Geração do dataset simulado
│   ├── train_model.py         # Treinamento dos modelos Random Forest
│   ├── dataset_simulado.csv   # Dataset gerado (6000 casos)
│   ├── modelo_prioridade.pkl  # Modelo treinado (prioridade)
│   ├── modelo_condicao.pkl    # Modelo treinado (condição clínica)
│   ├── encoder_consciencia.pkl
│   ├── encoder_condicao.pkl
│   └── features.pkl
├── templates/                 # Páginas HTML (Jinja2)
├── static/css/style.css       # Estilo visual do sistema
└── data/triagem.db            # Banco SQLite (criado automaticamente)
```

## Como executar

```bash
# 1. Criar ambiente virtual (opcional, recomendado)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. (Opcional) Retreinar os modelos com um novo dataset simulado
cd model
python3 train_model.py
cd ..

# 4. Rodar a aplicação
python3 app.py
```

Acesse **http://localhost:5000** no navegador.

## Fluxo de uso (conforme seção 3.4 do TCC)

1. O profissional acessa a interface web e cadastra o paciente (dados de
   identificação).
2. Informa os sinais vitais (temperatura, frequência cardíaca, pressão
   arterial, saturação de O₂, nível de consciência) e os sintomas relatados.
3. O módulo de processamento aplica os modelos de IA e retorna a sugestão de
   **condição clínica** e o **nível de prioridade** de atendimento.
4. O profissional registra sua decisão (acatar, ajustar ou divergir da
   sugestão), mantendo a autonomia e a responsabilidade pela decisão final.
5. A triagem é armazenada no histórico, disponível por paciente e de forma
   geral, para acompanhamento e futura geração de relatórios.

## Métricas do modelo (dados simulados, seção 3.3 do TCC)

O script `model/train_model.py` avalia os dois modelos com **acurácia,
precisão, recall, F1-score e validação cruzada (5-fold)**, impressas no
console ao final do treinamento.

## Variáveis utilizadas

- **Sinais vitais**: temperatura corporal, frequência cardíaca, pressão
  arterial sistólica/diastólica, saturação de O₂, nível de consciência.
- **Sintomas**: febre, dor no peito, falta de ar, tosse, dor de cabeça
  intensa, tontura, náusea/vômito, dor abdominal, sangramento ativo,
  confusão mental, convulsão, fraqueza súbita de um lado do corpo,
  dificuldade para falar, dor nas costas, erupção cutânea.

## Próximos passos sugeridos (alinhados ao cronograma do TCC)

- Testes de validação mais amplos com casos clínicos revisados por
  especialistas.
- Geração de relatórios administrativos/gerenciais a partir do histórico.
- Ajustes de segurança e conformidade com a LGPD antes de qualquer uso com
  dados reais de pacientes.
