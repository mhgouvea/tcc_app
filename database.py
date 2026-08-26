"""
Camada de armazenamento do sistema (SQLite), conforme a arquitetura em
três camadas descrita na seção 3.2 do TCC:
  1) Interface web (Flask/HTML/CSS)
  2) Processamento (IA / Random Forest)
  3) Armazenamento (registro de pacientes e triagens)
"""

import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "triagem.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            data_nascimento TEXT,
            sexo TEXT,
            cartao_sus TEXT,
            criado_em TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS triagens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id INTEGER NOT NULL,
            temperatura REAL,
            freq_cardiaca INTEGER,
            pas INTEGER,
            pad INTEGER,
            spo2 INTEGER,
            consciencia TEXT,
            sintomas TEXT,
            prioridade INTEGER,
            prioridade_label TEXT,
            condicao_sugerida TEXT,
            confianca_prioridade REAL,
            confianca_condicao REAL,
            decisao_profissional TEXT,
            observacoes TEXT,
            criado_em TEXT NOT NULL,
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
        );
        """
    )
    conn.commit()
    conn.close()


def criar_paciente(nome, data_nascimento, sexo, cartao_sus):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO pacientes (nome, data_nascimento, sexo, cartao_sus, criado_em) VALUES (?, ?, ?, ?, ?)",
        (nome, data_nascimento, sexo, cartao_sus, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    novo_id = cur.lastrowid
    conn.close()
    return novo_id


def listar_pacientes(busca=None):
    conn = get_conn()
    if busca:
        rows = conn.execute(
            "SELECT * FROM pacientes WHERE nome LIKE ? ORDER BY nome", (f"%{busca}%",)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM pacientes ORDER BY nome").fetchall()
    conn.close()
    return rows


def obter_paciente(paciente_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()
    conn.close()
    return row


def registrar_triagem(dados):
    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO triagens (
            paciente_id, temperatura, freq_cardiaca, pas, pad, spo2, consciencia,
            sintomas, prioridade, prioridade_label, condicao_sugerida,
            confianca_prioridade, confianca_condicao, decisao_profissional,
            observacoes, criado_em
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            dados["paciente_id"],
            dados["temperatura"],
            dados["freq_cardiaca"],
            dados["pas"],
            dados["pad"],
            dados["spo2"],
            dados["consciencia"],
            dados["sintomas"],
            dados["prioridade"],
            dados["prioridade_label"],
            dados["condicao_sugerida"],
            dados["confianca_prioridade"],
            dados["confianca_condicao"],
            dados.get("decisao_profissional", ""),
            dados.get("observacoes", ""),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    novo_id = cur.lastrowid
    conn.close()
    return novo_id


def listar_triagens(paciente_id=None):
    conn = get_conn()
    if paciente_id:
        rows = conn.execute(
            """SELECT t.*, p.nome as paciente_nome FROM triagens t
               JOIN pacientes p ON p.id = t.paciente_id
               WHERE t.paciente_id = ? ORDER BY t.criado_em DESC""",
            (paciente_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT t.*, p.nome as paciente_nome FROM triagens t
               JOIN pacientes p ON p.id = t.paciente_id
               ORDER BY t.criado_em DESC LIMIT 200"""
        ).fetchall()
    conn.close()
    return rows


def obter_triagem(triagem_id):
    conn = get_conn()
    row = conn.execute(
        """SELECT t.*, p.nome as paciente_nome FROM triagens t
           JOIN pacientes p ON p.id = t.paciente_id
           WHERE t.id = ?""",
        (triagem_id,),
    ).fetchone()
    conn.close()
    return row
