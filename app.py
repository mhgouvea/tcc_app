"""
SISTEMA DE APOIO À DECISÃO CLÍNICA BASEADO EM IA PARA TRIAGEM DE PACIENTES
Protótipo desenvolvido a partir do TCC de Matheus Henrique Gouvêa Nunes
(Sistemas de Informação - UNIARA), orientação de André Luiz da Silva.

Camada de interface web (Flask/HTML/CSS) - seção 3.2 do TCC.

Este é um PROTÓTIPO ACADÊMICO, treinado com dados simulados, e não deve
ser utilizado para decisões clínicas reais. Ele existe para demonstrar a
arquitetura proposta no trabalho: interface web -> processamento (IA) ->
armazenamento.
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import database as db
from ia_processamento import get_motor, SINTOMAS, SINTOMAS_LABELS, classificar_pressao_arterial

app = Flask(__name__)
app.secret_key = "tcc-triagem-ia-chave-secreta-dev"  # apenas para uso local/demonstração

db.init_db()


@app.context_processor
def inject_globals():
    return {"sintomas_lista": SINTOMAS, "sintomas_labels": SINTOMAS_LABELS}


@app.route("/")
def index():
    triagens_recentes = db.listar_triagens()[:8]
    pacientes = db.listar_pacientes()
    return render_template(
        "index.html",
        total_pacientes=len(pacientes),
        total_triagens=len(db.listar_triagens()),
        triagens_recentes=triagens_recentes,
    )


# --------------------- PACIENTES ---------------------

@app.route("/pacientes")
def pacientes_lista():
    busca = request.args.get("busca", "").strip()
    pacientes = db.listar_pacientes(busca or None)
    return render_template("pacientes.html", pacientes=pacientes, busca=busca)


@app.route("/pacientes/novo", methods=["GET", "POST"])
def paciente_novo():
    if request.method == "POST":
        nome = request.form.get("nome", "").strip()
        data_nascimento = request.form.get("data_nascimento", "").strip()
        sexo = request.form.get("sexo", "").strip()
        cartao_sus = request.form.get("cartao_sus", "").strip()

        if not nome:
            flash("Informe o nome do paciente.", "erro")
            return render_template("cadastro_paciente.html", form=request.form)

        novo_id = db.criar_paciente(nome, data_nascimento, sexo, cartao_sus)
        flash("Paciente cadastrado com sucesso.", "sucesso")
        return redirect(url_for("triagem_nova", paciente_id=novo_id))

    return render_template("cadastro_paciente.html", form={})


@app.route("/pacientes/<int:paciente_id>")
def paciente_detalhe(paciente_id):
    paciente = db.obter_paciente(paciente_id)
    if not paciente:
        flash("Paciente não encontrado.", "erro")
        return redirect(url_for("pacientes_lista"))
    triagens = db.listar_triagens(paciente_id)
    return render_template("paciente_detalhe.html", paciente=paciente, triagens=triagens)


# --------------------- TRIAGEM ---------------------

@app.route("/triagem/novo")
def triagem_nova_sem_paciente():
    return redirect(url_for("pacientes_lista"))


@app.route("/triagem/<int:paciente_id>", methods=["GET", "POST"])
def triagem_nova(paciente_id):
    paciente = db.obter_paciente(paciente_id)
    if not paciente:
        flash("Paciente não encontrado.", "erro")
        return redirect(url_for("pacientes_lista"))

    if request.method == "POST":
        try:
            vitais = {
                "temperatura": float(request.form.get("temperatura")),
                "freq_cardiaca": int(request.form.get("freq_cardiaca")),
                "pas": int(request.form.get("pas")),
                "pad": int(request.form.get("pad")),
                "spo2": int(request.form.get("spo2")),
                "consciencia": request.form.get("consciencia"),
            }
        except (TypeError, ValueError):
            flash("Verifique os valores numéricos dos sinais vitais.", "erro")
            return render_template("triagem.html", paciente=paciente, form=request.form)

        sintomas_selecionados = request.form.getlist("sintomas")
        observacoes = request.form.get("observacoes", "").strip()

        motor = get_motor()
        resultado = motor.classificar(vitais, sintomas_selecionados)

        triagem_id = db.registrar_triagem(
            {
                "paciente_id": paciente_id,
                "temperatura": vitais["temperatura"],
                "freq_cardiaca": vitais["freq_cardiaca"],
                "pas": vitais["pas"],
                "pad": vitais["pad"],
                "spo2": vitais["spo2"],
                "consciencia": vitais["consciencia"],
                "sintomas": ",".join(sintomas_selecionados),
                "prioridade": resultado["prioridade"],
                "prioridade_label": resultado["prioridade_label"],
                "condicao_sugerida": resultado["condicao_label"],
                "confianca_prioridade": resultado["confianca_prioridade"],
                "confianca_condicao": resultado["confianca_condicao"],
                "observacoes": observacoes,
            }
        )

        return redirect(url_for("triagem_resultado", triagem_id=triagem_id))

    return render_template("triagem.html", paciente=paciente, form={})


@app.route("/triagem/resultado/<int:triagem_id>", methods=["GET", "POST"])
def triagem_resultado(triagem_id):
    triagem = db.obter_triagem(triagem_id)
    if not triagem:
        flash("Triagem não encontrada.", "erro")
        return redirect(url_for("index"))

    if request.method == "POST":
        decisao = request.form.get("decisao_profissional", "")
        conn = db.get_conn()
        conn.execute(
            "UPDATE triagens SET decisao_profissional = ? WHERE id = ?",
            (decisao, triagem_id),
        )
        conn.commit()
        conn.close()
        flash("Decisão do profissional registrada.", "sucesso")
        return redirect(url_for("triagem_resultado", triagem_id=triagem_id))

    sintomas_marcados = [s for s in (triagem["sintomas"] or "").split(",") if s]
    pressao = classificar_pressao_arterial(triagem["pas"], triagem["pad"])
    return render_template(
        "resultado.html",
        triagem=triagem,
        sintomas_marcados=[SINTOMAS_LABELS.get(s, s) for s in sintomas_marcados],
        pressao=pressao,
    )


# --------------------- HISTÓRICO ---------------------

@app.route("/historico")
def historico():
    triagens = db.listar_triagens()
    return render_template("historico.html", triagens=triagens)


# --------------------- API (uso opcional / integração) ---------------------

@app.route("/api/classificar", methods=["POST"])
def api_classificar():
    """Endpoint JSON para testar o motor de IA diretamente (sem persistir)."""
    payload = request.get_json(force=True)
    try:
        vitais = {
            "temperatura": float(payload["temperatura"]),
            "freq_cardiaca": int(payload["freq_cardiaca"]),
            "pas": int(payload["pas"]),
            "pad": int(payload["pad"]),
            "spo2": int(payload["spo2"]),
            "consciencia": payload["consciencia"],
        }
        sintomas_selecionados = payload.get("sintomas", [])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"erro": f"Payload inválido: {e}"}), 400

    motor = get_motor()
    resultado = motor.classificar(vitais, sintomas_selecionados)
    resultado["pressao_arterial"] = classificar_pressao_arterial(vitais["pas"], vitais["pad"])
    return jsonify(resultado)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
