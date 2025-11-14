from flask import Flask, render_template, request, jsonify
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# -------- CONFIGURAÇÃO DA API GROQ (segura) --------
GROQ_KEY = os.environ.get("GROQ_API_KEY")
chat = ChatGroq(model="llama-3.3-70b-versatile")

# -------- PROMPT DA CATEQUISTA (personalidade) --------
PROMPT_CATEQUISTA = """
Você é uma Catequista Virtual Católico-Romana, especializada em catequese de adultos.
Ensina com fidelidade ao Magistério, clareza, profundidade e caridade.

Diretrizes:
- Ensina de acordo com a Bíblia, o Catecismo da Igreja Católica e a Tradição.
- Nunca inventa doutrina.
- Explica de forma clara e objetiva.
- Ajuda como apoio catequético, sem substituir o acompanhamento humano.

Estilo:
- Calma, firme, amorosa e fiel ao Magistério.
- Usa linguagem acessível mas profunda.
- Sempre dá referências quando necessário.
"""

def gerar_resposta(historico):
    """Gera resposta usando LangChain + Groq, com histórico limitado."""

    # Limitar o histórico (reduz custo e acelera)
    historico_limitado = historico[-6:]

    mensagens_modelo = [('system', PROMPT_CATEQUISTA)]
    mensagens_modelo.extend(historico_limitado)

    template = ChatPromptTemplate.from_messages(mensagens_modelo)
    chain = template | chat

    try:
        resposta = chain.invoke({}).content
        return resposta
    except Exception as e:
        return f"Ocorreu um erro ao gerar a resposta: {e}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.json
    historico = data.get("historico", [])

    resposta = gerar_resposta(historico)

    return jsonify({"resposta": resposta})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
