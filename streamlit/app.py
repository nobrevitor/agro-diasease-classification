import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURAÇÕES
# ============================================================

API_URL_MILHO = "https://agro-diasease-api.onrender.com/predict"

DATABRICKS_ENDPOINT_URL = (
    "https://dbc-9c92b906-eef5.cloud.databricks.com/"
    "serving-endpoints/soja/invocations"
)

CLASSES_SOJA = [
    "Crestamento",
    "Ferrugem",
    "Mancha_Parda",
    "Mosaico_Amarelo",
    "Oidio",
    "Podridao_Sul",
    "Queima_Bacteriana",
    "Septoriose",
    "Sindrome_Morte_Subita",
    "Soja_Saudavel",
    "Virus_Mosaico",
]


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(layout="wide", page_title="Detecção de Doenças")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
        background-size: cover;
    }
    h1 {
        background-color: #7ED957;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("DETECÇÃO DE DOENÇAS EM PLANTAS 🌱")


# ============================================================
# FUNÇÕES AUXILIARES — SOJA / DATABRICKS
# ============================================================

def preprocessar_imagem_soja(imagem_arquivo):
    imagem = Image.open(imagem_arquivo).convert("RGB")
    imagem = imagem.resize((224, 224))

    array = np.array(imagem).astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")

    array = (array - mean) / std

    # HWC -> CHW
    array = np.transpose(array, (2, 0, 1))

    # Adiciona dimensão do batch: (1, 3, 224, 224)
    array = np.expand_dims(array, axis=0)

    return array.astype("float32")


def create_tf_serving_json(data):
    return {
        "inputs": {
            name: data[name].tolist()
            for name in data.keys()
        }
    } if isinstance(data, dict) else {
        "inputs": data.tolist()
    }


def score_model(dataset):
    token = st.secrets.get("DATABRICKS_TOKEN") or os.environ.get("DATABRICKS_TOKEN")

    if token is None:
        raise Exception(
            "DATABRICKS_TOKEN não encontrado. "
            "Configure em st.secrets ou como variável de ambiente."
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )

    data_json = json.dumps(ds_dict, allow_nan=True)

    response = requests.request(
        method="POST",
        headers=headers,
        url=DATABRICKS_ENDPOINT_URL,
        data=data_json,
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()


def interpretar_resposta_soja(resposta):
    if "predictions" in resposta:
        scores = np.array(resposta["predictions"][0])
    elif "outputs" in resposta:
        scores = np.array(resposta["outputs"][0])
    else:
        scores = np.array(resposta[0])

    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    indice = int(np.argmax(probs))
    classe = CLASSES_SOJA[indice]
    confianca = float(probs[indice])

    return {
        "prediction": classe,
        "confidence": confianca,
        "scores": probs.tolist()
    }


def enviar_para_databricks_soja(imagem):
    try:
        imagem_array = preprocessar_imagem_soja(imagem)
        resposta = score_model(imagem_array)
        return interpretar_resposta_soja(resposta)

    except Exception as e:
        return {"erro": str(e)}


# ============================================================
# FUNÇÃO AUXILIAR — MILHO / API RENDER
# ============================================================

def enviar_para_api_milho(imagem):
    try:
        files = {
            "file": (
                imagem.name,
                imagem.getvalue(),
                imagem.type
            )
        }

        response = requests.post(
            API_URL_MILHO,
            files=files,
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:
        return {"erro": str(e)}


# ============================================================
# INTERFACE
# ============================================================

tab1, tab2 = st.tabs(["🌱 Soja", "🌽 Milho"])

with tab1:
    st.subheader("Análise de Soja")

    img_file_soja = st.file_uploader(
        "Carregue a foto da folha de soja",
        type=["jpg", "png", "jpeg"],
        key="soja"
    )

    if img_file_soja:
        st.image(img_file_soja, use_container_width=True)

        if st.button("Analisar Soja", key="btn_soja"):
            with st.spinner("IA analisando soja..."):
                resultado = enviar_para_databricks_soja(img_file_soja)

            if "erro" not in resultado:
                st.success(
                    f"Resultado: **{resultado.get('prediction', 'Não identificado')}**"
                )

                confianca = resultado.get("confidence", 0) * 100

                col1, col2 = st.columns(2)
                col1.metric("Confiança da IA", f"{confianca:.2f}%")

                with st.expander("Ver scores por classe"):
                    scores = resultado.get("scores", [])
                    df_scores = pd.DataFrame({
                        "classe": CLASSES_SOJA,
                        "probabilidade": scores
                    }).sort_values("probabilidade", ascending=False)

                    st.dataframe(df_scores, use_container_width=True)

            else:
                st.error(f"Erro na conexão: {resultado['erro']}")


with tab2:
    st.subheader("Análise de Milho")

    img_file_milho = st.file_uploader(
        "Carregue a foto da folha de milho",
        type=["jpg", "png", "jpeg"],
        key="milho"
    )

    if img_file_milho:
        st.image(img_file_milho, use_container_width=True)

        if st.button("Analisar Milho", key="btn_milho"):
            with st.spinner("IA analisando milho..."):
                resultado = enviar_para_api_milho(img_file_milho)

            if "erro" not in resultado:
                st.success(
                    f"Resultado: **{resultado.get('prediction', 'Não identificado')}**"
                )

                confianca = resultado.get("confidence", 0) * 100

                col1, col2 = st.columns(2)
                col1.metric("Confiança da IA", f"{confianca:.2f}%")

            else:
                st.error(f"Erro na conexão: {resultado['erro']}")
                st.warning(
                    "Dica: A primeira análise pode demorar até 1 min para o servidor acordar."
                )