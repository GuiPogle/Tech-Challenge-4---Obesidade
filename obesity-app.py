# =========================================================
# BIBLIOTECAS
# =========================================================
import datetime
import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_DADOS = os.path.join(BASE_DIR, "df_obesidade_final.xlsx")

# Configuração
st.set_page_config(
    page_title="App para Suporte na Detecção de Obesidade",
    page_icon="🩺",
    layout="wide"
)

ARQUIVO_REGISTROS = "registros_pacientes.xlsx"

labels_obesidade = {
    0: "Baixo peso",
    1: "Peso normal",
    2: "Sobrepeso Nível I",
    3: "Sobrepeso Nível II",
    4: "Obesidade Tipo I",
    5: "Obesidade Tipo II",
    6: "Obesidade Tipo III",
}

# Funções Auxiliares
def carregar_registros():
    if os.path.exists(ARQUIVO_REGISTROS):
        df = pd.read_excel(ARQUIVO_REGISTROS, engine="openpyxl")
        if "Excluir" not in df.columns:
            df.insert(0, "Excluir", False)
        return df
    return pd.DataFrame(columns=["Excluir"])

def salvar_registros():
    df = st.session_state.df_registros.drop(columns=["Excluir"], errors="ignore")
    df.to_excel(ARQUIVO_REGISTROS, index=False, engine="openpyxl")

# Treinamento e valiação do modelo de ML
@st.cache_resource
def modelo_XGBoost():

    df = pd.read_excel(CAMINHO_DADOS, engine="openpyxl")

    df["Screen_Exercise_Ratio"] = (
        df["TUE_tempo_uso_aparelhos_em_horas"] /
        (df["FAF_atividade_fisica_frequente_em_horas"] + 0.1)
    )

    df["Snack_Profile"] = (
        df["CAEC_lanche_entre_refeicoes"] *
        df["FAVC_consumo_frequente_de_alimentos_mt_caloricos"]
    )

    df["Metabolic_Effort"] = df["FAF_atividade_fisica_frequente_em_horas"]
    df.loc[df["MTRANS"].isin(["Walking", "Bike"]), "Metabolic_Effort"] += 1

    df["Vices_Score"] = df["SMOKE"] + df["CALC_bebe_alcool_frequentemente"]

    X = df.drop(columns=["Obesity"])
    y = df["Obesity"]

    num_features = [
        "Age", "Height", "Weight", "CH2O_L",
        "FAF_atividade_fisica_frequente_em_horas",
        "TUE_tempo_uso_aparelhos_em_horas",
        "IMC",
        "Screen_Exercise_Ratio",
        "Snack_Profile",
        "Metabolic_Effort",
        "Vices_Score"
    ]

    cat_features = [
        "Gender", "family_history",
        "FAVC_consumo_frequente_de_alimentos_mt_caloricos",
        "FCVC_comer_vegetais",
        "NCP_refeicoes_principais_diarias",
        "CAEC_lanche_entre_refeicoes",
        "SMOKE",
        "SCC_monitora_calorias_diarias",
        "CALC_bebe_alcool_frequentemente",
        "MTRANS"
    ]

    pipeline = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
        ])),
        ("model", XGBClassifier(
            objective="multi:softprob",
            num_class=7,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist"
        ))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))

    acuracia_media = np.mean(accuracies)

    pipeline.fit(X, y)

    return pipeline, acuracia_media

# Estado dos registros
if "df_registros" not in st.session_state:
    st.session_state.df_registros = carregar_registros()

modelo, acuracia_media = modelo_XGBoost()

# Header
st.title("🩺 App para Suporte na Detecção de Obesidade")

# Legenda
st.subheader("Legenda para preenchimento da ficha:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Binário**")
    st.dataframe(
        pd.DataFrame({
            "Opção": ["no", "yes"],
            "Valor": [0, 1]
        }),
        hide_index=True
    )

with col2:
    st.markdown("**Frequência**")
    st.dataframe(
        pd.DataFrame({
            "Descrição": ["Não", "Às vezes", "Frequentemente", "Sempre"],
            "Valor": [0, 1, 2, 3]
        }),
        hide_index=True
    )

with col3:
    st.markdown("**Grau de Obesidade**")
    st.dataframe(
        pd.DataFrame({
            "Classe": [
                "Baixo peso",
                "Peso normal",
                "Sobrepeso Nível I",
                "Sobrepeso Nível II",
                "Obesidade Tipo I",
                "Obesidade Tipo II",
                "Obesidade Tipo III",
            ],
            "Código": [0, 1, 2, 3, 4, 5, 6]
        }),
        hide_index=True
    )

# Anamnese
st.header("Anamnese do Paciente")

with st.form("form_anamnese"):

    nome_paciente = st.text_input("Nome do paciente")

    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.number_input("Idade", 0, 110, 25)
    height = st.number_input("Altura (m)", 1.2, 2.5, 1.70)
    weight = st.number_input("Peso (kg)", 30.0, 300.0, 70.0)

    family_history = st.selectbox("Histórico familiar de obesidade", [0, 1])
    favc = st.selectbox("Consumo frequente de alimentos calóricos", [0, 1])
    fcvc = st.slider("Consumo de vegetais", 1, 3)
    ncp = st.slider("Refeições principais/dia", 1, 4)
    caec = st.slider("Lanches entre refeições", 0, 3)
    smoke = st.selectbox("Fuma", [0, 1])
    ch2o = st.slider("Consumo diário de água", 0, 3)
    scc = st.selectbox("Monitora calorias", [0, 1])
    faf = st.slider("Atividade física (freq.)", 0, 3)
    tue = st.slider("Tempo de tela (h)", 0, 24)
    calc = st.slider("Consumo de álcool", 0, 4)
    mtrans = st.selectbox(
        "Meio de transporte",
        ["Automobile", "Motorbike", "Public_Transportation", "Walking", "Bike"]
    )

    submitted = st.form_submit_button("Calcular Predição")

# Predição
if submitted:

    imc = weight / (height ** 2)
    screen_exercise_ratio = tue / (faf + 0.1)
    snack_profile = caec * favc
    metabolic_effort = faf + (1 if mtrans in ["Walking", "Bike"] else 0)
    vices_score = smoke + calc

    df_paciente = pd.DataFrame([{
        "Nome": nome_paciente,
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "CH2O_L": ch2o,
        "FAF_atividade_fisica_frequente_em_horas": faf,
        "TUE_tempo_uso_aparelhos_em_horas": tue,
        "IMC": imc,
        "Screen_Exercise_Ratio": screen_exercise_ratio,
        "Snack_Profile": snack_profile,
        "Metabolic_Effort": metabolic_effort,
        "Vices_Score": vices_score,
        "family_history": family_history,
        "FAVC_consumo_frequente_de_alimentos_mt_caloricos": favc,
        "FCVC_comer_vegetais": fcvc,
        "NCP_refeicoes_principais_diarias": ncp,
        "CAEC_lanche_entre_refeicoes": caec,
        "SMOKE": smoke,
        "SCC_monitora_calorias_diarias": scc,
        "CALC_bebe_alcool_frequentemente": calc,
        "MTRANS": mtrans
    }])

    predicao = modelo.predict(df_paciente)[0]

    st.success(f"Classificação prevista: **{labels_obesidade[predicao]}**")

    st.markdown(
        f"""
        <div style="
            background-color:#0e1117;
            padding:10px;
            border-radius:8px;
            border:1px solid #262730;
            width:fit-content;
            margin-top:5px;
        ">
            <strong>Acurácia média do modelo:</strong> {acuracia_media:.2%}
        </div>
        """,
        unsafe_allow_html=True
    )

    registro = df_paciente.copy()
    registro.insert(0, "Excluir", False)
    registro["Data"] = datetime.datetime.now()
    registro["Predicao_Obesidade"] = labels_obesidade[predicao]

    st.session_state.df_registros = pd.concat(
        [st.session_state.df_registros, registro],
        ignore_index=True
    )

    salvar_registros()

# Registro de pacientes
st.header("📊 Histórico de Pacientes Registrados")

if not st.session_state.df_registros.empty:

    st.session_state.df_registros = st.data_editor(
        st.session_state.df_registros,
        num_rows="fixed",
        use_container_width=True
    )

    if st.button("🗑️ Excluir registro(s) selecionado(s)"):
        mask_excluir = st.session_state.df_registros["Excluir"].astype(bool)

        if mask_excluir.any():
            st.session_state.df_registros = (
                st.session_state.df_registros
                .loc[~mask_excluir]
                .reset_index(drop=True)
            )
            salvar_registros()
            st.success("Registro(s) excluído(s) com sucesso.")
            st.rerun()
        else:
            st.warning("Nenhum registro selecionado.")
else:
    st.info("Nenhum paciente registrado até o momento.")