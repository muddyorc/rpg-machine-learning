import streamlit as st
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_resource
def load_model_assets():

    try:

        model = joblib.load(
            "rpg_classifier.pkl"
        )

        features = joblib.load(
            "model_features.pkl"
        )

        classes = joblib.load(
            "model_classes.pkl"
        )

        return model, features, classes

    except FileNotFoundError:

        st.error(
            "Arquivos do modelo não encontrados.\n"
            "Execute train_model.py primeiro."
        )

        st.stop()


model, model_features, model_classes = (
    load_model_assets()
)


st.title(
    "🎮 Classificador de Classes RPG"
)

st.write(
    """
    Este aplicativo utiliza Machine Learning
    para prever a classe de um personagem RPG.

    Classes possíveis:

    - Guerreiro
    - Arqueiro
    - Mago
    """
)

st.sidebar.header(
    "Parâmetros do Personagem"
)

forca = st.sidebar.slider(
    "Força",
    1,
    10,
    5
)

agilidade = st.sidebar.slider(
    "Agilidade",
    1,
    10,
    5
)

inteligencia = st.sidebar.slider(
    "Inteligência",
    1,
    10,
    5
)

vitalidade = st.sidebar.slider(
    "Vitalidade",
    1,
    10,
    5
)


user_input_df = pd.DataFrame(
    [[
        forca,
        agilidade,
        inteligencia,
        vitalidade
    ]],
    columns=model_features
)

st.subheader(
    "Dados de Entrada do Usuário"
)

st.write(
    user_input_df
)


if st.sidebar.button(
    "Classificar Personagem"
):

    st.subheader(
        "Resultado da Classificação"
    )

    try:

        prediction = model.predict(
            user_input_df
        )

        prediction_proba = (
            model.predict_proba(
                user_input_df
            )
        )

        st.success(
            f"Classe prevista: "
            f"**{prediction[0]}**"
        )

        st.write(
            "Probabilidades:"
        )

        proba_df = pd.DataFrame(
            prediction_proba,
            columns=model_classes
        ).transpose()

        proba_df.columns = [
            "Probabilidade"
        ]

        st.dataframe(
            proba_df.style.format(
                {
                    "Probabilidade":
                    "{:.2%}"
                }
            )
        )

        fig, ax = plt.subplots()

        ax.bar(
            model_classes,
            proba_df[
                "Probabilidade"
            ]
        )

        ax.set_ylabel(
            "Probabilidade"
        )

        ax.set_title(
            "Probabilidade por Classe"
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Erro na previsão: {e}"
        )


st.subheader(
    "Contexto dos Dados RPG"
)

st.write(
    """
    Visualizações do conjunto
    utilizado para treinar a IA.
    """
)

try:

    original_df = pd.read_csv(
        "personagens_rpg.csv"
    )


    fig_scatter, ax_scatter = (
        plt.subplots()
    )

    sns.scatterplot(
        data=original_df,
        x="forca",
        y="agilidade",
        hue="classe",
        ax=ax_scatter
    )

    ax_scatter.set_title(
        "Força x Agilidade"
    )

    st.pyplot(
        fig_scatter
    )


    fig_hist, ax_hist = (
        plt.subplots()
    )

    sns.histplot(
        original_df,
        x="inteligencia",
        hue="classe",
        kde=True,
        ax=ax_hist
    )

    ax_hist.set_title(
        "Distribuição da Inteligência"
    )

    st.pyplot(
        fig_hist
    )

except Exception as e:

    st.warning(
        f"Erro ao carregar "
        f"visualizações: {e}"
    )