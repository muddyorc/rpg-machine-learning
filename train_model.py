import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import joblib

print("--- Iniciando Treinamento e Salvamento do Modelo ---\n")

try:
    df = pd.read_csv("personagens_rpg.csv")
    print("Dataset RPG carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar dataset: {e}")
    raise

print("2. Pré-processando e limpando dados...")

df.dropna(inplace=True)

print(
    f"Dados após limpeza: {df.shape[0]} linhas, "
    f"{df.shape[1]} colunas.\n"
)

features = [
    "forca",
    "agilidade",
    "inteligencia",
    "vitalidade"
]

target = "classe"

X = df[features]
y = df[target]

print(f"Features selecionadas: {features}")
print(f"Variável alvo: {target}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    f"Dados divididos: "
    f"Treino ({X_train.shape[0]} amostras), "
    f"Teste ({X_test.shape[0]} amostras).\n"
)

numerical_features = [
    "forca",
    "agilidade",
    "inteligencia",
    "vitalidade"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        )
    ]
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        )
    ]
)

print(
    "Pipeline de pré-processamento "
    "e modelo configurado.\n"
)

print("6. Treinando o modelo...")

model.fit(
    X_train,
    y_train
)

print("Modelo treinado com sucesso!\n")


print("7. Avaliando o modelo...\n")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Acurácia do modelo: {accuracy:.2f}\n")

print("Relatório de Classificação:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


joblib.dump(
    model,
    "rpg_classifier.pkl"
)

joblib.dump(
    features,
    "model_features.pkl"
)

joblib.dump(
    model.classes_,
    "model_classes.pkl"
)

print(
    "\nModelo salvo como:\n"
    "rpg_classifier.pkl\n"
    "model_features.pkl\n"
    "model_classes.pkl"
)

print(
    "\n--- Treinamento e Salvamento Concluídos ---"
)