import pandas as pd
import random

dados = []

# Guerreiros
for _ in range(150):
    dados.append([
        random.randint(7,10),
        random.randint(1,5),
        random.randint(1,4),
        random.randint(7,10),
        "Guerreiro"
    ])

# Arqueiros
for _ in range(150):
    dados.append([
        random.randint(2,5),
        random.randint(7,10),
        random.randint(2,5),
        random.randint(3,7),
        "Arqueiro"
    ])

# Magos
for _ in range(150):
    dados.append([
        random.randint(1,4),
        random.randint(1,5),
        random.randint(7,10),
        random.randint(2,6),
        "Mago"
    ])

df = pd.DataFrame(
    dados,
    columns=[
        "forca",
        "agilidade",
        "inteligencia",
        "vitalidade",
        "classe"
    ]
)

df.to_csv(
    "personagens_rpg.csv",
    index=False
)

print("Dataset criado com sucesso.")