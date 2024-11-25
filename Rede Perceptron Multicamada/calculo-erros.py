import numpy as np

# Valores desejados (d) fornecidos na tabela
d = np.array([
    0.4831, 0.5965, 0.5318, 0.6843, 0.2872,
    0.7663, 0.5666, 0.6601, 0.5427, 0.5836,
    0.6950, 0.6790, 0.2956, 0.7742, 0.4662,
    0.8093, 0.7581, 0.5826, 0.7938, 0.5012
])

y_gerado = np.array([0.4747012215154336, 0.5820686085509451, 0.5160278590512497, 0.7023155380324169, 0.30746875949722324, 0.7536467034539235, 0.5560377345935417, 0.6729389640930182, 0.5197034861016603, 0.5910429870232679, 0.6856095965968092, 0.6668916962474581, 0.3144159027477791, 0.7881981008340978, 0.455468126657403, 0.8259702972202417, 0.7844462079115723, 0.5812091884715517, 0.8040852027758838, 0.4843450433469388]
)


def calculate_metrics(y):
    # Verifica se o tamanho do array é válido
    if len(y) != len(d):
        raise ValueError(
            "O array fornecido deve ter 20 valores correspondentes ao conjunto de amostras.")

    # Calcula o erro relativo para cada amostra
    erro_relativo = np.abs((y - d) / d) * 100

    # Calcula o erro relativo médio (ERM)
    erro_relativo_medio = np.mean(erro_relativo)

    # Calcula a variância dos erros relativos
    variancia = np.var(erro_relativo)

    return erro_relativo_medio, variancia


erm, var = calculate_metrics(y_gerado)

print(f"Erro Relativo Médio (ERM): {erm:.2f}%")
print(f"Variância: {var:.2f}%")
