from PMC import PMC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados
dados = pd.read_csv("Rede Perceptron Multicamada/dados-apendice3.csv")

# Preparar os dados de entrada e saída
X = dados[['x1', 'x2', 'x3']].values
Y_expected = dados['d'].values

# Definir parâmetros iniciais
E = 1e-6
LR = 0.1
STRUCTURE = [3, 10, 1]

# Instanciar o perceptron
pmc = PMC(STRUCTURE)

EPOCH = 0
EQM = 0
dados = np.hstack((X, Y_expected.reshape(-1, 1)))

epochs = []
eqms = []
flag_epoch = False
INIT_EQM = pmc.get_ErrorQM(dados)

# Treinamento
while True:

    EQM_old = pmc.get_ErrorQM(dados)
    epochs.append(EPOCH)
    eqms.append(EQM_old)

    for amostra in dados:
        inputs = amostra[:-1].copy()
        expected_output = float(amostra[-1])
        y = pmc.predict(inputs)

        pmc.backpropagation(inputs, expected_output, LR)
    

    EPOCH += 1
    EQM = pmc.get_ErrorQM(dados)
    epochs.append(EPOCH)
    eqms.append(EQM_old)
    if EPOCH % 5 == 0:
        if flag_epoch == False:
            valor_antigo = INIT_EQM
            flag_epoch = True
        valor_atual = EQM
        prc = ((valor_atual - valor_antigo) / valor_antigo) * 100
        valor_antigo = valor_atual
        print(f"EPOCA: {EPOCH + 1}, ERRO %: {prc}")   

    if  abs(EQM - EQM_old) <= E:

        break


print(pmc.export_weights_and_bias())


print("Perceptron treinado com sucesso em " + str(EPOCH) + " epocas!")
print("paramentros inicias: ")
print(f"EQM: {INIT_EQM}")

print("\nparamentros finais: ")
print(f"EQM: {EQM}")

# Criar o gráfico
plt.plot(epochs, eqms)

# Adicionar título e rótulos
plt.title('Treinamento PMC - Erro Quadrático Médio')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio')

# Exibir o gráfico  
plt.show()
