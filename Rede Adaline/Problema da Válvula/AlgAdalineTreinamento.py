from Adaline import Adaline
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Carregar os dados
dados = pd.read_csv("Rede Adaline/dados-apendice2.csv")

# Preparar os dados de entrada e saída
X = dados[['x1', 'x2', 'x3','x4']].values
Y_expected = dados['d'].values

# Definir parâmetros iniciais
E = 1e-6
LR = 0.0025
INPUT_SIZE = len(X[0])
INITIAL_WEIGHTS = [random.random() for _ in range(INPUT_SIZE)]
BIAS = random.random()  # Inicializa o bias no peso do x0

# Instanciar o perceptron
adaline = Adaline(INPUT_SIZE, INITIAL_WEIGHTS, BIAS)  # INPUT_SIZE - 1 para os pesos

EPOCH = 0
EQM = 0
dados = np.hstack((X, Y_expected.reshape(-1, 1)))

epochs = []
eqms = []

#print(f"INPUT_SIZE: {INPUT_SIZE}, INITIAL_WEIGHTS: {INITIAL_WEIGHTS[1:]}, BIAS: {BIAS}")

# Treinamento
while True:

    EQM_old = adaline.get_ErrorQM(dados)
    epochs.append(EPOCH)
    eqms.append(EQM_old)

    for amostra in dados:
            inputs = amostra[:-1].copy()
            expected_output = float(amostra[-1])
            weights = adaline.get_weights_with_bias()
            
            combinador_linear = adaline.get_combinadorLinear(inputs)
            inputs = np.insert(inputs, 0, adaline.get_x0())

            weights = weights + LR * (expected_output - combinador_linear) * inputs

            bias = weights[0]
            weights = weights[1:]
            
            adaline.set_weights(weights)
            adaline.set_bias(bias)

    EPOCH += 1
    EQM = adaline.get_ErrorQM(dados)
    epochs.append(EPOCH)
    eqms.append(EQM_old)
    #if EPOCH % 10000 == 0:
        #print(f"EPOCA: {EPOCH + 1}, ERRO: {ERRO}")   
        #print(f"WEIGHTS: {perceptron.get_weights()}, BIAS: {perceptron.get_bias()}") 

    if abs(EQM_old - EQM) <= E:
        break

end_weights = adaline.get_weights()
end_bias = adaline.get_bias()
print("Perceptron treinado com sucesso em " + str(EPOCH) + " epocas!")
print("paramentros inicias: ")
print(f"WEIGHTS: {INITIAL_WEIGHTS}, BIAS: {BIAS}")

print("\nparamentros finais: ")
print(f"WEIGHTS: {end_weights}, BIAS: {end_bias}")

# Criar o gráfico
plt.plot(epochs, eqms)

# Adicionar título e rótulos
plt.title('Treinamento Adaline - Erro Quadrático Médio')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio')

# Exibir o gráfico  
plt.show()
