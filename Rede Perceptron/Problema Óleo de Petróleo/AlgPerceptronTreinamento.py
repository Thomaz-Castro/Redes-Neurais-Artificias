from Perceptron import Perceptron
import pandas as pd
import numpy as np
import random

# Carregar os dados
dados = pd.read_csv("Rede Perceptron/dados-apendice1.csv")

# Preparar os dados de entrada e saída
X = dados[['x1', 'x2', 'x3']].values
Y_expected = dados['d'].values

# Definir parâmetros iniciais
LR = 0.01
INPUT_SIZE = len(X[0])
INITIAL_WEIGHTS = [random.random() for _ in range(INPUT_SIZE)]
BIAS = random.random()  # Inicializa o bias no peso do x0

# Instanciar o perceptron
perceptron = Perceptron(INPUT_SIZE, INITIAL_WEIGHTS, BIAS)  # INPUT_SIZE - 1 para os pesos

EPOCH = 0
ERRO = 0
dados = np.hstack((X, Y_expected.reshape(-1, 1)))

#print(f"INPUT_SIZE: {INPUT_SIZE}, INITIAL_WEIGHTS: {INITIAL_WEIGHTS[1:]}, BIAS: {BIAS}")

# Treinamento
while True:
    ERRO = 0
    for amostra in dados:
        inputs = amostra[:-1].copy()
        expected_output = float(amostra[-1])

        prediction = perceptron.predict(inputs)
        if prediction != expected_output:
            ERRO += 1 

            weights = perceptron.get_weights()
            bias = perceptron.get_bias()
            weights = np.insert(weights, 0, bias)

            inputs = np.insert(inputs, 0, perceptron.get_x0())
            
            weights = weights + LR * (expected_output - prediction) * inputs

            bias = weights[0]
            weights = weights[1:]
            
            perceptron.set_weights(weights)
            perceptron.set_bias(bias)

    #if EPOCH % 10000 == 0:
        #print(f"EPOCA: {EPOCH + 1}, ERRO: {ERRO}")   
        #print(f"WEIGHTS: {perceptron.get_weights()}, BIAS: {perceptron.get_bias()}") 
    if ERRO == 0:
        break
    EPOCH += 1

end_weights = perceptron.get_weights()
end_bias = perceptron.get_bias()
print("Perceptron treinado com sucesso em " + str(EPOCH) + " epocas!")
print("paramentros inicias: ")
print(f"WEIGHTS: {INITIAL_WEIGHTS}, BIAS: {BIAS}")

print("\nparamentros finais: ")
print(f"WEIGHTS: {end_weights}, BIAS: {end_bias}")
