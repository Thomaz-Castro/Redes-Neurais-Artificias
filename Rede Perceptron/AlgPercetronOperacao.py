from Perceptron import Perceptron
import pandas as pd
import numpy as np
import random

# x1, x2, x3 vindos do dados-validacao.csv
X = [-1.8842,-0.2805,1.2548]


for cj_treinamento in range(1, 6):
    #escolhe o conjunto de treinamento
    PESOS = cj_treinamento


    W_T1 = [-3.071445634859616, 1.56257321, 2.47580116, -0.73253227]

    W_T2 = [-2.9163644372935456, 1.4286616, 2.40014206, -0.67964693]

    W_T3 = [-2.899059948257159, 1.40208578, 2.39647904, -0.69419459]

    W_T4 = [-3.1027082028121193, 1.6014818, 2.5071401, -0.74118858]

    W_T5 = [-3.1762558519812805, 1.60341652, 2.54741744, -0.75459944]

    W_TX = [W_T1, W_T2, W_T3, W_T4, W_T5]


    INPUT_SIZE = 3
    INITIAL_WEIGHTS = W_TX[PESOS-1][1:]
    BIAS = W_TX[PESOS-1][0]

    # Instanciar o perceptron
    perceptron = Perceptron(INPUT_SIZE, INITIAL_WEIGHTS, BIAS)  # INPUT_SIZE - 1 para os pesos

    y = perceptron.predict(X)

    print(f"Treinamento {cj_treinamento}: {y}")    

