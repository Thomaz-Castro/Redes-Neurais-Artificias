from Adaline import Adaline
import pandas as pd
import numpy as np
import random

# x1, x2, x3, x4 vindos do dados-validacao.csv
X = [1.6375,-0.7911,0.7537,0.5515]


for cj_treinamento in range(1, 6):
    #escolhe o conjunto de treinamento
    PESOS = cj_treinamento


    W_T1 = [-1.8130449051389497, 1.312868395607111, 1.6422829375622898, -0.42764327454631984, -1.1777638534032766]

    W_T2 = [-1.8130457226116479, 1.3129039006854841, 1.6423277550476791, -0.4275736518726781, -1.1777898239506002]

    W_T3 = [-1.8131469034931729, 1.3129249734739064, 1.642384493560875, -0.4276367977966956, -1.1778295930389462]

    W_T4 = [-1.8130329669420902, 1.312836747334026, 1.6422396056831714, -0.4276937704823136, -1.1777379772554548]

    W_T5 = [-1.8130297785523948, 1.3128656463043795, 1.6422749477634166, -0.42763306389906963, -1.1777581929788221]

    W_TX = [W_T1, W_T2, W_T3, W_T4, W_T5]


    INPUT_SIZE = 4
    INITIAL_WEIGHTS = W_TX[PESOS-1][1:]
    BIAS = W_TX[PESOS-1][0]

    # Instanciar o perceptron
    perceptron = Adaline(INPUT_SIZE, INITIAL_WEIGHTS, BIAS)  # INPUT_SIZE - 1 para os pesos

    y = perceptron.predict(X)

    print(f"Treinamento {cj_treinamento}: {y}")    

