
import numpy as np

class Adaline:
    def __init__(self, input_size:int, weights:list, bias:float, function="degrau-bipolar", x0=-1):
        self.x0 = x0
        self.input_size = input_size
        self.weights = weights
        self.bias = bias
        if self.input_size != len(self.weights):
            raise ValueError('O número de entradas deve ser igual ao número de pesos')
        match function:
            case "degrau-bipolar":
                self.function = lambda x: 1 if x >= 0 else -1
            case "degrau":
                self.function = lambda x: 1 if x >= 0 else 0
            case _:
                raise ValueError('Função de ativação inválida, funções aceitas apenas "degrau" ou "degrau-bipolar"')    
    
    def get_combinadorLinear(self, inputs):
        weights_with_bias = np.insert(self.weights, 0, self.bias)
        inputs = np.insert(inputs, 0, self.x0)
        soma = np.dot(weights_with_bias, inputs)
        return float(soma)

    def predict(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError('O número de entradas deve ser igual ao número de pesos')
               
        soma = self.get_combinadorLinear(inputs)

        return float(self.function(soma))
    
    def get_ErrorQM(self, amostras):
        erro = 0
        p = len(amostras)
        for amostra in amostras:
            inputs = amostra[:-1].copy()
            expected_output = float(amostra[-1])

            combinadorLinear = self.get_combinadorLinear(inputs)
            erro += (expected_output - combinadorLinear) ** 2
        return erro / p

    def get_weights_with_bias(self):
        return np.insert(self.weights, 0, self.bias)

    def get_weights(self):
        return self.weights.copy()
    
    def get_bias(self):
        return self.bias
    
    def set_weights(self, weights):
        if len(weights) != self.input_size:
            raise ValueError('O número de entradas deve ser igual ao número de pesos')
        self.weights = weights.copy()
    
    def set_bias(self, bias):
        self.bias = bias

    def get_x0(self):
        return self.x0
    
    def set_x0(self, x0):
        self.x0 = x0