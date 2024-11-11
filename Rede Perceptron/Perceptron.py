
import numpy as np

class Perceptron:
    def __init__(self, input_size:int, weights:list, bias:float, function="degrau-bipolar"):
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
            
    def predict(self, inputs):
        if len(inputs) != self.input_size+1:
            raise ValueError('O número de entradas deve ser igual ao número de pesos')
        
        weights_with_bias = np.insert(self.weights, 0, self.bias)
        soma = np.dot(weights_with_bias, inputs)

        return float(self.function(soma))
    
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