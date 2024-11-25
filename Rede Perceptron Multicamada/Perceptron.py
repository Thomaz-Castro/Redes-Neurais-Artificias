
import numpy as np

class Perceptron:
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
                if callable(function):
                    self.function = function
                else:
                    raise ValueError('Função de ativação inválida, passe uma lambda ou uma função valida')    
    
    def get_combinadorLinear(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError('O número de entradas deve ser igual ao número de pesos')
        weights_with_bias = np.insert(self.weights, 0, self.bias)
        inputs = np.insert(inputs, 0, self.x0)
        return np.dot(weights_with_bias, inputs)


    def predict(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError('O número de entradas deve ser igual ao número de pesos')
        
        soma = self.get_combinadorLinear(inputs)

        return float(self.activate_function(soma))
    
    def activate_function(self, x):
        return self.function(x)    
    
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