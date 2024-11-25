import numpy as np
from Perceptron import Perceptron  # Certifique-se de que a classe Perceptron está importada corretamente.

class PMC:
    def __init__(self, 
                 structure: list, 
                 weight_range=(0, 1), 
                 function="sigmoid", 
                 x0=-1):
        """
        Inicializa uma rede PMC (Perceptron Multicamadas).
        
        :param structure: Lista com o número de neurônios por camada. Exemplo: [3, 5, 2].
        :param weight_range: Tupla com o intervalo dos pesos iniciais. Exemplo: (-1, 1).
        :param function: Nome da função de ativação (string).
        :param x0: Valor do bias fictício único, por camada ou por neurônio.
        """
        self.structure = structure
        self.weight_range = weight_range
        self.layers = []  # Lista que armazenará as camadas de Perceptrons.

        # Dicionário de funções de ativação.
        self.default_functions = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh": lambda x: np.tanh(x),
            "relu": lambda x: max(0, x),
            "leaky-relu": lambda x: x if x > 0 else 0.01 * x,
            "elu": lambda x: x if x > 0 else 0.01 * (np.exp(x) - 1),
            "linear": lambda x: x,
            "swish": lambda x: x / (1 + np.exp(-x)),
            "gaussian": lambda x: np.exp(-x**2),
            "softplus": lambda x: np.log(1 + np.exp(x)),
            "softsign": lambda x: x / (1 + abs(x))
        }

        self.default_functions_derivatives = {
            "sigmoid": lambda x: self.default_functions["sigmoid"](x) * (1 - self.default_functions["sigmoid"](x)),
            "tanh": lambda x: 1 - np.tanh(x)**2,
            "relu": lambda x: 1 if x > 0 else 0,
            "leaky-relu": lambda x: 1 if x > 0 else 0.01,
            "elu": lambda x: 1 if x > 0 else 0.01 * np.exp(x),
            "linear": lambda x: 1,
            "swish": lambda x: self.default_functions["sigmoid"](x) + x * self.default_functions_derivatives["sigmoid"](x),
            "gaussian": lambda x: -2 * x * np.exp(-x**2),
            "softplus": lambda x: 1 / (1 + np.exp(-x)),
            "softsign": lambda x: 1 / (1 + abs(x))**2
        }


        # Validar e definir a função de ativação.
        if function not in self.default_functions:
            raise ValueError(f"Função de ativação '{function}' inválida. Escolha entre: {', '.join(self.default_functions.keys())}.")

        self.function = function

        self.activation_function = self.default_functions[function]

        # Configurar x0 (bias fictício) para todas as camadas.
        if not isinstance(x0, (int, float)):
            raise ValueError("O parâmetro 'x0' deve ser um número.")
        self.x0 = x0

        # Criar camadas (várias instâncias de Perceptron).
        for i in range(1, len(structure)):
            layer = []
            for _ in range(structure[i]):
                weights = np.random.uniform(self.weight_range[0], self.weight_range[1], structure[i - 1]).tolist()
                bias = np.random.uniform(self.weight_range[0], self.weight_range[1])
                perceptron = Perceptron(
                    input_size=structure[i - 1],
                    weights=weights,
                    bias=bias,
                    function=self.activation_function,
                    x0=x0
                )
                layer.append(perceptron)
            self.layers.append(layer)


    def get_all_weights_with_bias(self, one_list=False):
        all_weights = []
        for layer in self.layers:
            layer_weights = []
            for perceptron in layer:
                layer_weights.append(perceptron.get_weights_with_bias())
            all_weights.append(layer_weights)
        if one_list:
            return [weight for layer in all_weights for weight in layer]
        return all_weights
        
    def set_all_weights_with_bias(self, new_weights):
        """
        Atualiza os pesos e bias de todos os perceptrons na rede utilizando os métodos set_weights e set_bias.

        :param new_weights: Lista de arrays numpy contendo os novos pesos e bias para cada perceptron.
                            O primeiro valor do array deve ser o bias.
        """
        # Validar que o número de perceptrons coincide com os pesos fornecidos
        flat_weights = [weight for layer in self.get_all_weights_with_bias(one_list=True) for weight in layer]
        if len(new_weights) != len(flat_weights):
            raise ValueError(
                f"Quantidade de pesos fornecidos ({len(new_weights)}) não corresponde ao número de perceptrons ({len(flat_weights)})"
            )

        # Atualizar os pesos e bias nos perceptrons
        idx = 0
        for layer in self.layers:
            for perceptron in layer:
                weights_with_bias = new_weights[idx]
                
                # Separar bias e pesos
                bias = weights_with_bias[0]
                weights = weights_with_bias[1:]
                
                # Atualizar perceptron
                perceptron.set_bias(bias)
                perceptron.set_weights(weights)
                idx += 1


        
    def get_all_weights(self, one_list=False):
        all_weights = []
        for layer in self.layers:
            layer_weights = []
            for perceptron in layer:
                layer_weights.append(perceptron.get_weights())
            all_weights.append(layer_weights)
        if one_list:
            return [weight for layer in all_weights for weight in layer]
        return all_weights
    
    def import_weights_and_bias(self, data):
        """
        Importa pesos e bias para os perceptrons da rede a partir de uma lista de dicionários.
        :param data: Lista de dicionários contendo o ID do perceptron, seus pesos e seu bias.
        """
        for perceptron_data in data:
            layer_idx = int(perceptron_data["id"][1]) - 1  # Extrair layer index de LxNx
            neuron_idx = int(perceptron_data["id"][3:]) - 1  # Extrair neuron index de LxNx

            perceptron = self.layers[layer_idx][neuron_idx]
            perceptron.set_weights(np.array(perceptron_data["weights"]))
            perceptron.set_bias(perceptron_data["bias"])

    def export_weights_and_bias(self):
        """
        Exporta os pesos e bias de todos os perceptrons da rede em uma estrutura identificável.
        :return: Lista de dicionários contendo o ID do perceptron, seus pesos e seu bias.
        """
        export_data = []
        for layer_idx, layer in enumerate(self.layers, start=1):  # start=1 para Layer começar de 1
            for neuron_idx, perceptron in enumerate(layer, start=1):  # start=1 para Neuron começar de 1
                perceptron_id = f"L{layer_idx}N{neuron_idx}"  # ID no formato LxNx
                export_data.append({
                    "id": perceptron_id,
                    "weights": perceptron.get_weights(),  # Converter numpy array para lista
                    "bias": perceptron.get_bias()
                })
        return export_data

    
    def get_all_bias(self, one_list=False):
        all_bias = []
        for layer in self.layers:
            layer_bias = []
            for perceptron in layer:
                layer_bias.append(perceptron.get_bias())
            all_bias.append(layer_bias)
        if one_list:
            return [bias for layer in all_bias for bias in layer]
        return all_bias

    def get_ErrorQM(self, amostras):
        erro = 0
        p = len(amostras)
        for amostra in amostras:
            inputs = amostra[:-1].copy()
            expected_output = amostra[-1]
            
            # Verifica se expected_output é uma lista e, se não, converte em lista
            if not isinstance(expected_output, list):
                expected_output = [expected_output]
            
            predictions = self.predict(inputs)
            
            # Converte predictions e expected_output para arrays NumPy para garantir operações vetorizadas
            predictions = np.array(predictions)
            expected_output = np.array(expected_output)
            
            # Calcula o erro quadrático médio
            erro += 0.5 * np.sum((predictions - expected_output) ** 2)
            
        return erro / p

        

    def backpropagation(self, inputs, expected_output, LR=0.01):
        """
        Realiza o algoritmo de backpropagation para ajustar os pesos e bias da rede.

        :param inputs: Lista de entradas da rede.
        :param expected_output: Saída esperada para as entradas fornecidas.
        :param LR: Taxa de aprendizado.
        """
        # Obtenha os valores pré-ativação (z) e pós-ativação (y) de todas as camadas
        z_values_all, y_values_all = self.predict(inputs, all_data=True)

        # Inicialize o erro da última camada (delta) com base no erro da saída
        output_layer_output = np.array(y_values_all[-1])  # Saída da última camada
        expected_output = np.array(expected_output)  # Garantir que expected_output é array
        delta = expected_output - output_layer_output  # Erro (delta)
        output_layer_input = np.array(z_values_all[-1])  # Entrada da última camada (z)
        gradiente = delta * self.default_functions_derivatives[self.function](output_layer_input)
        
        # Iterar sobre as camadas ocultas e de saída (ignorando a camada de entrada)
        for layer_index in reversed(range(len(self.layers))):  # Começa do índice 1
            layer = self.layers[layer_index]
            # Obter saída da camada anterior com x0 incluído
            prev_layer_outputs = np.concatenate(([self.x0], np.array(inputs))) if layer_index == 0 else np.concatenate(([self.x0], np.array(y_values_all[layer_index - 1])))

            # Iterar sobre os neurônios da camada atual
            for perceptron_index, perceptron in enumerate(layer):
                # Obter os pesos e bias do perceptron atual
                weights = np.array(perceptron.get_weights())
                bias = perceptron.get_bias()
                # Atualizar os pesos, incluindo o bias
                if len(gradiente) > perceptron_index:
                    new_full_weights = np.concatenate(([bias], weights)) + LR * gradiente[perceptron_index] * prev_layer_outputs
                else:
                    # Se for um único perceptron, usamos gradiente diretamente sem acessar por índice
                    new_full_weights = np.concatenate(([bias], weights)) + LR * gradiente * prev_layer_outputs
                # Separar novamente o bias e os pesos
                new_bias = new_full_weights[0]
                new_weights = new_full_weights[1:]

                # Atualizar os valores no perceptron
                perceptron.set_weights(new_weights.tolist())
                perceptron.set_bias(new_bias)

            # Calcular o gradiente para a próxima camada
            if layer_index > 1:  # Se não for a segunda camada (índice 1), acumula o gradiente
                next_layer_gradiente = gradiente
                weights = np.array([perceptron.get_weights() for perceptron in layer])
                prev_layer_input = np.array(z_values_all[layer_index - 1])
                gradiente = np.sum(next_layer_gradiente[:, None] * weights * self.default_functions_derivatives[self.function](prev_layer_input))


            
                


    def predict(self, inputs: list, all_data=False):
        """
        Realiza a propagação para frente.

        :param inputs: Entradas iniciais para a rede.
        :param all_data: Retorna os valores z (pré-ativação) e y (pós-ativação) de todas as camadas, se True.
        :return: Saída final da rede ou todos os dados intermediários.
        """
        outputs_all_layers = []
        z_values_all_layers = []
        
        for layer in self.layers:
            z_values = []
            outputs = []
            for perceptron in layer:
                z = perceptron.get_combinadorLinear(inputs)  # Valor bruto (pré-ativação)
                y = perceptron.activate_function(z)         # Valor ativado (pós-ativação)
                z_values.append(z)
                outputs.append(y)
            if all_data:
                z_values_all_layers.append(z_values)
                outputs_all_layers.append(outputs)
            inputs = outputs  # A saída da camada atual é entrada da próxima.

        if all_data:
            return z_values_all_layers, outputs_all_layers  # Retorna z's e y's
        return inputs  # Apenas a saída final
