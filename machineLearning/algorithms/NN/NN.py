import numpy as np
import torch.nn.functional as ftn


class NN:
    # x: input dim
    # hidden_layer: [a, b, c, 1], 3 hidden layers, have a/b/c Neurons, 1 is the output layer
    # W, B is the list, 1 dim
    def __init__(self, x: int, hidden_layer: list):
        # initiate parameters
        self.hiddenLayer = len(hidden_layer)
        self.hidden = hidden_layer
        self.inputDim = x
        self.W = []
        self.B = []

    def check_WB_hidden_number(self, WB: list):
        w = self.hidden.copy()
        w.insert(0, self.inputDim)

        bwCount = len(WB)
        bStandard = sum(self.hidden)            # 加输出层的一个b
        wStandard = self.inputDim * self.hidden[0]
        for i in range(len(self.hidden) - 1):
            wStandard += self.hidden[i] * self.hidden[i + 1]

        if bwCount == wStandard + bStandard:
            self.W = WB[:wStandard]
            self.B = WB[wStandard:]
            self.change_WB_to_array(w)
            return True
        return False

    def change_WB_to_array(self, wSerial):
        w = [[] for x in range(self.hiddenLayer)]
        b = [[] for x in range(self.hiddenLayer)]
        start = 0
        for layer in range(1, len(wSerial)):
            for neuron in range(wSerial[layer]):
                w[layer - 1].append(np.array(self.W[start:(start + wSerial[layer - 1])]))
                start += wSerial[layer - 1]
        start = 0
        for layer_b in range(self.hiddenLayer):
            b[layer_b] = self.B[start:start + self.hidden[layer_b]]
            start += self.hidden[layer_b]
        self.W = w  # list[arr]
        self.B = b  # list

    def calculate(self, x):
        x = np.array(x)
        hiddenKeys = [[] for x in range(self.hiddenLayer)]
        for i in range(self.hidden[0]):
            # 加激活函数
            currentValue = ActivateFunction(
                np.sum(x * self.W[0][i]) + self.B[0][i]
            ).relu()
            hiddenKeys[0].append(currentValue)

        for layer in range(1, self.hiddenLayer):
            for i in range(self.hidden[layer]):
                currentValue = ActivateFunction(
                    np.sum(hiddenKeys[layer - 1] * self.W[layer][i]) + self.B[layer][i]
                ).relu()
                hiddenKeys[layer].append(currentValue)
        return float(hiddenKeys[-1][-1])


class ActivateFunction:
    def __init__(self, x):
        self.x = x

    def relu(self):
        return 1 / (1 + np.exp(-self.x))


# nn = NN(2, [3, 3, 1], [1,2,1,2,1,2,1,2,3,1,2,3,1,2,3,1,2,3, 1, 2, 3, 1,2,3,3])
#
# nn.calculate([1,2])



