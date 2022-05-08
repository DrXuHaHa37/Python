import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as ftn
from algorithms import draw


class BP:
    def __init__(self, x: np.array, y: np.array, generation, learnRate, hidden):

        self.generation = generation
        self.input = torch.tensor(x, dtype=torch.float)
        self.input_groups, self.input_attrs = self.input.shape
        self.output = torch.tensor(y, dtype=torch.float)
        self.output_groups, self.output_attrs = self.output.shape[0], 1
        self.hidden = hidden
        self.net = Net(self.input_attrs,
                       self.hidden,
                       self.output_attrs
                       )
        self.lossList = []
        self.algorithm(self.generation, learnRate)
        torch.save(self.net, 'matrixT_training.pkl')

    def algorithm(self, generations, lr):
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        loss_func = torch.nn.MSELoss()
        for g in range(generations):
            prediction = self.net(self.input)
            loss = loss_func(prediction, self.output)
            self.lossList.append(loss.data.numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X = range(self.generation)
        Y = self.lossList
        draw.draw_with_text(X, Y, str(lr), self.get_remark_list())

    def get_remark_list(self):
        remarkList = []
        for i in range(len(self.hidden)):
            remarkList.append('第{}隐层: {}个神经元'.format(i+1, self.hidden[i]))
        return remarkList

    def test_accuracy(self, x, y):
        inputE = torch.tensor(x, dtype=torch.float)
        outputE = torch.tensor(y, dtype=torch.float)
        predict = self.net(inputE)
        predict, outputE = predict.data.numpy(), outputE.data.numpy()
        print('------------')


class Net(torch.nn.Module):
    def __init__(self, n_input, list_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = list_hidden
        if not self.hidden:
            print('wrong hidden list')
        self.input = torch.nn.Linear(n_input, self.hidden[0])
        self.output = torch.nn.Linear(self.hidden[-1], n_output)
        if len(self.hidden) > 2:
            for i in range(len(self.hidden) - 2):


    def forward(self, X):
        k = ftn.relu(self.input(X))
        k = ftn.relu(self.hidden(k))
        k = self.output(k)
        return k


x = np.linspace(-1, 1, 100).reshape(100, 1)
y = x * x + 0.1 * np.random.rand(100, 1)
bp = BP(x, y, 200, 0.1, [3, 4])