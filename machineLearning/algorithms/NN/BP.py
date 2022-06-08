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
        self.net = Net4(self.input_attrs,
                        self.hidden,
                        self.output_attrs
                        )
        print(self.net)
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
            remarkList.append('hidden layer {}: {}'.format(i+1, self.hidden[i]))
        return remarkList

    def test_accuracy(self, x, y):
        inputE = torch.tensor(x, dtype=torch.float)
        outputE = torch.tensor(y, dtype=torch.float)
        predict = self.net(inputE)
        predict, outputE = predict.data.numpy(), outputE.data.numpy()
        print('------------')


class Net1(torch.nn.Module):
    def __init__(self, n_input, list_hidden: list, n_output):
        super(Net1, self).__init__()
        self.hidden = []
        if not list_hidden:
            print('wrong hidden list')
        self.input = torch.nn.Linear(n_input, list_hidden[0])
        self.output = torch.nn.Linear(list_hidden[-1], n_output)

    def forward(self, X):
        k = ftn.relu(self.input(X))
        k = self.output(k)
        return k


class Net2(torch.nn.Module):
    def __init__(self, n_input, list_hidden: list, n_output):
        super(Net2, self).__init__()
        self.hidden = []
        if not list_hidden:
            print('wrong hidden list')
        self.input = torch.nn.Linear(n_input, list_hidden[0])
        self.hidden = torch.nn.Linear(list_hidden[0], list_hidden[1])
        self.output = torch.nn.Linear(list_hidden[-1], n_output)

    def forward(self, X):
        k = ftn.relu(self.input(X))
        k = ftn.relu(self.hidden(k))
        k = self.output(k)
        return k


class Net3(torch.nn.Module):
    def __init__(self, n_input, list_hidden: list, n_output):
        super(Net3, self).__init__()
        self.input = torch.nn.Linear(n_input, list_hidden[0])
        self.hidden1 = torch.nn.Linear(list_hidden[0], list_hidden[1])
        self.hidden2 = torch.nn.Linear(list_hidden[1], list_hidden[2])
        self.output = torch.nn.Linear(list_hidden[-1], n_output)

    def forward(self, X):
        k = ftn.relu(self.input(X))
        k = ftn.relu(self.hidden1(k))
        k = ftn.relu(self.hidden2(k))
        k = self.output(k)
        return k


class Net4(torch.nn.Module):
    def __init__(self, n_input, list_hidden: list, n_output):
        super(Net4, self).__init__()
        self.input = torch.nn.Linear(n_input, list_hidden[0])
        self.hidden1 = torch.nn.Linear(list_hidden[0], list_hidden[1])
        self.hidden2 = torch.nn.Linear(list_hidden[1], list_hidden[2])
        self.hidden3 = torch.nn.Linear(list_hidden[2], list_hidden[3])
        self.output = torch.nn.Linear(list_hidden[-1], n_output)

    def forward(self, X):
        k = ftn.relu(self.input(X))
        k = ftn.relu(self.hidden1(k))
        k = ftn.relu(self.hidden2(k))
        k = ftn.relu(self.hidden3(k))
        k = self.output(k)
        return k


# x_data = np.linspace(-1, 1, 100).reshape(100, 1)
# y_data = x_data * x_data * x_data + 0.01 * np.random.randn(100, 1)
#
# bp = BP(x_data, y_data, 100, 0.1, [3, 3])
