import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as ftn


class BP:
    def __init__(self, x: np.array, y: np.array, generation, learnRate, hidden):

        self.generation = generation
        self.input = torch.tensor(x, dtype=torch.float)
        self.input_groups, self.input_attrs = self.input.shape
        self.output = torch.tensor(y, dtype=torch.float)
        self.output_groups, self.output_attrs = self.output.shape[0], 1
        self.hidden_1, self.hidden_2 = hidden
        self.net = Net(self.input_attrs,
                       self.hidden_1,
                       self.hidden_2,
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
        self.draw()

    def draw(self):
        x = range(self.generation)
        y = self.lossList
        plt.plot(np.array(x), np.array(y))
        plt.show()

    def test_accuracy(self, x, y):
        inputE = torch.tensor(x, dtype=torch.float)
        outputE = torch.tensor(y, dtype=torch.float)
        predict = self.net(inputE)
        predict, outputE = predict.data.numpy(), outputE.data.numpy()
        print('------------')


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(n_input, n_hidden_1)
        self.hidden = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.output = torch.nn.Linear(n_hidden_2, n_output)

    def forward(self, x):
        x = ftn.relu(self.input(x))
        x = ftn.relu(self.hidden(x))
        x = self.output(x)
        return x


x = np.linspace(-1, 1, 100).reshape(100, 1)
y = x * x + 0.1 * np.random.rand(100, 1)
bp = BP(x, y, 200, 0.1, [3, 4])