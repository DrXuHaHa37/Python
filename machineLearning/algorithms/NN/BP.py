import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as ftn

# x1_data = np.linspace(-1, 1, 100)
# x2_data = np.linspace(-1, 1, 100)
# x_data = np.vstack((x2_data, x1_data)).T
# y_data = x_data[:, 0].__pow__(2) + x_data[:, -1].__pow__(2)


class BP:
    def __init__(self, x: np.array, y: np.array, generation):

        self.generation = generation
        self.input = torch.tensor(x, dtype=torch.float)
        self.input_groups, self.input_attrs = self.input.shape
        self.output = torch.tensor(y, dtype=torch.float)
        self.output_groups, self.output_attrs = self.output.shape[0], 1
        self.hidden_1, self.hidden_2 = 7, 8
        self.net = Net(self.input_attrs,
                       self.hidden_1,
                       self.hidden_2,
                       self.output_attrs
                       )
        self.lossList = []
        self.algorithm(self.generation)
        torch.save(self.net, 'matrixT_training.pkl')

    def algorithm(self, generations):
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.2)
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

    def test_accuracy(self, x: np.array, y: np.array):
        inputE = torch.tensor(x, dtype=torch.float)
        outputE = torch.tensor(y, dtype=torch.float)
        groupsE, featuresE = inputE.shape
        predict = self.net(inputE)
        predict, outputE = predict.data.numpy(), outputE.data.numpy()


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