class Algo:
    def __init__(self, iters, groups):
        self.iter = iters
        self.group = groups


class PSO(Algo):
    def __init__(self, w, c, algo: Algo):
        self.iter = algo.iter
        self.group = algo.group
        self.w = w
        self.c = c

    def algorithm(self):
        pass


a = Algo(3, 8)

pso = PSO(1, 2, a)

print('---')
