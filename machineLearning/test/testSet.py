class a:
    def __init__(self, x):
        self.x = self.b(x).cal()
        print(self.x)

    class b:
        def __init__(self, i):
            self.input = i

        def cal(self):
            self.cla2()
            return self.input + 2

        def cla2(self):
            return self.input + 3


a(3)
