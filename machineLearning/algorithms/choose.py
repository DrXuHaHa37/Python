# 轮盘赌
def roulette(dic):
    probability = []
    total = sum(dic.values())
    accumulation = 0
    for v in dic.values():
        accumulation += v
        probability.append(accumulation/total)
    r = np.random.random()
    for i in range(len(probability)):
        if r < probability[i]:
            return list(enumerate(dic))[i][1]