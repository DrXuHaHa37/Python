import numpy as np
import random


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


def random_choice_m_in_n(m, n):
    nSequence = [x for x in range(n)]
    random.shuffle(nSequence)
    mSequence = []
    run = 0
    while run < m:
        mSequence.append(nSequence.pop())
        run += 1
    nSequence.sort()
    mSequence.sort()
    return mSequence, nSequence
