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


def random_choice_m_in_n(n):
    # 总共n个 抽取m个
    m = 3 * n / 4
    nSequence = [x for x in range(n)]
    random.shuffle(nSequence)
    mSequence = []
    run = 0
    while run < m:
        mSequence.append(nSequence.pop())
        run += 1
    nSequence.sort()
    mSequence.sort()
    # m 是多的 n是少的
    return mSequence, nSequence
