import os
import numpy as np
from opfunu.cec_based import cec2013
import math
from pyDOE2 import lhs


POPULATION_SIZE = 100                                                  # the number of individuals (POPULATION_SIZE > 4)
DIMENSION_NUM = 10                                                    # the number of variables
LOWER_BOUNDARY = -100                                                 # the maximum value of the variable range
UPPER_BOUNDARY = 100                                                  # the minimum value of the variable range
REPETITION_NUM = 30                                                   # the number of independent runs
MAX_FITNESS_EVALUATION_NUM = DIMENSION_NUM * 500                      # the maximum number of fitness evaluations
MAX_ITERATION = int(MAX_FITNESS_EVALUATION_NUM / POPULATION_SIZE)

Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
Population_fitness = np.zeros(POPULATION_SIZE)

Offspring = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
Offspring_fitness = np.zeros(POPULATION_SIZE)
varepsilon = 0.00000001
Fun_num = 1

ScoreMatrix = np.zeros((4, MAX_ITERATION))



def SMclear():
    global ScoreMatrix
    for i in range(len(ScoreMatrix)):
        for j in range(len(ScoreMatrix[i])):
            ScoreMatrix[i][j] = varepsilon


def scales(data):
    data = np.array(data)
    limit_scale = []
    for i in range(len(data[0])):
        d = data[:, i]
        limit_scale.append([min(d), max(d)])
    return limit_scale



def CheckIndi(Indi):
    range_width = UPPER_BOUNDARY - LOWER_BOUNDARY
    for i in range(DIMENSION_NUM):
        if Indi[i] > UPPER_BOUNDARY:
            n = int((Indi[i] - UPPER_BOUNDARY) / range_width)
            mirrorRange = (Indi[i] - UPPER_BOUNDARY) - (n * range_width)
            Indi[i] = UPPER_BOUNDARY - mirrorRange
        elif Indi[i] < LOWER_BOUNDARY:
            n = int((LOWER_BOUNDARY - Indi[i]) / range_width)
            mirrorRange = (LOWER_BOUNDARY - Indi[i]) - (n * range_width)
            Indi[i] = LOWER_BOUNDARY + mirrorRange
        else:
            pass


def Initialization(Func):
    global Population, Population_fitness, DIMENSION_NUM, POPULATION_SIZE
    for i in range(POPULATION_SIZE):
        for j in range(DIMENSION_NUM):
            Population[i][j] = np.random.uniform(LOWER_BOUNDARY, UPPER_BOUNDARY)
        Population_fitness[i] = Func(Population[i])


def normal(Fits):
    w = np.zeros(len(Fits))
    sums = sum(Fits)
    for i in range(len(Fits)):
        w[i] = Fits[i] / sums
    return w


def Search(Func, i, t):
    global Population, Population_fitness, Offspring, Offspring_fitness
    x_best = Population[np.argmin(Population_fitness)]

    r1, r2 = np.random.randint(0, POPULATION_SIZE), np.random.randint(0, POPULATION_SIZE)
    while r1 == r2:
        r2 = np.random.randint(0, POPULATION_SIZE)
    vc = 1 - (t / MAX_ITERATION)
    a = math.atanh(vc)
    if np.random.rand() < 0.5:
        Offspring[i] = Population[i] + (np.random.uniform(-a, a) + 0.1) * (x_best - Population[i]) + (
                    np.random.uniform(-a, a) + 0.1) * (Population[r1] - Population[r2])
    else:
        Offspring[i] = x_best + (np.random.uniform(-a, a) + 0.1) * (Population[r1] - Population[r2])

    CheckIndi(Offspring[i])
    Offspring_fitness[i] = Func(Offspring[i])


def Approach(Func, i, t):
    global Population, Population_fitness, Offspring, Offspring_fitness
    X_mean = np.zeros(DIMENSION_NUM)
    idx = np.argsort(Population_fitness)
    size = np.random.randint(1, int(POPULATION_SIZE / 2))
    for j in range(size):
        X_mean += Population[idx[j]]
    X_mean /= size
    for j in range(DIMENSION_NUM):
        Offspring[i][j] = X_mean[j] + 4 * np.random.rand() - 2
    CheckIndi(Offspring[i])
    Offspring_fitness[i] = Func(Offspring[i])


def weights(Fits):
    w = np.zeros(len(Fits))
    sums = 0
    for i in range(len(Fits)):
        sums += 1 / Fits[i]
    for i in range(len(Fits)):
        w[i] = (1 / Fits[i]) / sums
    return w


def Wrap(Func, i, t, k=10):
    global Population, Population_fitness, Offspring, Offspring_fitness
    samples = list(range(POPULATION_SIZE))
    samples.remove(i)
    samples = np.random.permutation(samples)[:k]
    subPop = Population[samples]
    subFit = Population_fitness[samples]

    W = weights(subFit)
    Off = np.zeros(DIMENSION_NUM)
    for j in range(len(W)):
        Off += W[j] * subPop[j]
    Offspring[i] = Off

    CheckIndi(Offspring[i])
    Offspring_fitness[i] = Func(Offspring[i])


def Mutate(Func, i, t):
    global Offspring, Offspring_fitness
    for j in range(DIMENSION_NUM):
        Offspring[i][j] = np.random.uniform(LOWER_BOUNDARY, UPPER_BOUNDARY)
    Offspring_fitness[i] = Func(Offspring[i])


def Select():
    global Population, Population_fitness, Offspring, Offspring_fitness
    for i in range(POPULATION_SIZE):
        if Offspring_fitness[i] < Population_fitness[i]:
            Population[i] = Offspring[i]
            Population_fitness[i] = Offspring_fitness[i]


def SequenceConstruct(t, size):
    global ScoreMatrix, Fun_num, DIMENSION_NUM
    sequence = np.zeros(POPULATION_SIZE)
    subscore = np.zeros(size)
    for i in range(size):
        for j in range(t):
            subscore[i] += ScoreMatrix[i][j]
    w = normal(subscore)
    for i in range(1, len(w)):
        w[i] += w[i-1]

    for i in range(POPULATION_SIZE):
        r = np.random.uniform(0, 1)
        for j in range(len(w)):
            if r < w[j]:
                sequence[i] = j
                break
    return sequence


def ScoreUpdate(sequence, size, t):
    global Population, Population_fitness, Offspring, Offspring_fitness, ScoreMatrix
    times = np.zeros(size)
    improve_time = np.zeros(size)
    improve_amount = np.zeros(size)
    base = varepsilon
    for i in range(len(sequence)):
        base += max((Population_fitness[i] - Offspring_fitness[i]), 0)
    for s in sequence:
        times[int(s)] += 1
    for i in range(len(sequence)):
        if Offspring_fitness[i] < Population_fitness[i]:
            improve_time[int(sequence[i])] += 1
            improve_amount[int(sequence[i])] += max((Population_fitness[i] - Offspring_fitness[i]), 0) / base
    for i in range(size):
        improve_time[i] = improve_time[i] / (times[i] + varepsilon)
    sum_t = sum(improve_time)
    for i in range(size):
        improve_time[i] /= (sum_t + varepsilon)
    for i in range(size):
        ScoreMatrix[i][t] = 0.5 * improve_time[i] + 0.5 * improve_amount[i]
    return


def EMSMO(Func, t):
    global Population, Population_fitness, Offspring, Offspring_fitness, ScoreMatrix
    archive = [Search, Approach, Wrap, Mutate]
    size = len(archive)
    sequence = SequenceConstruct(t, size)

    for i in range(POPULATION_SIZE):
        archive[int(sequence[i])](Func, i, t)
    ScoreUpdate(sequence, size, t)
    Select()


def RunEMSMO(Func):
    global MAX_ITERATION, Fun_num, Population_fitness, ScoreMatrix
    All_Trial_Best = []
    for i in range(REPETITION_NUM):                 # run the algorithm independently multiple times
        Best_list = []
        iteration = 1
        np.random.seed(2022 + 88*i)                 # fix the seed of random number
        Initialization(Func)                            # randomly initialize the population
        Best_list.append(min(Population_fitness))
        SMclear()
        while iteration <= MAX_ITERATION:
            EMSMO(Func, iteration)
            iteration += 1
            Best_list.append(min(min(Population_fitness), Best_list[-1]))
        All_Trial_Best.append(Best_list)
    np.savetxt('./EMSMO_Data/CEC2013/F{}_{}D.csv'.format(Fun_num, DIMENSION_NUM), All_Trial_Best, delimiter=",")


def main(Dim):
    global Fun_num, DIMENSION_NUM, MAX_FITNESS_EVALUATION_NUM, MAX_ITERATION, Population, Offspring, ScoreMatrix
    DIMENSION_NUM = Dim
    MAX_FITNESS_EVALUATION_NUM = DIMENSION_NUM * 500
    MAX_ITERATION = int(MAX_FITNESS_EVALUATION_NUM / POPULATION_SIZE)
    ScoreMatrix = np.zeros((4, MAX_ITERATION+1))
    Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
    Offspring = np.zeros((POPULATION_SIZE, DIMENSION_NUM))

    CEC2013Funcs = [cec2013.F12013(Dim), cec2013.F22013(Dim), cec2013.F32013(Dim), cec2013.F42013(Dim),
                    cec2013.F52013(Dim), cec2013.F62013(Dim), cec2013.F72013(Dim), cec2013.F82013(Dim),
                    cec2013.F92013(Dim), cec2013.F102013(Dim), cec2013.F112013(Dim), cec2013.F122013(Dim),
                    cec2013.F132013(Dim), cec2013.F142013(Dim), cec2013.F152013(Dim), cec2013.F162013(Dim),
                    cec2013.F172013(Dim), cec2013.F182013(Dim), cec2013.F192013(Dim), cec2013.F202013(Dim),
                    cec2013.F212013(Dim), cec2013.F222013(Dim), cec2013.F232013(Dim), cec2013.F242013(Dim),
                    cec2013.F252013(Dim), cec2013.F262013(Dim), cec2013.F272013(Dim), cec2013.F282013(Dim)]

    for i in range(20, len(CEC2013Funcs)):
        Fun_num = i + 1
        RunEMSMO(CEC2013Funcs[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('./EMSMO_Data/CEC2013') == False:
        os.makedirs('./EMSMO_Data/CEC2013')
    Dims = [50]
    for Dim in Dims:
        main(Dim)
