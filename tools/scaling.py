import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # pip install -U scikit-learn

prefix = "../build/cholesky/slurm-"

def linearRegression(numThreads, totalTime):
    x = numThreads
    y = np.zeros(len(totalTime))
    for count, time in enumerate(totalTime):
        y[count] = totalTime[0] / time
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    reg = LinearRegression(normalize=True)
    reg.fit(x,y)
    return [reg.coef_, reg.intercept_]

def getTimeFromLine(line):
    for token in line.split():
        try:
            if float(token) != -1:
                return float(token)
        except:
            pass
    print("error in line: ", line)
    return 0.


def sumTime(nbTimeSteps, jobIdArray, beginLines):
    totalTime = np.zeros(len(jobIdArray))
    for count, jobId in enumerate(jobIdArray):
        myFile = open(prefix+str(jobId)+'.out', 'r')
        Lines = myFile.readlines()
        time = np.zeros(nbTimeSteps+1)
        iter = 0
        for line in Lines:
            for k, beginLine in enumerate(beginLines):
                if line.startswith(beginLine):
                    time[iter] += getTimeFromLine(line)
                    if k == 0:
                        iter += 1
                if iter > nbTimeSteps:
                    break
        totalTime[count] = time.sum()
    return totalTime

def plotTimeSolve():
    numThreads = np.array([1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40])
    jobId = np.array([54304, 54303, 54302, 54301, 54300, 54493, 54492, 54491, 54299, 54786, 54785])
    time = sumTime(20, jobId, np.array(['----> CPUTIME : solve =']))
    a, b = linearRegression(numThreads, time)
    plt.plot(numThreads, time[0]/time, 'x')
    plt.plot(numThreads, (a*numThreads+b)[0,:], label="y = "+str(a[0,0])+"x+"+str(b[0]))
    plt.xlabel("Number of threads")
    plt.ylabel("Speed-up")
    plt.legend()
    plt.show()

def plotTimeContacts():
    numThreads = np.array([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])

    jobIdBruteForce = np.array([54773, 54774, 54775, 54776, 54777, 54778, 54769, 54779, 54781, 54782, 54783, 54784])
    timeBruteForce = sumTime(20, jobIdBruteForce, np.array(['----> CPUTIME : compute ']))

    jobIdKdTree = np.array([54304, 54303, 54302, 54301, 54494, 54300, 54493, 54492, 54491, 54299, 54786, 54785])
    timeKdTree = sumTime(20, jobIdKdTree, np.array(['----> CPUTIME : build ', '----> CPUTIME : compute ']))

    plt.plot(numThreads, timeBruteForce[0]/timeBruteForce, 'x', label='brute force')
    plt.plot(numThreads, timeKdTree[0]/timeKdTree, 'x', label='kd tree')
    plt.xlabel("Number of threads")
    plt.ylabel("Speed-up")
    plt.legend()
    plt.show()


def plotTimeUzawa():
    numThreads = np.linspace(1, 20, 20)
    nbIter = 100

    jobId = np.linspace(56887, 56906, 20, dtype=int)
    time = sumTime(nbIter, jobId, np.array(['----> CPUTIME : solve (U = A^T*L+U)']))
    mkl1, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $U = A^T L + U$")
    time = sumTime(nbIter, jobId, np.array(['----> CPUTIME : solve (U = -P^-1*U)']))
    mkl2, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $U =  - P^{-1} U$")
    time = sumTime(nbIter, jobId, np.array(['----> CPUTIME : solve (R = -A*U+R)']))
    mkl3, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $R = -AU+R$")

    plt.plot(numThreads, numThreads, color='k', label="y = x")

    plt.xlabel("Number of threads")
    plt.ylabel("Speed-up")
    plt.ylim([0, 10])
    plt.legend()
    plt.show()

plotTimeUzawa()


