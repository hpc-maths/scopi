import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # pip install -U scikit-learn

prefix = "../build/cholesky/slurm-"

def linearRegression(numThreads, totalTime):
    x = numThreads
    y = np.zeros(len(totalTime))
    for count, time in enumerate(totalTime):
        y[count] = totalTime[1] / time
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    reg = LinearRegression(normalize=True)
    reg.fit(x,y)
    return [reg.coef_, reg.intercept_]


def sumTime(nbTimeSteps, jobIdArray, beginLines, indexTime):
    totalTime = np.zeros(len(jobIdArray))
    for count, jobId in enumerate(jobIdArray):
        myFile = open(prefix+str(jobId)+'.out', 'r')
        Lines = myFile.readlines()
        time = np.zeros(nbTimeSteps+1)
        iter = 0
        for line in Lines:
            for k, beginLine in enumerate(beginLines):
                if line.startswith(beginLine):
                    time[iter] += float(line[indexTime[k]:-1])
                    if k == 0:
                        iter += 1
                if iter > nbTimeSteps:
                    break
        totalTime[count] = time.sum()
    return totalTime

def plotTimeSolve():
    numThreads = np.array([1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40])
    jobId = np.array([54304, 54303, 54302, 54301, 54300, 54493, 54492, 54491, 54299, 54786, 54785])
    time = sumTime(20, jobId, np.array(['----> CPUTIME : solve =']), np.array([24]))
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
    timeBruteForce = sumTime(20, jobIdBruteForce, np.array(['----> CPUTIME : compute ']), np.array([43]))

    jobIdKdTree = np.array([54304, 54303, 54302, 54301, 54494, 54300, 54493, 54492, 54491, 54299, 54786, 54785])
    timeKdTree = sumTime(20, jobIdKdTree, np.array(['----> CPUTIME : build ', '----> CPUTIME : compute ']), np.array([37, 43]))

    plt.plot(numThreads, timeBruteForce[0]/timeBruteForce, 'x', label='brute force')
    plt.plot(numThreads, timeKdTree[0]/timeKdTree, 'x', label='kd tree')
    plt.xlabel("Number of threads")
    plt.ylabel("Speed-up")
    plt.legend()
    plt.show()


def plotTimeUzawa():
    numThreads = np.linspace(1, 20, 20)
    jobId = np.array([55115, 55114, 55113, 55112, 55111, 55110, 55109, 55108, 55107, 55106, 55105, 55104, 55103, 55102, 55101, 55100, 55099, 55098, 55097, 55096])

    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (U = c)']), np.array([32]))
    mkl1, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $U = c$")
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (U = A^T*L+U)']), np.array([37]))
    mkl2, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $U = A^T L + U$")
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (U = -P^-1*U)']), np.array([38]))
    mkl3, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $U =  - P^{-1} U$")
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (R = d)']), np.array([32]))
    mkl4, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $R = D$")
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (R = -A*U+R)']), np.array([38]))
    mkl5, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $R = -AU+R$")
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (L = max(L-rho*R, 0))']), np.array([46]))
    mkl6, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $L = \max(L - rho R, 0)$")
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (cmax = min(R))']), np.array([40]))
    mkl7, = plt.plot(numThreads, time[0]/time, 'o', label = r"MKL, $cmax = \min(R)$")

    jobId = np.array([55316, 55315, 55314, 55313, 55312, 55311, 55310, 55309, 55308, 55307, 55306, 55305, 55304, 55303, 55302, 55301, 55300, 55299, 55298, 55297])

    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (U = c)']), np.array([32]))
    plt.plot(numThreads, time[0]/time, 'x', label = r"Matrix-free OpenMP, $U = c$", color=mkl1.get_color())
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (U = A^T*L+U)']), np.array([37]))
    plt.plot(numThreads, time[0]/time, 'x', label = r"Matrix-free OpenMP, $U = A^T L + U$", color=mkl2.get_color())
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (U = -P^-1*U)']), np.array([38]))
    plt.plot(numThreads, time[0]/time, 'x', label = r"Matrix-free OpenMP, $U =  - P^{-1} U$", color=mkl3.get_color())
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (R = d)']), np.array([32]))
    plt.plot(numThreads, time[0]/time, 'x', label = r"Matrix-free OpenMP, $R = D$", color=mkl4.get_color())
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (R = -A*U+R)']), np.array([38]))
    plt.plot(numThreads, time[0]/time, 'x', label = r"Matrix-free OpenMP, $R = -AU+R$", color=mkl5.get_color())
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (L = max(L-rho*R, 0))']), np.array([46]))
    plt.plot(numThreads, time[0]/time, 'x', label = r"Matrix-free OpenMP, $L = \max(L - rho R, 0)$", color=mkl6.get_color())
    time = sumTime(100, jobId, np.array(['----> CPUTIME : solve (cmax = min(R))']), np.array([40]))
    plt.plot(numThreads, time[0]/time, 'x', label = r"Matrix-free OpenMP, $cmax = \min(R)$", color=mkl7.get_color())

    plt.xlabel("Number of threads")
    plt.ylabel("Speed-up")
    plt.legend()
    plt.show()

plotTimeUzawa()


