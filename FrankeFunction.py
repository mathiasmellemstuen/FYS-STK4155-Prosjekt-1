import numpy as np

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def FrankeFunctionNoised(x, y, max_noise): 

    ff = FrankeFunction(x, y)
    noise = np.zeros(ff.shape)
    for i in range(len(x)):
        for j in range(len(x)):
            noise[i][j] = np.random.normal(0, max_noise, 1)

    return ff + noise