import numpy as np

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def FrankeFunctionNoised(x, y, max_noise): 

    ff = FrankeFunction(x, y)
    noise = np.random.normal(0, max_noise, len(x)*len(x))
    noise = noise.reshape(len(x), len(x))
    return ff + noise



def make_X(x,y,n = 5): 
    x = x.ravel()
    y = y.ravel()
    if n == 1:
        X = np.c_[np.ones(len(x)),		  
                            x,y]
    if n == 2:
        X = np.c_[np.ones(len(x)),		  
                            x,y,
                            x**2,x*y,y**2]
    if n == 3:
        X = np.c_[np.ones(len(x)),		  
                            x,y,
                            x**2,x*y,y**2,
                            x**3,x**2*y,x*y**2,y**3]
     
    if n == 4:
        X = np.c_[np.ones(len(x)),		  
                            x,y,
                            x**2,x*y,y**2,
                            x**3,x**2*y,x*y**2,y**3,
                            x**4,x**3*y,x**2*y**2,x*y**3,y**4]

    if n == 5:
        X = np.c_[np.ones(len(x)),		  
                            x,y,
                            x**2,x*y,y**2,
                            x**3,x**2*y,x*y**2,y**3,
                            x**4,x**3*y,x**2*y**2,x*y**3,y**4,
                            x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5]
        
    return X

#automated make X
def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def calc_beta(X,y):
    #beta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    #X = X + 0.0000001
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta



#z_noise = FrankeFunctionNoised(x,y, 0.001)

def MSE(y,y_tilde):
    sum = 0
    n = len(y)
    for i in range(n):
        sum += (y[i] - y_tilde[i])**2
    return sum/n

    


# Plot the surface.
def plot_surface(z, z_tilde):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    z_tilde2D = np.reshape(z_tilde, (20,20))
    surf = ax.plot_surface(x, y, z_tilde2D,
                linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
    # Customize the z axis.
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def R2score(y,y_tilde):
    y_mean = np.mean(y)
    mse = MSE(y,y_tilde)
    sum = 0
    n = len(y)
    for i in range(n):
            sum += (y[i] - y_mean)**2
    sum /= n 
    return 1 - mse/sum


def plot_MSE_R2_beta(x,y):
    mse_arr_train = np.zeros(5)
    mse_arr_test = np.zeros(5)
    R2_arr_train = np.zeros(5)
    R2_arr_test = np.zeros(5)
    z = FrankeFunctionNoised(x,y, 0.001)
    z = z.ravel()
    for n in range(1,6):
        X = make_X(x,y,n)    
        X_train, X_test, z_train, z_test= train_test_split(X, z, test_size=0.2)
        #Scaling
        X_train = (X_train - np.mean(X_train))/np.std(X_train)
        X_test = (X_test - np.mean(X_test))/np.std(X_test)
        z_train = (z_train - np.mean(z_train))/np.std(z_train)
        z_test = (z_test - np.mean(z_test))/np.std(z_test)
        
        beta = calc_beta(X_train,z_train)
        z_tilde_test = X_test @ beta
        z_tilde_train = X_train @ beta

        mse_arr_train[n-1] = MSE(z_train, z_tilde_train)
        mse_arr_test[n-1] = MSE(z_test, z_tilde_test)
        R2_arr_train[n-1] = R2score(z_train, z_tilde_train)
        R2_arr_test[n-1] = R2score(z_test, z_tilde_test)

    n_arr = np.linspace(1,5,5)
    fig, axs = plt.subplots(2)
    axs[0].plot(n_arr, mse_arr_train, label= "MSE_train")
    axs[0].plot(n_arr, mse_arr_test, label= "MSE_test")
    axs[0].set_xlabel("Polynomial degree")
    axs[0].set_ylabel("MSE")
    axs[0].legend()

    axs[1].plot(n_arr, R2_arr_train, label= "R2_train")
    axs[1].plot(n_arr, R2_arr_test, label= "R2_test")
    axs[1].set_xlabel("Polynomial degree")
    axs[1].set_ylabel("R2")
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def plot_MSE_variance(x,y):
    num = 13
    val = 0.1
    mse_arr_train = np.zeros(num-1)
    mse_arr_test = np.zeros(num-1)
    for i in range(1,num):
        z = FrankeFunctionNoised(x,y, val)
        z = z.ravel()
        X = create_X(x,y, i)
        X_train, X_test, z_train, z_test= train_test_split(X, z, test_size=0.2)
        #Scaling
        #X_train = (X_train - np.mean(X_train))/np.std(X_train)
        #X_test = (X_test - np.mean(X_test))/np.std(X_test)
        #z_train = (z_train - np.mean(z_train))/np.std(z_train)
        #z_test = (z_test - np.mean(z_test))/np.std(z_test)
        
        beta = calc_beta(X_train,z_train)
        z_tilde_test = X_test @ beta
        z_tilde_train = X_train @ beta

        mse_arr_train[i-1] = MSE(z_train, z_tilde_train)
        mse_arr_test[i-1] = MSE(z_test, z_tilde_test)

    n_arr = np.array([i for i in range(1,num)])
    plt.plot(n_arr, mse_arr_train, label= "MSE_train")
    plt.plot(n_arr, mse_arr_test, label= "MSE_test")
    plt.xlabel("Polynomials")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("MSE_test_train.png")
    plt.show()

if __name__ == "__main__":
    
    np.random.seed(10)

    X = make_X(x,y,5)
    z = FrankeFunctionNoised(x,y, 0.01)
    
    #Splitting data
    z_flat = z.ravel()
    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

    print(X_train.shape, X_test.shape)
    
    #Scaling
    X_train = (X_train - np.mean(X_train))/np.std(X_train)
    X_test = (X_test - np.mean(X_test))/np.std(X_test)

    z_train = (z_train - np.mean(z_train))/np.std(z_train)
    z_test = (z_test - np.mean(z_test))/np.std(z_test)


    beta = calc_beta(X,z_flat)
    z_tilde = X @ beta
    #scaler = sk.StandardScaler()

    #plot_MSE_R2_beta(x,y)
    #plot_surface(z, z_tilde)
    plot_MSE_variance(x,y)

