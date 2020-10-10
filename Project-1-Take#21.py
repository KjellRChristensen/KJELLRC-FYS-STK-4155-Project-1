#from TestPythonCode import KFold, KFoldBootstrap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.utils import resample
import numpy as np
import pandas as pd
import sys, os
from random import random, seed
from imageio import imread
from numpy.random import normal, uniform


#from numba import njit
#
fig = plt.figure()
#ax = fig.gca(projection='3d')


# Disable print(x) statements
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore 
def enablePrint():
    sys.stdout = sys.__stdout__

#blockPrint()
#enablePrint()

# Definition of the Franke's function as in the text.
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Define DesingMatrix, to run with Numba and in parallel, if posible
#@njit(parallel=True)
def DesignMatrix(x,y,n):
        if len(x.shape) > 1:
                x = np.ravel(x)
                y = np.ravel(y)
        N = len(x)
        l = int((n+1)*(n+2)/2)          # Number of elements/columns in beta                                                               
        X = np.ones((N,l))

        for i in range(1,n+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,q+k] = (x**(i-k))*(y**k)
        return X

#Add noise to Z
def AddNoise(z,n):
    return z + (0.005*np.random.randn(n))

#Find beta, by using matrix invers functionality
#For Franke's we are passing in Z for the "y" vector.
def BetaMatrixInv(X,y):
    Beta= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return (X @ Beta)

def ConfidenceIntervals(VarPredict,MeanPredict,n):
    # (80% =Z=1.282), (85%=Z=1.440),(90%=Z=1.645),
    # (95% =Z=1.960), (99%=Z=2.576),(99.5%=Z=2.807 (99.9%=Z=3.291)
    # We use confidence interval 99.5%
    z=2.807
    print("99.5% Confidence interval")
    print("Mean +- Z*(StandardDev /SQR(n)")
    print("Say for 99.5% we have")        
    print("Mean +- 2.807*(SQR(VarPredict)/SQR(n))")
    print(MeanPredict-(z*np.sqrt(VarPredict)/np.sqrt(n)))
    print(MeanPredict+(z*np.sqrt(VarPredict)/np.sqrt(n)))
    #Define Mean Square Error (OLS)
    return

def MSE(YData,YPredict):
    DimY=len(YData)
    MSE= (YData -YPredict).T@(YData - YPredict)/DimY
    return MSE

def GetVariance(YPredict):
    return np.sum((YPredict - np.mean(YPredict))**2)/np.size(YPredict)

def GetMean(YPredict):
    return  np.mean(YPredict, axis=0)

def SubMean(YPredict):
    YPredict = YPredict - np.mean(YPredict, axis=0)
    return YPredict

def GetR2 (YData,YPredict):
    YDataMean=np.mean(YData)    
    return 1-((YData-YPredict).T@(YData-YPredict))/((YData-YDataMean).T@(YData-YDataMean))

def SplitDataSet(x_,y_,z_,i):
    #Quick way to delete and extract 
    #elements from list is by using np
    #delete & take
    #Try np.setdiff1d(a,b)
    x_learn=np.delete(x_,i)
    y_learn=np.delete(y_,i)
    z_learn=np.delete(z_,i)
    x_test=np.take(x_,i)
    y_test=np.take(y_,i)
    z_test=np.take(z_,i)
    return (x_learn,y_learn,z_learn,x_test,y_test,z_test)

def BootStrap(X,Degrees,Lamdas,Bootstraps,x,y,z,Samples,ReggrType):
    # Bootstrap also used for OLS, Ridge & Lasse and need a "switch (if statement)"
    # -Select # of bootstraps with replacement.
    # We need to switch on Bootstrap-values of Lamdas
    # x_train, x_test, y_train, y_test, z_train, z_test = SplitDataSet(x, y, z, test_size=0.2, shuffle=True)
    # Need modificatino to shuffle
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, shuffle=True)
    
    #Dette må parameteriseres
    
    s = (x_test.shape[0],Bootstraps)
    z_test1 = np.zeros(s)
    s = (x_train.shape[0],Bootstraps)
    z_train1 = np.zeros(s)

    for i in range(Bootstraps):
        z_test1[:,i]=z_test

    if (ReggrType=="OLS"):
        DegreeOrLamda=Degrees
    else:
        #Need to extract # of elements in the set
        DegreeOrLamda=len(Lamdas)

    #Setting up the stuct based on # of Degrees and
    #length of lampda elements passed in.
    error_test = np.zeros(DegreeOrLamda)
    bias___ = np.zeros(DegreeOrLamda)
    variance___ = np.zeros(DegreeOrLamda)
    polylamda = np.zeros(DegreeOrLamda)
    error_train = np.zeros(DegreeOrLamda)     
    
    for CurrDegreeOrLamda in range(DegreeOrLamda):
        z_pred = np.empty((z_test.shape[0],Bootstraps))
        z_pred_train = np.empty((z_train.shape[0],Bootstraps))
        
        # If Regression is not OLS, then we need
        # to preserve en loop through the values of the
        # lamdas - not the index
        # 
        if (ReggrType!="OLS"):
            CurrLamda=Lamdas[CurrDegreeOrLamda]
      
        for i in range(Bootstraps):
            xResample, yResample, zResample = resample(x_train, y_train, z_train)
            z_test1[:,i]=z_test
            z_train1[:,i] = zResample
           
            XTrain = DesignMatrix(xResample,yResample,CurrDegreeOrLamda)
            XTest= DesignMatrix(x_test,y_test,CurrDegreeOrLamda)

            if (ReggrType == "OLS"):
                print("Bootstrap based on OLS")
                #Beta=linear_model.LinearRegression(fit_intercept=False).fit(X,z)
                z_pred[:, i] = linear_model.LinearRegression(fit_intercept=False).fit(XTrain,zResample).predict(XTest).ravel()
                z_pred_train[:, i] = linear_model.LinearRegression(fit_intercept=False).fit(XTrain,zResample).predict(XTrain).ravel()
            elif (ReggrType == "Ridge"):
                print("Bootstrap based on Ridge")
                #Beta=linear_model.Ridge(alpha=Lamdas)
                z_pred[:, i] = linear_model.Ridge(alpha=CurrLamda).fit(XTrain,zResample).predict(XTest).ravel()
                z_pred_train[:, i] = linear_model.Ridge(alpha=CurrLamda).fit(XTrain,zResample).predict(XTrain).ravel()
            else:
                print("Bootstrap based on Lasso")
                #Beta=linear_model.Lasso(alpha=Lamdas)
                z_pred[:, i] = linear_model.Lasso(alpha=CurrLamda).fit(XTrain, zResample).predict(XTest).ravel()
                z_pred_train[:, i] = linear_model.Lasso(alpha=CurrLamda).fit(XTrain, zResample).predict(XTrain).ravel()
        
        if (ReggrType=="OLS"):
            polylamda[CurrDegreeOrLamda]    = CurrDegreeOrLamda
            error_test[CurrDegreeOrLamda]   = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
            bias___[CurrDegreeOrLamda]      = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance___[CurrDegreeOrLamda]  = np.mean( np.var(z_pred, axis=1, keepdims=True))
            error_train[CurrDegreeOrLamda]  = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        else:
            polylamda[Lamdas.index(CurrLamda)]   = CurrLamda
            error_test[Lamdas.index(CurrLamda)]  = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
            bias___[Lamdas.index(CurrLamda)]     = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance___[Lamdas.index(CurrLamda)] = np.mean( np.var(z_pred, axis=1, keepdims=True))
            error_train[Lamdas.index(CurrLamda)] = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        
        print(CurrDegreeOrLamda)
        print(error_test)
        print(bias___)
        print(variance___)
        print(bias___+variance___)
    return (polylamda,error_train,error_test, bias___,variance___ )

def KFold(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType):
    # K-Fold used for OLS,Ridge & Lasse and need a "switch (if statement)"
    # to select the specific model configuration.
    # k,x,y,z,m,model
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, shuffle=True)
    
    j=np.arange(Samples)
    np.random.shuffle(j)
    n_k=int(Samples/Folds)
    

    #Dette må parameteriseres
    
    s = (x_test.shape[0],Folds)
    z_test1 = np.zeros(s)
    s = (x_train.shape[0],Folds)
    z_train1 = np.zeros(s)

    for i in range(Folds):
        z_test1[:,i]=z_test

    if (ReggrType=="OLS"):
        DegreeOrLamda=Degrees
    else:
        #Need to extract # of elements in the set
        DegreeOrLamda=len(Lamdas)

    #Setting up the stuct based on # of Degrees and
    #length of lampda elements passed in.
    error_test = np.zeros(DegreeOrLamda)
    bias___ = np.zeros(DegreeOrLamda)
    variance___ = np.zeros(DegreeOrLamda)
    polylamda = np.zeros(DegreeOrLamda)
    error_train = np.zeros(DegreeOrLamda)     
    
    for CurrDegreeOrLamda in range(DegreeOrLamda):
        z_pred = np.empty((z_test.shape[0],Folds))
        z_pred_train = np.empty((z_train.shape[0],Folds))
        
        # If Regression is not OLS, then we need
        # to preserve en loop through the values of the
        # lamdas - not the index
        # 
        if (ReggrType!="OLS"):
            CurrLamda=Lamdas[CurrDegreeOrLamda]

        # Change Bottstraps with Folds
        for Fold in range(Folds):
            i=Fold
            # xResample,yResample,zResample,xTest,yTest,zTest=train_test_split(x, y, z, test_size=0.2, shuffle=True)
            xResample,yResample,zResample,xTest,yTest,zTest=SplitDataSet(x, y, z,j[i*n_k:(i+1)*n_k])
            
            z_test1[:,Fold]=zTest
            z_train1[:,Fold] = zResample
            
            XTrain = DesignMatrix(xResample,yResample,CurrDegreeOrLamda)
            XTest= DesignMatrix(x_test,y_test,CurrDegreeOrLamda)

            if (ReggrType == "OLS"):
                print("K-Fold based on OLS")
                Betas=linear_model.LinearRegression(fit_intercept=False).fit(X,z)
                z_pred[:, Fold] = linear_model.LinearRegression(fit_intercept=False).fit(XTrain,zResample).predict(XTest).ravel()
                z_pred_train[:, Fold] = linear_model.LinearRegression(fit_intercept=False).fit(XTrain,zResample).predict(XTrain).ravel()
            elif (ReggrType == "Ridge"):
                print("K-Fold based on Ridge")
                Betas=linear_model.Ridge(alpha=Lamdas)
                z_pred[:, Fold] = linear_model.Ridge(alpha=CurrLamda).fit(XTrain,zResample).predict(XTest).ravel()
                z_pred_train[:, Fold] = linear_model.Ridge(alpha=CurrLamda).fit(XTrain,zResample).predict(XTrain).ravel()
            else:
                print("K-Fold based on Lasso")
                Betas=linear_model.Lasso(alpha=Lamdas)
                z_pred[:, Fold] = linear_model.Lasso(alpha=CurrLamda).fit(XTrain, zResample).predict(XTest).ravel()
                z_pred_train[:, Fold] = linear_model.Lasso(alpha=CurrLamda).fit(XTrain, zResample).predict(XTrain).ravel()
                
        if (ReggrType=="OLS"):
            polylamda[CurrDegreeOrLamda]    = CurrDegreeOrLamda
            error_test[CurrDegreeOrLamda]   = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
            bias___[CurrDegreeOrLamda]      = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance___[CurrDegreeOrLamda]  = np.mean( np.var(z_pred, axis=1, keepdims=True))
            error_train[CurrDegreeOrLamda]  = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        else:
            polylamda[Lamdas.index(CurrLamda)]   = CurrLamda
            error_test[Lamdas.index(CurrLamda)]  = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
            bias___[Lamdas.index(CurrLamda)]     = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance___[Lamdas.index(CurrLamda)] = np.mean( np.var(z_pred, axis=1, keepdims=True))
            error_train[Lamdas.index(CurrLamda)] = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        
        print(CurrDegreeOrLamda)
        print(error_test)
        print(bias___)
        print(variance___)
        print(bias___+variance___)
    
        #error_test = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
        #bias___ = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
        #variance___ = np.mean( (z_pred - np.mean(z_pred, axis=1, keepdims=True))**2 )
        #error_train = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
        #(error_test, bias___,variance___ , error_train, R2_t, np.std(Betas, axis = 0), np.mean(Betas, axis = 0))



    return (polylamda,error_train,error_test, bias___,variance___ )

def OLSConfidenceInterval (x,y,n,X,PolyDimRange):
    # This is the main program for Part B
    # Loop through all relevant plynomials (2-5) and for each loop:
    # -Create the Design Matrix for current polinomial
    # -
    # -Loop through a selected points of 
    for dim in range(PolyDimRange-1):
        X1=DesignMatrix(x,y,dim+2)
        X11= pd.DataFrame(X1)
        print(X11)
        Z1=FrankeFunction(x,y)
        Z1 = AddNoise(Z1,n)
        Z1Tilde=BetaMatrixInv(X1,Z1)
        #KRC - Dette skal puttes inn i en strukt for analyse - print out!
        # The mean squared error 
        #ßprint("Polynomial DIM    : %")
        print("Ploynomial Dim    : %d" % int(dim+2))
        print("Mean value        : %.2f" % np.mean(Z1Tilde))
        print("Mean(Code) value  : %.2f" % GetMean(Z1Tilde))
        print("Scaling YPredict - mean")
        #print(SubMean(Z1Tilde))
        #print("Variance          : %.2f" % np.var(Z1,Z1Tilde))
        print("Mean squared error: %.5f" % mean_squared_error(Z1, Z1Tilde))
        print("MSE(code) value")
        print(MSE(Z1,Z1Tilde))   
        print("Variance(code):")
        print(GetVariance(Z1Tilde))                       
        # Explained variance score: 1 is perfect prediction                                 
        print('R2 Variance score: %.5f' % r2_score(Z1, Z1Tilde))
        print("R2(code) value")
        print(GetR2(Z1,Z1Tilde))
        VarYTilde=GetVariance(Z1Tilde)
        ConfidenceIntervals(VarYTilde,np.mean(Z1Tilde),len(Z1Tilde.shape))
        # Loop and add to dictionary
    return

def KfoldCrossValidation(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType):
    # This is the "main" program for Part b)
    KFoldRetur=KFold(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType)
    #return (polylamda,error_train,error_test, bias___,variance___ , error_train)
    #Plot for max polynomial degree 5 for ErrorTrain
    plt.plot(KFoldRetur[0], KFoldRetur[1], label='Error train')
    plt.plot(KFoldRetur[0], KFoldRetur[2], label='Error test')
    plt.plot(KFoldRetur[0], KFoldRetur[3], label='Bias')
    plt.plot(KFoldRetur[0], KFoldRetur[4], label='Variance')
    #plt.plot(KFoldRetur[0], KFoldRetur[1], label='Error train')
    #plt.plot(f[0], f[2], label='bias')
    #plt.plot(f[0], f[3], label='Variance')
    if (ReggrType=="OLS"):
        plt.xlabel("Polynomial")
        pltTitle=str(ReggrType) + ":K-Fold" + ":Samples:" + str(Samples) + ":Folds:" + str(Folds)
        pltFigName=str(ReggrType) +":K-Fold" + ":Samples:" + str(Samples) + ":Folds:" + str(Folds) + ".png"
    else:
        plt.xlabel("Lamdas") 
        pltTitle=str(ReggrType)+ ":K-Fold" + ":Samples:" + str(Samples) + "Lamdas:"
        pltFigName=str(ReggrType)+ ":K-Fold" + ":Samples:" + str(Samples) + "Lamdas:" + ".png"
    plt.ylabel("Loss/Error")
    plt.title(pltTitle)
    plt.legend()
    plt.savefig(pltFigName)
    plt.show()
    return

def BootStrapCrossValidation(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType):
    # This is the "main" program for Part b)
    BootstrapRetur=BootStrap(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType)
    #return (polylamda,error_train,error_test, bias___,variance___ , error_train)
    #Plot for max polynomial degree 5 for ErrorTrain
    plt.plot(BootstrapRetur[0], BootstrapRetur[1], label='Error train')
    plt.plot(BootstrapRetur[0], BootstrapRetur[2], label='Error test')
    plt.plot(BootstrapRetur[0], BootstrapRetur[3], label='Bias')
    plt.plot(BootstrapRetur[0], BootstrapRetur[4], label='Variance')
    #plt.plot(BootstrapRetur[0], BootstrapRetur[1], label='Error train')
    #plt.plot(BootstrapRetur[0], BootstrapRetur[2], label='bias')
    #plt.plot(BootstrapRetur[0], BootstrapRetur[3], label='Variance')
    if (ReggrType=="OLS"):
        plt.xlabel("Polynomial")
        pltTitle=str(ReggrType) + ":Bootstrap" + ":Samples:" + str(Samples) + ":Polynomial:" + str(Degrees)
        pltFigName=str(ReggrType) +"-Bootstrap" + "-Samples" + str(Samples) + "-Polynomial-" + str(Degrees) + ".png"
    else:
        plt.xlabel("Lamdas") 
        pltTitle=str(ReggrType)+ ":Bootstrap" + ":Samples:" + str(Samples) + "Lamdas:"
        pltFigName=str(ReggrType)+ "-Bootstrap" + "-Samples-" + str(Samples) + "Lamdas-"+".png"
    plt.ylabel("Loss/Error")
    plt.title(pltTitle)
    plt.legend()
    plt.savefig(pltFigName)
    plt.show()
    return 

def RigdeRegression(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType):
    # This is the "main" program for Part d)
      # This is the "main" program for Part b)
    KFoldRetur=KFold(X,Degrees,Lamdas,Folds,x,y,z,Samples,ReggrType)
    #return (polylamda,error_train,error_test, bias___,variance___ , error_train)
    #Plot for max polynomial degree 5 for ErrorTrain
    plt.plot(KFoldRetur[0], KFoldRetur[1], label='Error train')
    plt.plot(KFoldRetur[0], KFoldRetur[2], label='Error test')
    plt.plot(KFoldRetur[0], KFoldRetur[3], label='Bias')
    plt.plot(KFoldRetur[0], KFoldRetur[4], label='Variance')
    #plt.plot(KFoldRetur[0], KFoldRetur[1], label='Error train')
    #plt.plot(f[0], f[2], label='bias')
    #plt.plot(f[0], f[3], label='Variance')
    pltTitle=str(ReggrType) + ":Samples:" + str(Samples) + ":Lamda:" + str(Lamdas)
    plt.xlabel("Polynomial")
    plt.ylabel("Loss/Error")
    plt.title(pltTitle)
    plt.legend()
    plt.show()
    return 

def LassoRegression(x,y,z,k,SetOfLamdas):
    # This is the "main" program for Part e)
    # NB
    # Give a critical discussion of the three methods and a 
    # judgement of which model fits the data best.
    #
    SetOfLamdas=[0.001, 0.01, 0.1, 1, 2]
    error = np.zeros(len(SetOfLamdas))
    bias = np.zeros(len(SetOfLamdas))
    variance = np.zeros(len(SetOfLamdas))
    polylamda = np.zeros(len(SetOfLamdas))
    for lamda in SetOfLamdas: 
        lamda_fold =KFold(Folds,x,y,z,Dim,Lamda,"Lasso")
        error_ = lamda_fold[0]
        bias_ = lamda_fold[2]
        #print(bias_)
        variance_ = lamda_fold[3]
       # print('AAA')
        #print(SetOfLamdas.index(lamda))
        polylamda[SetOfLamdas.index(lamda)] = lamda
        error[SetOfLamdas.index(lamda)] = error_
        bias[SetOfLamdas.index(lamda)] = bias_
        variance[SetOfLamdas.index(lamda)] = variance_
        #plt.plot(f[0], f[1], label='Error')
        #plt.plot(f[0], f[2], label='bias')
        #plt.plot(f[0], f[3], label='Variance')
        #plt.legend()
        #plt.show()
    return

def IntroRealData():
    #############################################################
    ######                      PART F                     ######
    #############################################################
    # Import and prepare data analysis
    # 
    #
    # Load the terrain                                                                                  
    # terrain = imread('SRTM_data_Norway_1.tif')
    terrain = imread('SRTM_data_Norway_2.tif')

    # just fixing a set of points
    N = 1000
    m = 5 # polynomial order                                                                            
    terrain = terrain[:N,:N]
    # Creates mesh of image pixels                                                                      
    x = np.linspace(0,1, np.shape(terrain)[0])
    y = np.linspace(0,1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x,y)
    # Note the use of meshgrid
    # NB-KRCC
    # X = create_X(x_mesh, y_mesh,m)
    # you have to provide this function

    # Show the terrain                                                                                  
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return

def RealDataAnalyzis(PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples):
    #############################################################
    ######                      PART G                    ######
    #############################################################
    # Import and prepare data analysis
    # Load the terrain                                                                                  
    # terrain = imread('SRTM_data_Norway_1.tif')
    terrain = imread('SRTM_data_Norway_2.tif')

    # just fixing a set of points
    N = 1000                                                                  
    terrain = terrain[:N,:N]

    #Extract x and y from topographic data (image)
    x=terrain[0]
    y=terrain[1]
    #Scale x & y's
    x=SubMean(x)
    y=SubMean(y)

    #PolyDim=10
    Samples=N
   
    z = FrankeFunction(x, y)

    # Plot the surface
    # z need to be in two dim, to be able to plot?

    #fig = plt.figure()
    #ax = fig.gca(projection='3d') 

    #surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Plot the surface.surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.show()

    # z = AddNoise(z,Samples)
    X = DesignMatrix(x,y,PolyDim)
    BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"OLS")
    BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"Ridge")
    BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"Lasso")
    
    # Creates mesh of image pixels                                                                      
    x_m = np.linspace(0,1, np.shape(terrain)[0])
    y_m = np.linspace(0,1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x_m,y_m)
    

    # Show the terrain       
    fig = plt.figure()
    ax = fig.gca(projection='3d')                                                                           
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return

def OLSRigdeLassoRealData():
    # Run selected samples of data points
    # and run analyzis of all three methods 
    # and present this i documentation and 
    # plots (located in plots folder on GitHub repository)
    
    return

def MainModule():
    Samples = 2000
    x = np.random.uniform(0,1,size=Samples)
    y = np.random.uniform(0,1,size=Samples)

    #Scale the data based on subracting the mean
    x=SubMean(x)
    y=SubMean(y)

    #Part-A
    # Test for plynomials up to 5'th degree by using the Franke's function
    # Regression analysis using OLS/MSE, to find confidence intervals,
    # varances, MSE. Use scaling of the data (subtractiong mean) and add noise.
    #
    PolyDim=20
    z = FrankeFunction(x, y)
    z = AddNoise(z,Samples)
    X = DesignMatrix(x,y,PolyDim)
    Lamdas = [0.0001, 0.001, 0.01, 0.1, 1, 2]
    Bootstraps=1000
    Folds=5
    
    # Set to 15 to getter a smoother curve
    #OLSConfidenceInterval (x,y,Samples,X,PolyDim)
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"OLS")
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"Ridge")
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"Lasso")
   
    #Part-B
    # Bias-Variance crossvalidation using OLS with Bootstrapping
    #
    #KfoldCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"OLS")
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,x,y,z,Samples,"OLS")

    #Part-C
    #Compare K-fold and Bootstrap on the OLS/MSE
    #KfoldCrossValidation(X,PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples,"OLS")
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples,"OLS")

    #Part-D
    #Use Matrix Inversion for 
    # Make an analysis on lamdas, by using bootstrap
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples,"Ridge")

    #Part-E
    #Use Matrix Inversion for 
    # Make an analysis on lamdas, by using bootstrap
    # Compare and analyze the three methods (OLS,Ridge,Lasso)  
    #BootStrapCrossValidation(X,PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples,"Lasso")

    #Part-F
    # Set up a routine to read REAL data.
    #IntroRealData()

    #Part-G
    # Set up a routine to read REAL data.
    RealDataAnalyzis(PolyDim,Lamdas,Folds,Bootstraps,x,y,z,Samples)
    
    return

# Call Main Module and mark selektet Parts of the 
# program to run
MainModule()
