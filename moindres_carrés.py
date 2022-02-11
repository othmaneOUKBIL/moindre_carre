from matplotlib import pyplot as plt
import numpy as np


'''GENERATE AND PLOT ALL THE DATA'''


N=15
x=np.linspace(0,5,N)
x=np.transpose(x)
y=(x-2)**2 +np.random.randn(N)

#polynomial of order k

k=5

#compute Least Squares fit

A=[]
for i in range(k+1):
    A+=[x**(k-i)]
A=np.transpose(A)
what=np.linalg.lstsq(A,y,rcond=None)[0]
x1 = np.linspace(0,5,200)
x1 = np.transpose(x1)
A1=[]
for i in range(k+1):
    A1+=[x1**(k-i)]

A1=np.transpose(A1)
y1=np.dot(A1,what)
#plot the data points and the best fit
plt.figure(1)
plt.plot(x,y,'b.',x1,y1,'k-','MarkerSize',5)
err=np.linalg.norm(y- np.dot(A,what))

'''NO CROSS-VALIDATAION'''

kmax=5  #maximum polynomial order

#generate one large A matrix. We can get the matrices for particular k
#values by just picking the correct columns from this matrix.

A=[]
for i in range(kmax+1):
    A+=[x**(kmax-i)]

A=np.transpose(A)
err=np.zeros((kmax,1))

for i in range(kmax):
    Amat=A[:,kmax-i-1:]
    what2= np.linalg.lstsq(Amat,y,rcond=None)[0]
    #compute the error and divide by the number of points
    err[i] = np.linalg.norm(y - np.dot(Amat,what2)) / N
    
    
'''USIGN CROSS-VALIDATION'''


T = 12           #number of training points to use
trials = 1000      #number of averaging trials

#generate one large A matrix. We can get the matrices for particular k
#values by just picking the correct columns from this matrix.


A=[]
for i in range(k+1):
    A+=[x**(k-i)]

A=np.transpose(A)
errcv = np.zeros((kmax,trials))
for t in range(1,trials):
    r=np.random.permutation(N)
    train=r[:T]
    test=r[T:]
    for i in range(kmax):
        Atrain=A[train,kmax-i+1:]
        Atest=A[test,kmax-i+1:]
        ytrain=y[train]
        ytest=y[test]
        what2=np.linalg.lstsq(Atrain,ytrain,rcond=None)[0]
        #compute error and divide by the number of test points
        errcv[i][t]=(np.linalg.norm(ytest - np.dot(Atest,what2)) / (N-T))
avg_err_cv = np.mean(errcv,axis=1)

#compare the performance of the least-squares on training data to the
#performance when using cross-validation
plt.figure(2)
polynomial_model = ['Model 1','Model 2','Model 3','Model 4','Model 5']

xaxis = np.arange(kmax)

plt.bar(xaxis-0.2,err.reshape(5,),0.4,label = 'No Cross Validation',color = 'b')
plt.bar(xaxis+0.2,avg_err_cv.reshape(5,),0.4,label= 'Cross Validation',color = 'r')

plt.xticks(xaxis,polynomial_model)
plt.legend()
plt.show()

        
    



    






