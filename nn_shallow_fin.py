#lung cancer classification using Shallow neural-network
#Size of input neuron = 15
#Size of 1-hidden layer = 500
#Size of Output neuron =1
#Neural network configuration 15 - 500 - 1. Two layer neural network(ip layer is ignored))  
import csv
import numpy as np
import matplotlib.pyplot as plt
def normalize(X):
          X_norm = X
          [r,c]=np.shape(X)
          mu = np.zeros([c,1])
          sigma = np.zeros([c,1])    
          for i in range(c):
                    mu[i] = np.mean(X[:,i])
                    sigma[i] = np.std(X[:,i])
                    X_norm[:,i] =(X[:,i]-mu[i])/sigma[i];
          return [X_norm,mu,sigma]
def sigmoid(z):
          return (1/(1+np.exp(-z)))
a=np.zeros((259,1))
b=np.zeros((259,1))
c=np.zeros((259,1))
d=np.zeros((259,1))
e=np.zeros((259,1))
f=np.zeros((259,1))
g=np.zeros((259,1))
h=np.zeros((259,1))
p=np.zeros((259,1))
j=np.zeros((259,1))
k=np.zeros((259,1))
l=np.zeros((259,1))
m=np.zeros((259,1))
n=np.zeros((259,1))
o=np.zeros((259,1))
Y=np.zeros((259,1))
X1=np.zeros((259,15))
i=0
#lung_cancer_train.csv file includes 259 training examples
with open("lung_cancer_train.csv", "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for lines in csv_reader:
      b[i]=lines['age']
      if(lines['gender']=='M'):
                a[i]=1
      else:
                a[i]=0
      if(lines['lung_cancer']=='true'):
                Y[i]=1
      else:
                Y[i]=0
      c[i]=lines['smoking']
      d[i]=lines['yellow_fingers']
      e[i]=lines['anxiety']
      f[i]=lines['peer_pressure']
      g[i]=lines['chronic_disease']
      h[i]=lines['fatigue']
      p[i]=lines['allergy']
      j[i]=lines['wheezing']
      k[i]=lines['alcohol_consuming']
      l[i]=lines['coughing']
      m[i]=lines['shortness_of_breath']
      n[i]=lines['swallowing_difficulty']
      o[i]=lines['chest_pain']
      i=i+1
for i in range(259):
              
  X1[i][0]=a[i]
  X1[i][1]=b[i]
  X1[i][2]=c[i]
  X1[i][3]=d[i]
  X1[i][4]=e[i]
  X1[i][5]=f[i]
  X1[i][6]=g[i]
  X1[i][7]=h[i]
  X1[i][8]=p[i]
  X1[i][9]=j[i]
  X1[i][10]=k[i]
  X1[i][11]=l[i]
  X1[i][12]=m[i]
  X1[i][13]=n[i]
  X1[i][14]=o[i]
[X,mu,sigma]=normalize(X1)
X=X.T
Y=Y.T
n_h=500 #hidden layer size
n_x=X.shape[0] #ip neuron size =15
n_y=Y.shape[0] #op neuron size =1
W1=np.random.randn(n_h,n_x)*0.01
b1=np.zeros(shape=(n_h,1))
W2=np.random.randn(n_y,n_h)*0.01
b2=np.zeros(shape=(n_y,1))
m=309
cost1=[]
for i in range(2000):
              Z1= np.dot(W1,X)+ b1
              A1=np.tanh(Z1) #activation function at ip side of hidden layer 
              Z2= np.dot(W2,A1) + b2
              A2=sigmoid(Z2)#activation function at ip side of output layer
              #A2=A2.reshape(-1,1)
              #print(A2.shape)
              logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
              cost = - np.sum(logprobs) / m
              cost = float(np.squeeze(cost))
              cost1.append(cost)
              #cost= - (np.dot(np.log(A2),Y) + np.dot((1-Y),np.log(1-A2)))/m
              #print(cost)
              dZ2 = A2 - Y
              dW2 = (1 / m) * np.dot(dZ2, A1.T)
              db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
              dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
              dW1 =  np.dot(dZ1, X.T)
              db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
              lr=0.05
              W1 = W1- lr*dW1
              W2 = W2- lr*dW2
              b1= b1- lr*db1
              b2= b2- lr*db2
plt.plot(range(len(cost1)),cost1)
plt.xlabel('No. of iterations')
plt.ylabel('Cost function')
plt.title('Lung cancer classification using Neural network')
plt.show()
Ypredict=np.round(A2)
print('Training Accuracy: %d'%float((np.dot(Y,Ypredict.T)+ np.dot(1-Y,1-Ypredict.T))/float(Y.size)*100)+ '%')
at=np.zeros((50,1))
bt=np.zeros((50,1))
ct=np.zeros((50,1))
dt=np.zeros((50,1))
et=np.zeros((50,1))
ft=np.zeros((50,1))
gt=np.zeros((50,1))
ht=np.zeros((50,1))
pt=np.zeros((50,1))
jt=np.zeros((50,1))
kt=np.zeros((50,1))
lt=np.zeros((50,1))
mt=np.zeros((50,1))
nt=np.zeros((50,1))
ot=np.zeros((50,1))
Yt=np.zeros((50,1))
X1t=np.zeros((50,15))
#lung_cancer_test.csv file includes 50 testing examples
i=0
with open("lung_cancer_test.csv", "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for lines in csv_reader:
      bt[i]=lines['age']
      if(lines['gender']=='M'):
                at[i]=1
      else:
                at[i]=0
      if(lines['lung_cancer']=='true'):
                Yt[i]=1
      else:
                Yt[i]=0
      ct[i]=lines['smoking']
      dt[i]=lines['yellow_fingers']
      et[i]=lines['anxiety']
      ft[i]=lines['peer_pressure']
      gt[i]=lines['chronic_disease']
      ht[i]=lines['fatigue']
      pt[i]=lines['allergy']
      jt[i]=lines['wheezing']
      kt[i]=lines['alcohol_consuming']
      lt[i]=lines['coughing']
      mt[i]=lines['shortness_of_breath']
      nt[i]=lines['swallowing_difficulty']
      ot[i]=lines['chest_pain']
      i=i+1
for i in range(50):
              
  X1t[i][0]=at[i]
  X1t[i][1]=bt[i]
  X1t[i][2]=ct[i]
  X1t[i][3]=dt[i]
  X1t[i][4]=et[i]
  X1t[i][5]=ft[i]
  X1t[i][6]=gt[i]
  X1t[i][7]=ht[i]
  X1t[i][8]=pt[i]
  X1t[i][9]=jt[i]
  X1t[i][10]=kt[i]
  X1t[i][11]=lt[i]
  X1t[i][12]=mt[i]
  X1t[i][13]=nt[i]
  X1t[i][14]=ot[i]
[Xtest,mu,sigma]=normalize(X1t)
Xtest=Xtest.T
Ytest=Yt.T
Z1test=np.dot(W1,Xtest)+ b1
Atest=np.tanh(Z1test)
Z2test=np.dot(W2,Atest) + b2
A2test=sigmoid(Z2test)
Ytest_predict= np.round(A2test)
print('Testing Accuracy: %d'%float((np.dot(Ytest,Ytest_predict.T)+ np.dot(1-Ytest,1-Ytest_predict.T))/float(Ytest.size)*100)+ '%')

