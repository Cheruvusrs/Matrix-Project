import numpy as np 
import matplotlib.pyplot as plt
a=np.array([2,1])
b=np.array([1,-1])

N1=np.vstack((a,b))
p1=np.array([3,1])
C=np.matmul(np.linalg.inv(N1),p1)

P=np.array([1,-1])
n=np.matmul(np.array([[0,-1],[1,0]]),P-C)
r=np.linalg.norm(P-C)
len=100

print(n)


x1=np.zeros((2,len))
x2=np.zeros((2,10))
x3=np.zeros((2,10))
x4=np.zeros((2,10))
lam1=np.linspace(0,2*np.pi,len)
lam2=np.linspace(-10,10,10)


for i in range(len):
	temp1= np.array([C[0]+r*np.cos(lam1[i]),C[1]+r*np.sin(lam1[i])])
	x1[:,i]=temp1.T
for i in range(10):
	temp1= np.array([lam2[i],3-2*(lam2[i])])
	x2[:,i]=temp1.T
for i in range(10):
	temp1= np.array([lam2[i],lam2[i]-1])
	x3[:,i]=temp1.T	
for i in range(10):
	temp1=P+lam2[i]*n
	x4[:,i]=temp1.T
plt.plot(x1[0,:],x1[1,:])
plt.axis('equal')
plt.plot(x2[0,:],x2[1,:])
plt.axis('equal')
plt.plot(x3[0,:],x3[1,:])
plt.axis('equal')
plt.plot(x4[0,:],x4[1,:])
plt.axis('equal')
plt.plot(C[0],C[1],'o')
plt.text(C[0]*(1-0.2),C[1]*(1-0.2),'C')
plt.plot(P[0],P[1],'o')
plt.text(P[0]*(1-0.2),P[1]*(1-0.2),'P')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-3,3)
plt.ylim(-2,4)
	
	
plt.grid()
plt.show()
