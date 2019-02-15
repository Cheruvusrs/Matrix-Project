import numpy as np
import matplotlib.pyplot as plt
PQ=np.array([1,-np.sqrt(3)])
D=np.array([(3*(np.sqrt(3)))/2,1.5])
C=D-(1/(np.linalg.norm(np.array([np.sqrt(3),1]))))*(np.array([np.sqrt(3),1]))
omat=np.array([[0,1],[-1,0]])
z=np.linspace(0,2*(np.pi),1000)
M=np.array([[0.5,-np.sqrt(3)/2],[np.sqrt(3)/2,0.5]]).T
RQ=np.matmul(M,PQ)
RP=np.matmul(M,RQ)
Q=D-((np.sqrt(3))/np.linalg.norm(PQ))*(PQ)
P=2*D-Q
R=3*C-P-Q
E=(R+Q)/2
F=(R+P)/2




print(RQ)
print(RP)
print(R)




len=10
lam_1=np.linspace(0,1,len)
x_PQ = np.zeros((2,len))
x_RQ= np.zeros((2,len))
x_RP = np.zeros((2,len))
for i in range(len):
	temp1= P+lam_1[i]*(Q-P)
	x_PQ[:,i]=temp1.T
	temp2=R+lam_1[i]*(Q-R)
	x_RQ[:,i]=temp2.T
	temp3=R+lam_1[i]*(P-R)
	x_RP[:,i]=temp3.T
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')
plt.plot(x_RQ[0,:],x_RQ[1,:],label='$RQ$')
plt.plot(x_RP[0,:],x_RP[1,:],label='$RP$')
plt.plot(P[0],P[1],'o')
plt.text(P[0]*(1+0.1),P[1]*(1-0.1),'P')
plt.plot(Q[0],Q[1],'o')
plt.text(Q[0]*(1-0.2),Q[1]*(1),'Q')
plt.plot(R[0]*(1),R[1],'o')
plt.text(R[0]*(1-0.2),R[1]*(1+0.1),'R')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')

plt.plot(E[0],E[1],'o')

plt.plot(D[0],D[1],'o')





plt.plot(P[0],P[1],'o')
plt.plot(C[0],C[1],'o')
plt.text(C[0]*(1+0.1),C[1]*(1-0.1),'C')




len=10
lam_1=np.linspace(0,1,len)
x_EC = np.zeros((2,len))
x_DC= np.zeros((2,len))
x_FC = np.zeros((2,len))
for i in range(len):
	temp1= E+lam_1[i]*(C-E)
	x_EC[:,i]=temp1.T
	temp2=D+lam_1[i]*(C-D)
	x_DC[:,i]=temp2.T
	temp3=F+lam_1[i]*(C-F)
	x_FC[:,i]=temp3.T
plt.plot(x_EC[0,:],x_EC[1,:],label='$EC$')
plt.plot(x_FC[0,:],x_FC[1,:],label='$FC$')
plt.plot(x_DC[0,:],x_DC[1,:],label='$DC$')

plt.text(E[0]*(1+0.1),E[1]*(1-0.1),'E')

plt.text(D[0]*(1-0.1),D[1]*(1),'D')
plt.plot(F[0]*(1),F[1],'o')
plt.text(F[0]*(1-0.2),F[1]*(1+0.1),'F')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')

X=np.array([C[0]+np.cos(z),C[1]+np.sin(z)])


plt.plot(X[0],X[1])
plt.grid()
plt.axis('equal')
plt.show()
