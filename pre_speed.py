#coding=utf-8
import scipy.io as scio
import numpy as np
import sys
import sktensor
import time
from sktensor.dtensor import dtensor
'''
path = '/home/qiushye/trafficAllDayFSOdata.mat'
FSO = scio.loadmat(path)['trafficDayFSOdata']
Occu = FSO[200:215,:,:,2]
scio.savemat('/home/qiushye/Occudata.mat',{'Occu':Occu})
Flow = FSO[200:215,:,:,0]
scio.savemat('/home/qiushye/Flowdata.mat',{'Flow':Flow})
Speed = FSO[200:215,:,:,1]
scio.savemat('/home/qiushye/Speeddata.mat',{'Speed':Speed})
print(Flow.mean())
'''
gz_path = '/home/qiushye/GZ_data/speed_tensor.mat'
Vdata = scio.loadmat(gz_path)['tensor']
miss_pos = np.where(Vdata==0)
SP = Vdata.shape
print(SP)
A,B,C = miss_pos[0].tolist(),miss_pos[1].tolist(),miss_pos[2].tolist()
dim1_miss = set(A)
Ndata = np.zeros((SP[0]-len(dim1_miss),SP[1],SP[2]))
Zdata = np.zeros((len(dim1_miss),SP[1],SP[2]))
j,k = 0,0
for i in range(SP[0]):
    if i not in dim1_miss:
        Ndata[j,:,:] = Vdata[i,:,:]
        j += 1
    else:
        Zdata[k,:,:] = Vdata[i,:,:]
        k += 1
time_s = time.time()
S = Vdata.copy()
SD = dtensor(Vdata)
T_ = SD.unfold(1)
print(time.time()-time_s,'s')
SVD_ = np.linalg.svd(T_)
print(time.time()-time_s,'s')
sys.exit()
Y = S
X = Y.copy()
M = {}
MX,MY,M_fold = {},{},{}
n = 1
if 1:
            MX[n] = dtensor(X).unfold(n)
            MY[n] = dtensor(Y).unfold(n)
            print(time.time()-time_s,'s')
            M_temp = (0.1*MX[n]+0.1*MY[n])/0.2
            para_fi = 1/0.2
            U,sigma,VT = np.linalg.svd(M_temp)
            print(time.time()-time_s,'s')
            row_s = len(sigma)
            mat_sig = np.zeros((row_s,row_s))
            max_rank = 0
            for ii in range(row_s):
                mat_sig[ii,ii] = max(sigma[ii]-para_fi,0)
            M[n] = np.dot(np.dot(U[:,:row_s],mat_sig),VT[:row_s,:])
            M_fold[n] = M[n].fold()
            print(time.time()-time_s,'s')
X = np.sum([0.1*M_fold[i] for i in range(3)],axis=0)
Y_temp = np.sum([0.1*M_fold[i] for i in range(3)],axis=0)
time_e = time.time()
print(str(time_e-time_s)+'s')
