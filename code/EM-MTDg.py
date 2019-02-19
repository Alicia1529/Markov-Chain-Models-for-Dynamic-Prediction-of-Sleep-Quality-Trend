#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:02:15 2018

@author: Alicia
"""

import numpy as np
import math    
import matplotlib.pyplot as plt  
import data
import dice
import copy
import MTD as mtd_b
import random
from mpl_toolkits.mplot3d  import Axes3D
import matplotlib as mpl

class EM_MTDg:
    def __init__(self):
        d = data.Data()
        b = mtd_b.MTD()
        self.data = d.get_d()        
        self.order = 2
        self.epsilon = 0.05
        self.dim = int(1/d.states)
        self.occurence = self.compute_occurence()[0]
        self.total_n = self.compute_occurence()[1]
        self.init_phi = np.array([b.lambda1,b.lambda2]) 
        self.init_Q = [b.q1,b.q2]
#        self.final_phi = np.array([b.lambda1,b.lambda2]) 
#        self.final_Q = [b.q1,b.q2]
        self.final_phi = self.iteration(100)[0]
        self.final_Q = self.iteration(100)[1]        
        self.P = self.P(6)
    
    def Estimation_step(self,phi_k,Q_k):#change the number of nested loops by hand
        P = np.zeros(shape=(self.dim,self.dim,self.dim,self.order)) #Xt-2 Xt-1 Xt order
        for g in range(self.order):#step
            for s2 in range(self.dim):
                for s1 in range(self.dim):
                    for s0 in range(self.dim):
                        if g ==0:
                            ig = s1
                        else:
                            ig = s2
                        numerator = phi_k[g]*Q_k[g][ig][s0]
                        denominator = phi_k[0]*Q_k[0][s1][s0]+phi_k[1]*Q_k[1][s2][s0]
                        if denominator != 0:
                            P[s2][s1][s0][g] = numerator/denominator    
        return P

    def Maximization_step(self,P):
        phi_k = np.zeros(self.order)
        Q_k = [np.zeros(shape=(self.dim,self.dim)),np.zeros(shape=(self.dim,self.dim))]
        
        for i in range(self.order): #step
            phig = 0
            for s2 in range(self.dim):
                for s1 in range(self.dim):
                    for s0 in range(self.dim):
                        phig += P[s2][s1][s0][i]*self.occurence[s2][s1][s0]
            phig /= (self.total_n-self.order)
            phi_k[i] = phig
        
        # g = 1 
        for i in range(self.dim):
            for j in range(self.dim):# which is also s0 in the numerator 
                numerator = 0
                denominator = 0
                for s2 in range(self.dim):
                    numerator += P[s2][i][j][0]*self.occurence[s2][i][j]
                    for s0 in range(self.dim):
                        denominator += P[s2][i][s0][0]*self.occurence[s2][i][s0]
                if denominator != 0:
                    Q_k[0][i][j] = numerator/denominator  
                    
                        
        # g = 2
        for i in range(self.dim):
            for j in range(self.dim):# which is also s0 in the numerator 
                numerator = 0
                denominator = 0
                for s1 in range(self.dim):
                    numerator += P[i][s1][j][0]*self.occurence[i][s1][j]
                    for s0 in range(self.dim):
                        denominator += P[i][s1][s0][0]*self.occurence[i][s1][s0]
                if denominator != 0:
                    Q_k[1][i][j] = numerator/denominator  
        return phi_k, Q_k
    
    def iteration(self,iter_num):
        P_k = self.Estimation_step(self.init_phi,self.init_Q)
        phi_k,Q_k = self.Maximization_step(P_k)
        X = []
        Y = []
        Z = []
        for i in range(iter_num):   
            old = (phi_k,Q_k)
            X.append(phi_k[0])
            Y.append(phi_k[1])
            P_k = self.Estimation_step(phi_k,Q_k)
            phi_k,Q_k = self.Maximization_step(P_k)
            new = (phi_k,Q_k)
            Z.append(self.compute_loglikelihood(new[0],new[1]))            
            delta = self.compute_loglikelihood(new[0],new[1])-self.compute_loglikelihood(old[0],old[1])
#            print("delta:\n",delta)
#            print(self.compute_loglikelihood(phi_k,Q_k))
#            print(phi_k)            
            if delta<self.epsilon:
                break
        plt.rcParams['figure.figsize'] = (8.0, 4.0)     
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = X,Y,Z
#        ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
        #  将数据点分成三部分画，在颜色上有区分度
#        ax.scatter(x[:10], y[:10], z[:10], c='y')  # 绘制数据点
#        ax.scatter(x[10:30], y[10:30], z[10:30], c='r')
#        ax.scatter(x[20:], y[20:], z[20:], c='g')
        
        for idx in range(len(Z)):
            xs = x
            ys = y
            zs = [z[idx]]
         
            # You can provide either a single color or an array. To demonstrate this,
            # the first bar of each set will be colored cyan.
            ax.bar(xs, ys, zs, zdir='z', alpha=0.8)        
        
        
        ax.set_zlabel('LL')  # 坐标轴
        ax.set_ylabel('φ2')
        ax.set_xlabel('φ1')
        plt.show()        
    
        
        
        
        
        return phi_k,Q_k
        
        
        
    def compute_loglikelihood(self,phi,Q):
        log_likelihood = 0
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]-2):
                if self.data[i][j+2]!= 0:
                    s2 = int(self.data[i][j] )-1
                    s1 = int(self.data[i][j+1]) -1
                    s0 = int(self.data[i][j+2]) -1
                    log_likelihood += math.log(phi[0]*Q[0][s1][s0]+phi[1]*Q[1][s2][s0])
        return log_likelihood

    def compute_occurence(self): #assume it's 2nd order  Xt-2 Xt-1 Xt
        occurence = np.zeros(shape=(self.dim, self.dim,self.dim))
        total_n = 0

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]-self.order):
                if self.data[i][j+2] > 0:
                    last2 = int(self.data[i][j])-1
                    last1 = int(self.data[i][j+1])-1
                    last0 = int(self.data[i][j+2])-1
                    total_n += 1
                    occurence[last2][last1][last0]+=1
                    
        return occurence,total_n    
#(array([[[ 695.,  172.,   18.,    0.,    0.],
#        [   6.,   87.,   60.,   12.,   12.],
#        [   1.,    1.,    7.,    9.,    5.],
#        [   0.,    0.,    0.,    0.,    2.],
#        [   0.,    0.,    0.,    0.,    1.]],
#
#       [[ 177.,    3.,    0.,    0.,    0.],
#        [  96.,  298.,  132.,    0.,    0.],
#        [   0.,    1.,  160.,   38.,    0.],
#        [   0.,    1.,    1.,    4.,    6.],
#        [   0.,    0.,    0.,    0.,   12.]],
#
#       [[   5.,    0.,    0.,    0.,    0.],
#        [  80.,  141.,    7.,    0.,    0.],
#        [   3.,  171.,  586.,  178.,    0.],
#        [   0.,    0.,    8.,  184.,   38.],
#        [   0.,    0.,    0.,    0.,    5.]],
#
#       [[   0.,    0.,    0.,    0.,    0.],
#        [   8.,    0.,    0.,    0.,    0.],
#        [   1.,   61.,  185.,    5.,    0.],
#        [   0.,    6.,  200.,  579.,  124.],
#        [   0.,    0.,    0.,    7.,  163.]],
#
#       [[   0.,    0.,    0.,    0.,    0.],
#        [   1.,    0.,    0.,    0.,    0.],
#        [   0.,    1.,    0.,    0.,    0.],
#        [   0.,    1.,   43.,  142.,    0.],
#        [   0.,    1.,    1.,  179.,  670.]]]), 5801)
            
    def BIC(self):
        # LL is the log-likelihood of the model
        # p is the number of independent parameters 
        # N the number of data
        n = self.compute_occurence()[1]
        phi_k,Q_k = self.iteration(100)
        
        ll= self.compute_loglikelihood(phi_k,Q_k)

        p_mtdg = 2*self.dim *(self.dim-1)+1
        BIC_mtdg = -2*ll+p_mtdg*math.log(n)

        
        return ll,BIC_mtdg
    
    #given sequence, predict num points afterwards
    def prediction(self,sequence,num):
        d = dice.Dice(self.dim)
        #for mtdg prediction
        sequence_mtdg = sequence[:]
        for j in range(num):
            prob = [0]
            last = sequence_mtdg[-1] 
            last2 = sequence_mtdg[-2]
            for i in range(self.dim-1):         
                p = self.final_phi[0] * self.final_Q[0][last-1][i]+self.final_phi[1] * self.final_Q[1][last2-1][i] 
                prob.append(p+prob[-1])
            
            prob.append(1)
            d.set_bounds(prob)
            sequence_mtdg.append(d.roll())

        plt.figure(figsize=(8,5),dpi = 80)
        plt.title('Prediction by MTDg model')
        plt.subplot(1,1,1)   
        plt.yticks([0,1,2,3,4,5])
        X1 = []
        Y1 = []
        idx = 0
        for i in sequence_mtdg:
            X1.append(idx)
            Y1.append(i)
            idx+=1
        
        plt.plot(X1,Y1,'red')         
        return sequence_mtdg
    
    
        #randomly choose 3+num points in whole data and return the whole sequence and pre-sequence
    def prediction_sequence(self,num):
        n = 3
        #i is day number and j is the time in a day 
        i = random.randint(0,len(self.data)-1)
        #need n points in the sequence and predict m afterwards
        while True:
            j=random.randint(0,len(self.data[0])-num-n)          
            if self.data[i][j+num+n-1]==0:
                continue
            else:
                sequence = self.data[i][j:j+num+n]
                break
        #only 2 points in the pre-sequence
        pre_sequence = sequence[1:3]

        return list(pre_sequence), list(sequence),(i,j)    
    
    def prediction_test(self,pre_sequence,num):
        last2 = int(pre_sequence[-2])
        last = int(pre_sequence[-1])
        #print(last, last2)
        #probability that a point is lower
        global P 
        P = [0]
        if last>1:
            for j in range(self.dim):
                p = [self.final_phi[1]*self.final_Q[0][last2-1][j]+self.final_phi[0]*self.final_Q[1][last-1][j]]
                self.pre_recursion(pre_sequence+[j+1],0,num,P,p)
                #self.pre_recursion([last2,last]+[j+1],0,num,P,p)

                
        return P[0]
 
    def pre_recursion(self,seq,cur_step,max_step,P,p):
        if cur_step<max_step and seq[-1]>=seq[1]:
            cur_step+=1
            for i in range(self.dim):
                new_seq = copy.deepcopy(seq)
                new_seq.append(i+1)
                last2 = int(new_seq[-3]-1)
                last1 = int(new_seq[-2]-1)
                last0 = int(new_seq[-1]-1)
                new_p = copy.deepcopy(p)
                new_p[0] = new_p[0]*(self.final_phi[1]*self.final_Q[1][last2][last0]+self.final_phi[0]*self.final_Q[0][last1][last0])
                self.pre_recursion(new_seq,cur_step,max_step,P,new_p)
        else:
            if seq[-1]<seq[1]:
                #print(seq,p,P)
                P[0]= p[0]+P[0]
                
    def P(self,num):
        P_total = np.zeros(shape=(num, self.dim, self.dim))
        for k in range(num):
            for i in range(self.dim):
                for j in range(self.dim):
                    #last2 is j last is i
                    P_total[k][j][i] = self.prediction_test([j+1,i+1],k)
                    #print(j,i,"prob",self.prediction_test(j+1,i+1,6))

        return P_total
        
    def wake_up(self,num,day,time,plot = ""):
        p_wake = 0.4
        pre_sequence,sequence,(day,time)= self.prediction_sequence(num)
        last = pre_sequence[-1]
        last2 = pre_sequence[-2]
        ori_num = num
        Sleep = True
        i = 0
        while Sleep:
            last2 = int(sequence[i+1])
            last =  int(sequence[i+2])
            prob = float(self.P[num-1][last2-1][last-1])
            if prob < p_wake:
                Sleep = False  
                break 
            else:
                i += 1
                num -=1
            if num ==0:
                Sleep = False 
                break 
                
        seq = sequence[:i+3]
        wake_up_t = time+len(seq)-1
        
        if plot =="figure1":
            plt.figure(figsize=(8,5),dpi = 80)
            plt.title('Wake up time by MTDg model ')
            plt.subplot(1,1,1)   
            plt.yticks([0,1,2,3,4,5])
            
            X1 = []
            Y1 = []            
            X2 = []
            Y2 = []  
            for i in range(len(self.data[day])):
                if self.data[day][i] !=0:
                    X1.append(i)
                    Y1.append(self.d.transformed_interval[day][i])
                    if i>time and i<time+1+ori_num+3:
                        X2.append(i)
                        Y2.append(self.d.transformed_interval[day][i])
            
            plt.plot(X2,Y2,'magenta')    
            plt.plot(X1,Y1,'cadetblue')
            wake_up_t = time+len(seq)-1
            y = self.d.transformed_interval[day][wake_up_t]
            plt.plot([wake_up_t],[y],'+')
        
        elif plot =="figure2":   
            plt.figure(figsize=(8,5),dpi = 80)
            plt.title('Wake up time by MTD model ')
            plt.subplot(1,1,1)   
            plt.yticks([0,1,2,3,4,5])
            
            X1 = []
            Y1 = []            
            X2 = []
            Y2 = []     
            for i in range(len(self.data[day])):
                if self.data[day][i] !=0:
                    X1.append(i)
                    Y1.append(self.data[day][i])
                    if i>time and i<time+1+9:
                        X2.append(i)
                        Y2.append(self.data[day][i])
            
            plt.plot(X2,Y2,'magenta')    
            plt.plot(X1,Y1,'cadetblue')
            
            y = self.data[day][wake_up_t]
            plt.plot([wake_up_t],[y],'+')
        
#        print('time:',time)
#        print('wake up time',wake_up_t)
                    
        return sequence,seq 
    
    def error(self,sequence,seq,num,day,time):
        #day,time,sequence,seq = self.wake_up(num,day,time)
        #print(sequence,seq)
        lowest = min(sequence[2:])
        corrext_time = sequence.index(lowest)
        error = seq[-1]-lowest
#        print("error",error)
        return error
        
    
    
       
if __name__ == "__main__": 
    mtdg = EM_MTDg()

    
    
#    P = mtdg.Estimation_step(mtdg.init_phi,mtdg.init_Q)
#    phi_1,Q_1 = mtdg.Maximization_step(P)
#    print('Initial:')
#    print('phi:')
#    print(em.init_phi)
#    print('Q:')
#    print(em.init_Q[0])
#    print(em.init_Q[1])
#    print('loglikelihood:',em.compute_loglikelihood(em.init_phi,em.init_Q))
#    print('\n')
#    print('Iteration:')
#    ll,BIC = em.BIC()
#    print("MTDg log-likelihood:\n",ll)
#    print("MTDg BIC:\n",BIC)
#    print(em.prediction([2,2,3],10))
    
#    cur_step = 0
#    max_step = 6
#    num = 6
    #print(mtd.prediction_test(list(m),2))
#    (day,time) = mtdg.prediction_sequence(num)[2]
    #sequence,seq = mtdg.wake_up(num,day,time)
    #print("wake up:", mtd.wake_up(num,day,time,"figure1"))
    #mtdg.error(sequence,seq,num,day,time)
    
    
    
    
    
   #calculate average error of x times 
#    total = 0 
#    x = 10000
#    for i in range(x):
#        (day,time) = mtdg.prediction_sequence(num)[2]
#        sequence,seq =  mtdg.wake_up(num,day,time)
#        #print("wake up:", mtd.wake_up(num,day,time))
#        total += mtdg.error(sequence,seq,num,day,time)
#        
#    total /= (x*5)
#    print("MTDg:",total)
    
