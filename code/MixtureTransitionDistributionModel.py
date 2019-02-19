#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:15:13 2018

@author: chenzhengyi
"""

import MC
import data
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import dice
import copy
np.set_printoptions(suppress=True)

class MTD:
    def __init__(self):
        self.d = data.Data()
        self.data = self.d.get_d()
        self.dim = int(1/self.d.states) #dimension as states
        self.order = 2

        self.c1 = self.cal_cg(1)    
        self.q1 = self.cal_qg(1,self.c1) #n-step , number count
        self.c2 = self.cal_cg(2) 
        self.q2 = self.cal_qg(2,self.c2) 

        self.lambda1 = self.init_value()[0]
        self.lambda2 = self.init_value()[1]
        self.Q = self.init_value()[2]
        self.predict_number = 6
        self.P = self.P(6)
   
    def cal_cg(self,step):
        cg = np.zeros(shape=(self.dim, self.dim))
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]-step):
                if self.data[i][j+step] > 0:
                    row = int(self.data[i][j]) -1
                    col = int(self.data[i][j+step]) -1
                    cg[row][col]+=1                              
                    cg[row][col] = int(cg[row][col])
        return cg      
 
    def cal_qg(self,step,cg):
        qg = np.zeros(shape=(self.dim, self.dim))
        for i in range(self.dim):
            Sum = sum(cg[i,:])
            for j in range(self.dim):
                qg[i][j] = cg[i][j] / Sum        
        return qg
              
    def cal_TCg(self,step,cg):
        TCg = 0
        for i in range(self.dim):
            for j in range(self.dim):
                TCg += cg[i][j]
        return TCg
    
    def cal_u(self,cg,step): #ci step, depends on g
        # When order is 2, g=1 g=2
        TCg = self.cal_TCg(step,cg)
                        
        numerator = 0
        for i in range(self.dim):
            for j in range(self.dim):
                if cg[i][j] !=0:
                    numerator += cg[i][j]*math.log2((sum(cg[i,:])*sum(cg[:,j]))/(cg[i][j]*TCg))                    

        denominator = 0
        for j in range(self.dim):
            denominator += sum(cg[:,j])*math.log2(sum(cg[:,j])/TCg)
            
        u = numerator/denominator
        return u

    def init_value (self):
        u1 = self.cal_u(self.c1,1)
        u2 = self.cal_u(self.c2,2)
        lambda1 = u1/(u1+u2)
        lambda2 = u2/(u1+u2)

        if lambda1 > lambda2:
            init_Q = self.q1
        else:
            init_Q = self.q2
        return lambda1, lambda2, init_Q

    def reevaluate_lambda(self, lambda_p, lambda_n, theta):
        if lambda_p + theta[0] > 1:
            theta[0] = 1 - lambda_p
        elif lambda_n - theta[0] < 0:
            theta[0] = lambda_n
        elif lambda_p + theta[0] < 1 and lambda_n - theta[0] > 0:
            lambda_p += theta[0]
            lambda_n -= theta[0]
#            theta[0] /= 2
        return lambda_p, lambda_n, theta
    
    # update lambda and Q
    def update(self, theta, pairs, f): 
        # compute partial derivative of lambda1 and lambda2
        p_lambda1 = 0
        p_lambda2 = 0

        for i in range(self.dim):
            for j in range(len(pairs)):
                i_1, i_2 = pairs[j][0], pairs[j][1]
                row = i_1 + i_2 * self.dim
                sum_g = self.lambda1 * self.Q[i_1][i] + self.lambda2 * self.Q[i_2][i]
                if sum_g != 0:
                    p_lambda1 += f[row][i] * self.Q[i_1][i] / sum_g
                    p_lambda2 += f[row][i] * self.Q[i_2][i] / sum_g

        if p_lambda1 > p_lambda2:
            lambda_p = self.lambda1 # lambda+
            lambda_n = self.lambda2 # lambda-
        else:
            lambda_p = self.lambda2
            lambda_n = self.lambda1
        if lambda_p == 1:
            return self.lambda1, self.lambda2, self.Q
        else:
            self.lambda1, self.lambda2, theta = self.reevaluate_lambda(lambda_p, lambda_n, theta)
        
        # compute partial derivative of Q
        p_Q = np.zeros(shape=(self.dim, self.dim))
        for i in range(self.dim):
            for j in range(len(pairs)):
                # for lambda1
                i_0, i_1, i_2 = pairs[j][0], i, pairs[j][1]  
                row = i_1 + i_2 * self.dim
                sum_g = self.lambda1 * self.Q[i_1][i_0] + self.lambda2 * self.Q[i_2][i_0]
                if sum_g != 0:
                    p_Q[i_1][i_0] += f[row][i_0] * self.lambda1 / sum_g
                # for lambda2
                i_0, i_1, i_2 = pairs[j][0], pairs[j][1], i
                row = i_1 + i_2 * self.dim
                sum_g = self.lambda1 * self.Q[i_2][i_0] + self.lambda2 * self.Q[i_1][i_0]
                if sum_g != 0:
                    p_Q[i_2][i_0] += f[row][i_0] * self.lambda2 / sum_g
        
        # for each row of Q, compare derivatives and update
        for row in range(self.dim):
            max_pQ = np.argpartition(p_Q[row], -1)[::-1] # return indecies of values
            
            min_pQ = np.argpartition(p_Q[row], -1)
            imax = max_pQ[0] # Q+ 
            
            ind_min = 0 
            while self.Q[row][min_pQ[ind_min]] == 0:
                ind_min += 1
            imin = min_pQ[ind_min] # Q-
            
            if self.Q[row][imax] == 1:
                return self.lambda1, self.lambda2, self.Q
            else:
                if self.Q[row][imax] + theta[row+1] > 1:
                    theta[row+1] = 1 - self.Q[row][imax]
                elif self.Q[row][imin] - theta[0] < 0:
                    theta[row+1] = self.Q[row][imin]
                elif self.Q[row][imax] + theta[row+1] < 1 and self.Q[row][imin] - theta[row+1] > 0:
                    self.Q[row][imax] += theta[row+1]
                    self.Q[row][imin] -= theta[row+1]
#                    theta[row+1] /= 2
                  
#        print(self.lambda1, self.lambda2)
#        print(self.Q)
#        print(theta)
        return theta
    
    def main(self):
        # set hyperparameters
        theta = [0.01 for i in range(self.dim+1)] # initailize (dim + 1) delta to the same value
        threshold = 0.0001
        max_iter = 50
        
        pairs = [] #generate a (self.dim^2) list to hold all possible pairs
        for i in range(self.dim):
            for j in range(self.dim):
                pairs.append([i,j])
                
        n = MC.MC()
        f, p = n.rtm(self.data)
        
        log_likelihood = 0
        for k in range(max_iter):
            theta = self.update(theta, pairs, f)

            # compute new log likelihood
            new_log_likelihood = 0
            for i in range(self.dim):
                for j in range(len(pairs)):
                    # for lambda1
                    i_1, i_2 = pairs[j][0], pairs[j][1]
                    row = i_1 + i_2 * self.dim                 
                    denominator = self.lambda1*self.Q[i_1][i]+self.lambda2*self.Q[i_2][i]
                    if denominator != 0:
                        new_log_likelihood += f[row][i] * math.log(denominator)
            # check whether log likelihood converges
            if log_likelihood == 0:
                log_likelihood = new_log_likelihood
            elif new_log_likelihood - log_likelihood < threshold:
                #print(new_log_likelihood - log_likelihood, "gives log_likelihood:", log_likelihood)
                return new_log_likelihood
            else:
                log_likelihood = new_log_likelihood
        print()
        print("final:", self.lambda1, self.lambda2)
        print("final:", self.Q)
        print("log_likelihood:", log_likelihood)
        return log_likelihood
        
    
    
    def BIC(self):
        # LL is the log-likelihood of the model
        # p is the number of independent parameters 
        # N the number of data
        n = 0 
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if self.data[i][j]!=0:
                    n+=1

        p_mtd = self.dim *(self.dim-1)+1
        BIC_mtd = -2*self.main()+p_mtd*math.log(n)

        
        return BIC_mtd
    
    #predict num points given the sequence
    def prediction(self,sequence,num):
        d = dice.Dice(self.dim)
        #for mtd prediction
        sequence_mtd = sequence[:]
        for j in range(num):
            prob = [0]
            last = sequence_mtd[-1] 
            last2 = sequence_mtd[-2]
            for i in range(self.dim-1): 
                p = self.lambda1 * self.Q[last-1][i]+self.lambda2 * self.Q[last2-1][i] 
                prob.append(p+prob[-1])
            
            prob.append(1)
            d.set_bounds(prob)
            sequence_mtd.append(d.roll())

        plt.figure(figsize=(8,5),dpi = 80)
        plt.title('Prediction by MTD model')
        plt.subplot(1,1,1)   
        plt.yticks([0,1,2,3,4,5])
        X2 = []
        Y2 = []
        idx = 0
        for i in sequence_mtd:
            X2.append(idx)
            Y2.append(i)
            idx+=1
        
        plt.plot(X2,Y2,'red')         
        return sequence_mtd
    
    #randomly choose 3+num points in whole data and return the whole sequence and pre-sequence
    def prediction_sequence(self,num):
        n = 3
        #i is day number and j is the time in a day 
        i = random.randint(0,len(self.data)-1)
        #need n points in the sequence and predict num afterwards
        while True:
            j=random.randint(0,len(self.data[0])-n-num)          
            if self.data[i][j+num+n-1]==0:
                continue
            else:
                sequence = self.data[i][j:j+n+num]
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
                p = [self.lambda2*self.Q[last2-1][j]+self.lambda1*self.Q[last-1][j]]
                self.pre_recursion(pre_sequence+[j+1],0,num,P,p)
                #self.pre_recursion([last2,last]+[j+1],0,num,P,p)
                
        #return the probabiliy
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
                new_p[0] = new_p[0]*(self.lambda2*self.Q[last2][last0]+self.lambda1*self.Q[last1][last0])
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
        print("error",error)
        
        return error
        
if __name__ == "__main__":
    mtd = MTD()
    #print("lambda1:\n",mtd.lambda1)
    #print("lambda2:\n",mtd.lambda2)
    #print("initial Q:\n",mtd.Q)
    
    cur_step = 0
    max_step = 6
    num = 6
    
    pre_s,s,(day,time) = mtd.prediction_sequence(num)
    
    #print(mtd.prediction_test(list(pre_s),2))
    #sequence,seq =  mtd.wake_up(num,day,time)
    
    #print("wake up:", mtd.wake_up(num,day,time,"figure1"))
    
#    mtd.error(sequence,seq,num,day,time)
#    mtd.main()
#    print(mtd.prediction([1,1,1],10))
#    ll = mtd.main()
#    print(" ")
#    print("MTD loglikelihood:\n",ll)
#    print("MTD BIC:\n",mtd.BIC())

    
    #test x times average error
    total = 0 
    x = 100000
    for i in range(x):
        (day,time) = mtd.prediction_sequence(num)[2]
        sequence,seq =  mtd.wake_up(num,day,time)
        #print("wake up:", mtd.wake_up(num,day,time))
        total += mtd.error(sequence,seq,num,day,time)
        
    total /= (x*5)
    print("MTD",total)