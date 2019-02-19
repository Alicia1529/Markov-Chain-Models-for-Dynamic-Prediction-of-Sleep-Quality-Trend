import data
import dice
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

class MC:

    def __init__(self):

        self.d = data.Data()
        self.data = self.d.get_d()
        self.dim = int(1/self.d.states)
        
        self.f1 = self.mc1(self.data)[0]
        self.p1 = self.mc1(self.data)[1]
        
        self.f2 = self.rtm(self.data)[0]
        self.p2 = self.rtm(self.data)[1]
        
        self.f3 = self.mc3(self.data)[0]#
        self.p3 = self.mc3(self.data)[1]#
        
        self.pred_mc2 = [self.make_prediction_matrix_mc2(i) for i in range(1,7)]
    
    #count total frequency
    def total_count(self, data):
        total = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                total += data[i][j]
        return total

    # First Order Markov Chain
    def mc1(self, data):
        p1 = np.zeros(shape=(self.dim, self.dim))
        f1 = np.zeros(shape=(self.dim, self.dim))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]-1):
                if data[i][j+1] > 0:
                    row = int(data[i][j]) - 1
                    col = int(data[i][j+1]) - 1
                    f1[row][col] += 1
                    f1[row][col] = int(f1[row][col])

        for i in range(self.dim):
            Sum = sum(f1[i,:])
            for j in range(self.dim):
                p1[i][j] = f1[i][j] / Sum

        return f1, p1

    # Reduced Trnsition Matrix
    def rtm(self, data):
        p2 = np.zeros(shape=(self.dim*self.dim, self.dim))
        f2 = np.zeros(shape=(self.dim*self.dim, self.dim))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]-2):
                if self.data[i][j+2] > 0:
                    last2 = int(data[i][j])
                    last1 = int(data[i][j+1])
                    last0 = int(data[i][j+2])
                    row = last1 - 1 + (last2-1) * self.dim
                    col = last0 - 1
                    f2[row][col] += 1
                    f2[row][col] = int(f2[row][col])

        for i in range(self.dim*self.dim): #row
            Sum = sum(f2[i,:])
            if Sum != 0:
                p2[i,:] = f2[i,:] / Sum

        return f2, p2  
    
    def mc3(self, data):#
        p3 = np.zeros(shape=(self.dim**3, self.dim))
        f3 = np.zeros(shape=(self.dim**3, self.dim))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]-3):
                if self.data[i][j+3] > 0:
                    last3 = int(data[i][j])
                    last2 = int(data[i][j+1])
                    last1 = int(data[i][j+2])
                    last0 = int(data[i][j+3])
                    row = last1 - 1 + (last2-1) * self.dim + (last3-1)*self.dim*self.dim
                    col = last0 - 1
                    f3[row][col] += 1
                    f3[row][col] = int(f3[row][col])

        for i in range(self.dim**3): #row
            Sum = sum(f3[i,:])
            if Sum != 0:
                p3[i,:] = f3[i,:] / Sum

        return f3, p3   
    
    def log_likelihood(self, data):
        log_likelihood_mc1 = 0
        for i in range(self.dim):
            for j in range(self.dim):
                if self.p1[i][j]!=0:
                    log_likelihood_mc1 += self.f1[i][j] * math.log(self.p1[i][j])
                    
        log_likelihood_mc2 = 0
        for i in range(self.dim**2):
            for j in range(self.dim):
                if self.p2[i][j]!=0:
                    log_likelihood_mc2 += self.f2[i][j] * math.log(self.p2[i][j])
                    
        log_likelihood_mc3 = 0
        for i in range(self.dim**3):
            for j in range(self.dim):
                if self.p3[i][j]> 0:
                    log_likelihood_mc3 += self.f3[i][j] * math.log(self.p3[i][j])
                
        return log_likelihood_mc1,log_likelihood_mc2,log_likelihood_mc3
    
    def BIC(self, data):
        # LL is the log-likelihood of the model
        # p is the number of independent parameters 
        #N the number of data
        n = 0 
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if self.data[i][j]!=0:
                    n+=1

        p_mc1 = self.dim *(self.dim-1)
        p_mc2 = self.dim **2*(self.dim-1)
        p_mc3 = self.dim **3*(self.dim-1)
        
        BIC_mc1 = -2*self.log_likelihood(data)[0]+p_mc1*math.log(n)
        BIC_mc2 = -2*self.log_likelihood(data)[1]+p_mc2*math.log(n)
        BIC_mc3 = -2*self.log_likelihood(data)[2]+p_mc3*math.log(n)
        
        return BIC_mc1,BIC_mc2,BIC_mc3
    
    def prediction_mc1(self, sequence, num):
        d = dice.Dice(self.dim)

        #for MC1 prediction
        sequence_mc1 = sequence[:]
        for j in range(num):
            prob = [0]
            last = sequence_mc1[-1]        
            for i in range(self.dim-1):              
                prob.append(self.p1[last-1][i]+prob[-1])

            prob.append(1)
            d.set_bounds(prob)
            sequence_mc1.append(d.roll())
                 
        plt.figure(figsize=(8,5),dpi = 80)
        plt.subplot(1,1,1)
        plt.yticks([0,1,2,3,4,5])
        plt.title('Prediction by MC1 model')
        X1 = []
        Y1 = []
        idx = 0
        for i in sequence_mc1:
            X1.append(idx)
            Y1.append(i)
            idx+=1
        plt.plot(X1,Y1,'red') 
        
        #for MC2 prediction
        sequence_mc2 = sequence[:]
        for j in range(num):
            prob = [0]
            last = sequence_mc2[-1] 
            last2 = sequence_mc2[-2]
            for i in range(self.dim-1): 
                row = last - 1 + (last2-1) * self.dim 
                prob.append(self.p2[row][i]+prob[-1])
            prob.append(1)
            d.set_bounds(prob)
            sequence_mc2.append(d.roll())

        plt.figure(figsize=(8,5),dpi = 80)
        plt.subplot(1,1,1)   
        plt.yticks([0,1,2,3,4,5])
        plt.title('Prediction by MC2 model')
        X2 = []
        Y2 = []
        idx = 0
        for i in sequence_mc2:
            X2.append(idx)
            Y2.append(i)
            idx+=1
        plt.plot(X2,Y2,'blue') 
        
        #for MC3 prediction
        sequence_mc3 = sequence[:]
        for j in range(num):
            prob = [0]
            last = sequence_mc3[-1] 
            last2 = sequence_mc3[-2]
            last3 = sequence_mc3[-3]
            for i in range(self.dim-1): 
                row = last - 1 + (last2-1) * self.dim + (last3-1)*self.dim*self.dim
                prob.append(self.p3[row][i]+prob[-1])
            prob.append(1)
            d.set_bounds(prob)

            sequence_mc3.append(d.roll())      

        plt.figure(figsize=(8,5),dpi = 80)
        plt.subplot(1,1,1)   
        plt.yticks([0,1,2,3,4,5])
        plt.title('Prediction by MC3 model')
        X3 = []
        Y3 = []
        idx = 0
        for i in sequence_mc3:
            X3.append(idx)
            Y3.append(i)
            idx+=1
        plt.plot(X3,Y3,'green') 
        
        return sequence_mc1,sequence_mc2,sequence_mc3

    def dependency(self, mc=1):
        alpha = 0
        if mc == 1:
            for i in range(self.dim):
                for j in range(self.dim):
                    pj = np.sum(self.f1[:,j])/self.total_count(self.f1)
                    alpha += self.f1[i][j] * math.log1p(self.p1[i][j]/pj)
        else:
            for i in range(self.dim*self.dim):
                for j in range(self.dim):
                    pj = np.sum(self.f2[:,j])/self.total_count(self.f2)
                    alpha += self.f2[i][j] * math.log1p(self.p2[i][j]/pj)

        print(2 * alpha)
        return 2 * alpha

    def temporary_stationary(self):
        beta = 0
        # set the interval as 22 days
        d1 = self.data[0:22]
        temp_p = [self.mc1(d1)] #[([f1], [p1]), ([f2], [p2]), ...]
        d2 = self.data[22:44]
        temp_p.append(self.mc1(d2))
        d3 = self.data[44:66]
        temp_p.append(self.mc1(d3))
        # get the orinigal p for the whole dataset
        self.mc1(self.data)
        print(self.p1)

        for k in range(3):
            for i in range(self.dim):
                for j in range(self.dim):
                    beta += temp_p[k][0][i][j] * math.log1p(temp_p[k][1][i][j]/self.p1[i][j])

        print(2*beta)
        return 2*beta
    
    def make_prediction_matrix_mc2(self, num):# t0->t1
        matrix = np.zeros(shape=(self.dim, self.dim))
        for t0 in range(self.dim):
            for t1 in range(self.dim):
                global P_mc2
                P_mc2 = [0]
                seq = [t0+1,t1+1]
                if t1+1 > 1:
                    #one step
                    for i in range(self.dim):
                        row = int( t0*self.dim+t1 )
                        p = [self.p2[row][i]]
                        self.mc2_recursion(seq+[i+1],1,num,P_mc2,p)
                matrix[t0][t1] = P_mc2[0]               

        return matrix
 
    def mc2_recursion(self,seq,cur_step,max_step,P_mc2,p): #current step;maximum step
        if cur_step<max_step and seq[-1]>=seq[1]:
            cur_step += 1 
            for i in range(self.dim):
                new_seq = copy.deepcopy(seq)
                new_seq.append(i+1)
                row = int((new_seq[-3]-1)*self.dim+new_seq[-2]-1)
                col = int(new_seq[-1]-1)
                new_p = copy.deepcopy(p)
                new_p[0] = new_p[0]*self.p2[row][col]
#                print('p0cheng',row,col,self.p2[row][col],'p0',new_p[0])
                self.mc2_recursion(new_seq,cur_step,max_step,P_mc2,new_p)
        else:
            if seq[-1]<seq[1]:
                P_mc2[0] = p[0]+P_mc2[0]

    def mc2_wakeup(self,day,time,num,wakeup,plot = ''): #t0->t1
        t0 = int(self.data[day][time])
        t1 = int(self.data[day][time+1])
        wakeup_time = time
        X = []
        Y = []

        for i  in range(num):
            if self.pred_mc2[num-i-1][t0-1][t1-1] <wakeup:
                wakeup_time = time+i+1
                break
            else:
                t0 = t1
                t1 = int(self.data[day][time+1+i+1])
        if wakeup_time == time:
            wakeup_time = time+num+1
                
        if plot=='figure1':       
            plt.figure(figsize=(8,5),dpi = 80) 
            plt.subplot(1,1,1)   
            plt.yticks([0,1,2,3,4,5])
            plt.title('Wake Up Prediction')
            X1 = []
            X2 = []
            Y1 = []
            Y2 = []            
            for i in range(len(self.data[day])):
                if self.data[day][i] !=0:
                    X1.append(i)
                    Y1.append(self.d.transformed_interval[day][i])
                    if i>time and i<=time+1+num:
                        X2.append(i)
                        Y2.append(self.d.transformed_interval[day][i])
            y = self.d.transformed_interval[day][wakeup_time]
            plt.plot(X1,Y1,'blue')
            plt.plot(X2,Y2,'red')
            plt.plot([wakeup_time],[y],'ro')            
        elif plot =='figure2':
            plt.figure(figsize=(8,5),dpi = 80) 
            plt.subplot(1,1,1)   
            plt.yticks([0,1,2,3,4,5])
            plt.title('Wake Up Prediction')
            X1 = []
            X2 = []
            Y1 = []
            Y2 = []            
            for i in range(len(self.data[day])):
                if self.data[day][i] !=0:
                    X1.append(i)
                    Y1.append(self.data[day][i])
                    if i>time and i<=time+1+num:
                        X2.append(i)
                        Y2.append(self.data[day][i])
            y = self.data[day][wakeup_time]            
            plt.plot(X1,Y1,'blue')
            plt.plot(X2,Y2,'r+')
            plt.plot([wakeup_time],[y],'ro')
            
        return wakeup_time
    
    def wake_up_test(self, iteration, num, mc2=True): # for mc2
        if num != 6:
            self.pred_mc2 = [self.make_prediction_matrix_mc2(i) for i in range(1,num+1)]
        
        X=[]
        Y=[]
        for k in range(100):
            X.append(k/100)
            error = 0
            for i in range(iteration):
                day = np.random.randint(0,self.d.days)
                time = np.random.randint(0,len(self.data[0])-2-num)
                while self.data[day][time+num+2] == 0:
                    time = np.random.randint(0,len(self.data[0])-2-num)
                if mc2:
                    wakeup_time = self.mc2_wakeup(day, time, num,k/100) #t0->t1
                else:
                    wakeup_time = self.traditional_wakeup(day, time, num)
                min_sleep_quality = min(self.data[day][time+1:time+1+num+1])
                error+= self.data[day][wakeup_time] - min_sleep_quality 
    #            print('each time:',self.data[day][wakeup_time] - min_sleep_quality)
            Y.append(error/iteration/(self.dim-1))
        plt.figure(figsize=(8,5),dpi = 80) 
        plt.subplot(1,1,1)   
        plt.plot(X,Y,'blue')        
#        print('error/iteration',error/iteration/self.dim)
#        return error/iteration/self.dim
        
    def traditional_wakeup(self, day, time, num):
        t0 = int(self.data[day][time+1])
        wakeup_time = time
        for i  in range(num):
            if t0 == 2:
                wakeup_time = time + 1 + i
                break
            else:
                t0 = int(self.data[day][time+1+i+1])
        if wakeup_time == time:
            wakeup_time = time+num+1
            
        return wakeup_time

    def test(self):
        # First order markov chain
        FirstOrderMarkovChain = np.array([[0.7,0.2,0.1],[0.2,0.6,0.2],[0.1,0.4,0.5]])
        P1 = np.zeros(9).reshape(3,3)
        Output1 = [1]
        dim = FirstOrderMarkovChain.shape[0]
        d = dice.Dice(dim)

        for i in range(5000):
            last = Output1[-1]
            prob = [0]
            for i in range(dim-1):
                prob.append(FirstOrderMarkovChain[last-1][i]+prob[-1])
            prob.append(1)
            d.set_bounds(prob)
            Output1.append(d.roll())

        for j in range(len(Output1)-1):
            row = Output1[j]-1
            col = Output1[j+1]-1
            P1[row][col]+=1
            
        for k in range(dim):
            Sum = sum(P1[k,:])
            for m in range(dim):
                P1[k][m] /=Sum

        print("First Order Input:\n", FirstOrderMarkovChain)
        print("First Order Output:\n", P1)

        # Second order markov chain
        SecondOrderMarkovChain = np.dot(FirstOrderMarkovChain,FirstOrderMarkovChain)
        P2 = np.zeros(9).reshape(3,3)

        for j in range(len(Output1)-2):
           row = Output1[j]-1
           col = Output1[j+2]-1
           P2[row][col]+=1
           
        for k in range(dim):
           Sum = sum(P2[k,:])
           for m in range(dim):
               P2[k][m] /=Sum

        print("Second Order Input:\n", SecondOrderMarkovChain)
        print("Second Order Output:\n", P2)

        # Reduced Transition Matrix
        ReducedTransitionMatrix = np.array([[0.7,0.2,0.1],[0.2,0.6,0.2],[0.1,0.4,0.5], 
                                           [0.1,0.8,0.1],[0.2,0.4,0.4],[0.1,0.7,0.2],
                                           [0.4,0.3,0.3],[0.2,0.1,0.7],[0.5,0.1,0.4]])
        P = np.zeros(27).reshape(9,3)
        Output = [1,2]
        dim0 = ReducedTransitionMatrix.shape[0] #9
        dim1 = ReducedTransitionMatrix.shape[1] #3
        d = dice.Dice(dim1)

        for i in range(10000):
            last1 = Output[-1]
            last2 = Output[-2]
            prob = [0]
            for i in range(dim1-1):
                idx = last2-1+(last1-1)*3
                prob.append(ReducedTransitionMatrix[idx][i]+prob[-1])
            prob.append(1)
            d.set_bounds(prob)
            Output.append(d.roll())

        for j in range(len(Output)-2):
            last2 = Output[j]
            last1 = Output[j+1]
            last0 = Output[j+2]
            row = last2-1+(last1-1)*3
            col = last0-1
            P[row][col]+=1

        for k in range(dim0): #row  
            Sum = sum(P[k,:])
            P[k,:] /= Sum

        print("Reduced Transition Matrix Input:\n", ReducedTransitionMatrix)
        print("Reduced Transition Matrix Output:\n", P)

if __name__ == "__main__": 
    mc = MC()
    print('Who',mc.d.file_name)
    print("Days:",mc.d.days)
    print("States:",mc.dim)
    print("Time interval",mc.d.time_interval)
    
    mc.mc2_wakeup(2,60,6,0.15,'figure1')
    mc.mc2_wakeup(2,60,6,0.15,'figure2')


#    mc.wake_up_test(100000, 6)
#    mc.wake_up_test(100000, 6, False)

#    mc.dependency()
#    mc.temporary_stationary()

#    print("MC1 log-likeligood:\n",mc.log_likelihood(mc.data)[0])
#    print("MC2 log-likeligood:\n",mc.log_likelihood(mc.data)[1])
#    print("MC3 log-likeligood:\n",mc.log_likelihood(mc.data)[2])
    
#    print("MC1 BIC:\n",mc.BIC(mc.data)[0])
#    print("MC2 BIC:\n",mc.BIC(mc.data)[1])
#    print("MC3 BIC:\n",mc.BIC(mc.data)[2])
    
    #print(mc.rtm(mc.data))