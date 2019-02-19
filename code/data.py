import math
import numpy as np
import matplotlib.pyplot as plt

class Data:

    def __init__(self):
        # change variables
        self.days = 66
        self.states = 1 / 5
        self.time_interval = 5 #mins
        self.file_name = "../input/anqi-data.txt"
        self.raw_d = [[] for i in range(self.days)]  #887 dataset , unmodified
        self.transformed_d = [[] for i in range(self.days)] #887 dataset in state
        self.pix2state()
        self.transformed_interval = self.extract_minutes()[0]#111 in decimal
        self.d = self.extract_minutes()[1] #in state

    def pix2state(self):
        f = open(self.file_name, "r")

        l = 0
        # raw_data contains a list of [number of days] lists,
        # each containing [number of pixels] state elements
        for line in f:
            if line =="\n":
                pass
            else:
                line = line.strip().split(" ")
                self.raw_d[l].append(line[1])
                self.raw_d[l].append(line[2])
                self.transformed_d[l].append(line[1])
                self.transformed_d[l].append(line[2])
                for i in range(len(line)-3):
                    self.raw_d[l].append(line[i+3])
                    self.transformed_d[l].append(int(float(line[i+3])//self.states+1))
                l += 1

        f.close()

    def extract_minutes(self):
        max_sleep_time = 0
        pix = int(self.transformed_d[0][1])

        for i in range(self.days):
            sleep_time = float(self.transformed_d[i][0])
            max_sleep_time = sleep_time if sleep_time > max_sleep_time else max_sleep_time

        shortest_interval = math.floor(pix*self.time_interval/max_sleep_time)
        longest_night = math.floor(pix/shortest_interval) + 1
        t_d = np.zeros(shape=(self.days, longest_night))
        d = np.zeros(shape=(self.days, longest_night))

        for i in range(self.days):
            sleep_time = float(self.transformed_d[i][0])
            interval = math.floor(pix*self.time_interval/sleep_time)
            take = 0
            for j in range(2, pix, interval):
                t_d[i][take] = self.raw_d[i][j]
                d[i][take] = self.transformed_d[i][j]
                take += 1
        return (t_d,d)

    def get_d(self):
        return self.d
    def plot_raw_d(self,num,d=False): #input the n-th day
        plt.figure(figsize=(8,5),dpi = 80)
        plt.subplot(1,1,1)
        X1 = []
        Y1 = []
        idx = 0
        for i in self.raw_d[num][2:]:
            X1.append(idx)
            Y1.append(i)
            idx+=1
        if d ==True:
            X2 = []
            Y2 = []
            for i in self.transformed_d[num][2:]:
                if self.d[num][i]!=0:
                    Y2.append(i*self.states*100-10)
            plt.plot(X1,Y1,'r-',X1,Y2,'b-')
        else:
            plt.plot(X1,Y1,'red') # original

    def plot_d(self,num,d = False): #input the n-th day the transformed curve; every t minute
        plt.figure(figsize=(8,5),dpi = 80)
        plt.subplot(1,1,1)
        X = []
        Y1 = []
        Y2 = []
        idx = 0
        for i in range(2,len(self.d[num])):
            if self.d[num][i]!=0:
                X.append(idx)
                Y1.append(self.transformed_interval[num][i])
                Y2.append(self.d[num][i]*self.states-0.1)
                idx+=1
        if d ==True:
            print(len(X))
            plt.plot(X,Y1,'r-',X,Y2,'b-')
        else:
            plt.plot(X,Y2,'r-')

if __name__ == "__main__":
    data = Data()
#	d = data.get_d()
#	print(d[1][130:])
#
#	print(len(data.raw_d[-1]))
#	data.plot_raw_d(0,True)
#	data.plot_d(0,True)
#    c1 = 0
#    c2 = 0
#    c3 = 0
##    c4 = 0
##    c5 = 0
#
#    for i in range(data.days):
#        for j in range(len(data.d[0])-1):
#            if data.d[i][j]==2 and data.d[i][j+1]!=0:
#                if data.d[i][j+1]==1:
#                    c1+=1
#                elif data.d[i][j+1]==2:
#                    c2+=1
#                elif data.d[i][j+1]==3:
#                    c3+=1
##                elif data.d[i][j+1]==4:
##                    c4+=1
##                elif data.d[i][j+1]==5:
##                    c5+=1
#    print(c1,c2,c3)
#
#plt.rcParams['figure.figsize'] = (8.0, 4.0)
#X=[1,2,3]#[1,2,3,4,5]
#Y=[c1,c2,c3]  #[c1,c2,c3,c4,c5]
#fig = plt.figure()
#plt.bar(X,Y,0.4,color="skyblue")
#plt.xlabel("To State #")
#plt.ylabel("Number of Occurence")
#plt.title("From state 3")
#
#
#plt.show()
#plt.savefig("barChart.jpg")
