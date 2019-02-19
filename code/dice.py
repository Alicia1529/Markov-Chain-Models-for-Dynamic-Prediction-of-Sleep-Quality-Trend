# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:23:36 2016

@author: zhengzhang
"""
import random

class Dice:
    def __init__(self, sides=3):
        self.n_sides = sides
        self.bounds = [float(x)/sides for x in range(0, sides)]
        self.bounds.append(1.0)
        self.point = None
        self.lands = 0
   
    def set_bounds(self, r):
        assert len(r) == len(self.bounds)
        self.bounds = r
        
    def get_bounds(self):
        return self.bounds
                
    def roll(self):
# replace the following line with you code
# it should set self.point as a random var in [0, 1]
# return which side the dice lands on 
        self.point = random.uniform(0,1) # it is not randint
        for i in range(self.n_sides):
            if self.point >self.bounds[i] and self.point <= self.bounds[i+1]:
                break
        self.lands = i+1
        return self.lands


if __name__ == "__main__":       
    d = Dice()
    ''' make a biased dice '''
    #set dice probability here
    
    d.set_bounds([0.0, 0.7,0.9,1.0])
    for i in range(50):
        print(d.roll())

