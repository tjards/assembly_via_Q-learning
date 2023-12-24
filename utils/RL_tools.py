#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:28:01 2023

This program implements reinforcement learning for application on
the multi agent simulator. 

@author: tjards
"""

#%% import stuff
# ------------
import numpy as np
import random

#%% hyper parameters
# ----------------
options_range   = [2, 9]
nOptions       = 7         # number of action options
time_horizon    = 50        # how long to apply action and away reward (1 sec/0.02 sample per sec = 50) 


#%% define the q learning agent
# ----------------------------
class q_learning_agent:
    
    def __init__(self, nAgents):
        
        # initialize parameters
        self.nAgents       = nAgents
        self.nOptions       = nOptions
        #self.action_options = np.linspace(options_range[0], options_range[1], nAction)
        self.action_options = {state: np.linspace(options_range[0], options_range[1], self.nOptions) for state in range(self.nAgents)}
        self.explore_rate   = 1     # 1 = always learn, 0 = always exploit best option
        self.learn_rate     = 0.9   # alpha/lambda
        self.discount       = 0.8   # balance immediate/future rewards, (gamma): 0.8 to 0.99
        self.time_horizon   = time_horizon
        self.time_count     = 0
        
        # initialize Q table
        self.state          = {} 
        self.action         = {}
        
        #self.select_action()
        
        # for i in range(self.nAgents):
        #     self.state["Agent " + str(i)] = i
        #     self.action[i] = []
        #     for j in range(self.nAgents-1): 
        #         self.action[i] += [self.action_options[i][np.random.choice(self.nOptions)]] 
                
        self.nState         = self.nAgents
        self.nAction        = self.nAgents-1 * self.nOptions
        self.reward         = 0
        self.Q              = {}
        for i in range(self.nAgents):
            self.Q["Agent " + str(i)] = {}
            for j in range(self.nAgents-1):
                self.Q["Agent " + str(i)]["Neighbour " + str(j)] = {}
                for k in range(self.nOptions):
                    option_label = self.action_options[i][k]
                    self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(option_label)] = 0
                    
        self.select_action()
        
        
       # = np.zeros((self.nState, self.nAction))
        
    def select_action(self):
        
        if random.uniform(0, 1) < self.explore_rate:
            
            # Explore (select randomly)        
            #self.action = np.random.choice(self.nAction) 
            
            for i in range(self.nAgents):
                self.state["Agent " + str(i)] = i
                
                #self.action[i] = []
                #for j in range(self.nAgents-1): 
                #    self.action[i] += [self.action_options[i][np.random.choice(self.nOptions)]] 
                
                self.action["Agent " + str(i)] = {}
                for j in range(self.nAgents-1): 
                    self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = self.action_options[i][np.random.choice(self.nOptions)]
                    
        else:
            
            # test
            #self.Q["Agent 9"]["Neighbour 0"]["Option " + str(self.action_options[9][0])] = 999
            
            # Exploit (select best)
            for i in range(self.nAgents):
                self.action["Agent " + str(i)] = {}
                for j in range(self.nAgents-1): 
                    # get the max value
                    #self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)].values())
                    self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)], key=self.Q["Agent " + str(i)]["Neighbour " + str(j)].get)

    def compute_reward(self):
        
        self.reward = None
        
    def update_q_table(self):
        
        # note: for the "next state", we will use the "invereted" actions (i.e. what if the neighbour used the select states). 
        #   this assseses the future rewards of selecting, as there is a larger consensus at play.
        #   what about max rewards
        
        self.next_action = np.argmax(self.Q[self.next_state,:])
        self.Q[self.state, self.action] += np.multiply(self.learn_rate, self.reward + self.discount*self.Q[self.next_state, self.next_action] - self.Q[self.state, self.action])
        
        
#%% test

# nState = each agent has 1 state, a vector of control parameters
# nAction = each agent has nAgent-1 actions, the control parameter corresponding to each neighbour
# the Q-table keeps track of all these, so it has 1 "meta state" and nAgent * nAction-1 "meta actions"

# here, we input nState = number of agents, nAction = number of actions to choose from 

nAgents = 10
learning_agent = q_learning_agent(nAgents)   
test = learning_agent.action 

# initialize state-action
#np.zeros((len(self.params),len(self.params))) 

# randomly select indicies         
#[random.randint(0, nAction) for _ in range(nState)]          
            
        
        