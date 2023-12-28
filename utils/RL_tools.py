#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:28:01 2023

This program implements reinforcement learning for application on
the multi agent simulator. 

@author: tjards


Dev notes:
    
    27 Dec 2023: need to increase exploit rate over time 


"""

#%% import stuff
# ------------
import numpy as np
import random
import os
import json
import copy
from scipy.spatial import distance

#%% hyper parameters
# ----------------
options_range   = [2, 8]    # range of action options [min, max]
nOptions        = 3         # number of action options (evenly spaced between [min, max])
time_horizon    = 250       # how long to apply action and await reward (eg., 1 sec/0.02 sample per sec = 50)
time_horizon_v  = 0.2       # optional, max speed constraint to permit new action (higher makes more stable)

#%% data saving
# -------------
data_directory = 'Data'
file_path = os.path.join(data_directory, "data_Q.json")

# converts to dict to json'able
def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

#%% define the q learning agent
# ----------------------------
class q_learning_agent:
    
    def __init__(self, nAgents):
        
        # learning hyperparameters
        self.nAgents        = nAgents
        self.nOptions       = nOptions # defined above
        self.action_options = {state: np.linspace(options_range[0], options_range[1], self.nOptions) for state in range(self.nAgents)}
        self.explore_rate   = 1     # [0,1], 1 = always learn, 0 = always exploit best option
        self.learn_rate     = 0.5   # [0,1]
        self.discount       = 0.8   # balance immediate/future rewards, (gamma): 0.8 to 0.99
        self.time_horizon   = time_horizon
        self.time_horizon_v = time_horizon_v
        
        # initialize timers (global)
        self.time_count     = 0     # initialize 
        self.Q_update_count = 0     # initialize 
        
        # initialize timer (local)
        self.time_count_i = np.zeros((nAgents))
        self.Q_update_count_i = np.zeros((nAgents))
        
        # initialize data 
        self.data           = {}
        
        # initialize state/action
        self.state          = {} 
        self.action         = {}
        self.nState         = self.nAgents
        self.nAction        = self.nAgents * self.nOptions
        self.reward         = 0
        self.Q              = {}
        
        # initalize the Q-table       
        for i in range(self.nAgents):
            self.Q["Agent " + str(i)] = {}
            for j in range(self.nAgents):
                self.Q["Agent " + str(i)]["Neighbour " + str(j)] = {}
                # if not itself
                if i != j:
                    for k in range(self.nOptions):
                        option_label = self.action_options[i][k]
                        self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(option_label)] = 0
        
        # select an initial action
        self.select_action()

    # %% select an action 
    # ---------------------

    # global case
    # -----------         
    def select_action(self):
        
        # explore 
        if random.uniform(0, 1) < self.explore_rate:
            
            # for each agent 
            for i in range(self.nAgents):
                
                #self.state["Agent " + str(i)] = i 
                self.action["Agent " + str(i)] = {}
                
                # search through each neighbour
                for j in range(self.nAgents):
                    
                    # not itself
                    if i != j:
                        
                        # select an action (randomly)
                        self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = self.action_options[i][np.random.choice(self.nOptions)]
        
        # exploit             
        else:
            
            # for each agent 
            for i in range(self.nAgents):
                
                self.action["Agent " + str(i)] = {}
                
                # search through each agent 
                for j in range(self.nAgents): 
                    
                    # not itself
                    if i != j:
                        
                        # get the key for the max reward
                        temp = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)], key=self.Q["Agent " + str(i)]["Neighbour " + str(j)].get)
                        
                        # select the action corresponding to the max reward
                        self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = float(temp.replace("Option ",""))
                        
    # local case
    # ---------
    def select_action_i(self, i):
        
        if random.uniform(0, 1) < self.explore_rate:
            
            #self.state["Agent " + str(i)] = i 
            self.action["Agent " + str(i)] = {}

            for j in range(self.nAgents):
                
                if i != j:
                    
                    self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = self.action_options[i][np.random.choice(self.nOptions)]
                    
        else:
            

            self.action["Agent " + str(i)] = {}
            
            for j in range(self.nAgents): 

                if i != j:
                    
                    temp = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)], key=self.Q["Agent " + str(i)]["Neighbour " + str(j)].get)
                        
                    self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = float(temp.replace("Option ",""))

    #%% compute reward
    # ---------------                    
    # this can be called for 1 or multiple agents. Just ensure to pass in the state for 1 x agent if the former.
    
    def compute_reward(self, states_q, landmarks):
        
        # initialize reward signal and temp helpers
        self.reward = 0
        summerizer = 0.0001
        normalizer = 0.0001
        
        # for each agent 
        for i in range(states_q.shape[1]):
            
            # cycle through landmarks
            for j in range(landmarks.shape[1]):
                
                # accumulate distances between agents and landmarks
                summerizer += np.linalg.norm(states_q[0:3,i]-landmarks[0:3,j])
                normalizer += 1
        
        # compute reward signal
        self.reward = 1/np.divide(summerizer,normalizer) 
        
       # print("Reward signal: ", self.reward)
 
    #%% link to parameters used by controller
    # ---------------------------------------

    # global case
    def match_parameters(self,paramClass):
        
        # set controller parameters
        if paramClass.d_weighted.shape[1] != len(self.action):
            raise ValueError("Error! Mis-match in dimensions of controller and RL parameters")
        
        # for each control parameter (i.e. lattice lengths)
        for i in range(paramClass.d_weighted.shape[1]):
            
            # for each neighbour
            for j in range(len(self.action)): # this -1 needs to go
                
                # not itself
                if i != j:
                    
                    paramClass.d_weighted[i, j] = self.action["Agent " + str(i)]["Neighbour Action " + str(j)]        
    
    # local case 
    def match_parameters_i(self,paramClass, i): 
        
        # for each neighbour
        for j in range(len(self.action)):
            
            # load the neighbour action
            if i != j:
                
                paramClass.d_weighted[i, j] = self.action["Agent " + str(i)]["Neighbour Action " + str(j)]        
               
    #%% update q-table
    # ----------------

    # global case     
    def update_q_table(self):
                
        # for each agent
        for i in range(self.nAgents):
            
            # and its neighbour
            for j in range(self.nAgents):
                
                # not itself
                if i != j:
                
                    # update the q table with selected action
                    selected_option = self.action["Agent " + str(i)]["Neighbour Action " + str(j)]
                    
                    # we will use this same action for the discounted future rewards, but from the neighbour's perspective
                    future_option = self.action["Agent " + str(i)]["Neighbour Action " + str(j)] 
                    
                    #self.state = ["Agent " + str(i), "Neighbour " + str(j)]
                    #self.action = ["Option " + str(selected_option)]
                    
                    # Q(s,a)
                    Q_current = self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(selected_option)] 
                    
                    # Q(s+,a)
                    Q_future = self.Q["Agent " + str(j)]["Neighbour " + str(i)]["Option " + str(future_option)] # this needs to flip i/j eventually 
                    
                    #self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(selected_option)] += np.multiply(self.learn_rate, self.reward + self.discount*Q_future - Q_current)
                    self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(selected_option)] = (1 - self.learn_rate)*Q_current + self.learn_rate*(self.reward + self.discount*Q_future)
                    
        #print('Reward at ',  self.time_count, ' : ', self.reward)
        self.Q_update_count += 1
        
        self.data[self.Q_update_count] = copy.deepcopy(self.Q)
        
        if self.Q_update_count > 10:
            self.Q_update_count = 0
            
            data = convert_to_json_serializable(self.data)

            with open(file_path, 'w') as file:
                json.dump(data, file)
    
    # local case
    def update_q_table_i(self, i):
        
        for j in range(self.nAgents):
            
            if i != j:
            
                # update the q table with selected action
                selected_option = self.action["Agent " + str(i)]["Neighbour Action " + str(j)]
                
                # we will use this same action for the discounted future rewards, but from the neighbour's perspective
                future_option = self.action["Agent " + str(i)]["Neighbour Action " + str(j)] 
                
                #self.state = ["Agent " + str(i), "Neighbour " + str(j)]
                #self.action = ["Option " + str(selected_option)]
                
                # Q(s,a)
                Q_current = self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(selected_option)] 
                
                # Q(s+,a)
                Q_future = self.Q["Agent " + str(j)]["Neighbour " + str(i)]["Option " + str(future_option)] # this needs to flip i/j eventually 
                
                #self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(selected_option)] += np.multiply(self.learn_rate, self.reward + self.discount*Q_future - Q_current)
                self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(selected_option)] = (1 - self.learn_rate)*Q_current + self.learn_rate*(self.reward + self.discount*Q_future)
                    
        #print('Reward at ',  self.time_count, ' : ', self.reward)
        self.Q_update_count += 1
        
        self.data[self.Q_update_count] = copy.deepcopy(self.Q)
        
        if self.Q_update_count > 10*self.nAgents:
            self.Q_update_count = 0
            
            data = convert_to_json_serializable(self.data)

            with open(file_path, 'w') as file:
                json.dump(data, file)
  
     