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
import os
import json
import copy
from scipy.spatial import distance


#%% hyper parameters
# ----------------
options_range   = [2, 8]
nOptions       = 3         # number of action options
time_horizon    = 250        # how long to apply action and away reward (1 sec/0.02 sample per sec = 50), also speed 
time_horizon_v  = 0.5        # max speed to transition learning (makes learnig more stable)

data_directory = 'Data'
#file_path = os.path.join(data_directory, f"data_{formatted_date}.json")
file_path = os.path.join(data_directory, "data_Q.json")

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
        
        # initialize parameters
        self.nAgents       = nAgents
        self.nOptions       = nOptions
        #self.action_options = np.linspace(options_range[0], options_range[1], nAction)
        self.action_options = {state: np.linspace(options_range[0], options_range[1], self.nOptions) for state in range(self.nAgents)}
        self.explore_rate   = 1     # 1 = always learn, 0 = always exploit best option
        self.learn_rate     = 0.5   # alpha/lambda
        self.discount       = 0.8   # balance immediate/future rewards, (gamma): 0.8 to 0.99
        self.time_horizon   = time_horizon
        self.time_horizon_v   = time_horizon_v
        self.time_count     = 0
        self.Q_update_count = 0
        
        # for individual timecounts
        self.time_count_i = np.zeros((nAgents))
        self.Q_update_count_i = np.zeros((nAgents))
        

        self.data           = {}
        
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
        #self.nAction        = self.nAgents-1 * self.nOptions
        self.nAction        = self.nAgents * self.nOptions
        self.reward         = 0
        self.Q              = {}
        
        # for i in range(self.nAgents):
        #     self.Q["Agent " + str(i)] = {}
        #     for j in range(self.nAgents-1):
        #         self.Q["Agent " + str(i)]["Neighbour " + str(j)] = {}
        #         for k in range(self.nOptions):
        #             option_label = self.action_options[i][k]
        #             self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(option_label)] = 0
       
        for i in range(self.nAgents):
            self.Q["Agent " + str(i)] = {}
            for j in range(self.nAgents):
                self.Q["Agent " + str(i)]["Neighbour " + str(j)] = {}
                # if not itself
                if i != j:
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
                # for j in range(self.nAgents-1): 
                #     self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = self.action_options[i][np.random.choice(self.nOptions)]
                for j in range(self.nAgents):
                    if i != j:
                        self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = self.action_options[i][np.random.choice(self.nOptions)]
                    
        else:
            
            # test
            #self.Q["Agent 9"]["Neighbour 0"]["Option " + str(self.action_options[9][0])] = 999
            
            # Exploit (select best)
            for i in range(self.nAgents):
                self.action["Agent " + str(i)] = {}
                # for j in range(self.nAgents-1): 
                #     # get the max value
                #     #self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)].values())
                #     self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)], key=self.Q["Agent " + str(i)]["Neighbour " + str(j)].get)
                for j in range(self.nAgents): 
                    # get the max value
                    #self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)].values())
                    if i != j:
                        temp = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)], key=self.Q["Agent " + str(i)]["Neighbour " + str(j)].get)
                        
                        self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = float(temp.replace("Option ",""))
                        
            # print
            #print("Q table:", self.Q)
                        

    def compute_reward(self, states_q, landmarks):
        
        # test: just want to learn the maximum separation for now
        self.reward = 0
        summerizer = 0.0001
        normalizer = 0.0001
        
        for i in range(states_q.shape[1]):
            for j in range(landmarks.shape[1]):
                summerizer += np.linalg.norm(states_q[0:3,i]-landmarks[0:3,j])
                normalizer += 1
        self.reward = 1/np.divide(summerizer,normalizer) 
        
        print("Reward signal: ", self.reward)
 
        #distances = distance.pdist(states_q.transpose())
        #distance_matrix = distance.squareform(distances)
        #max_distance = np.max(distance_matrix)
        
        #if max_distance < 10:
        #    print(max_distance)
        
        #self.reward = max_distance


    

    def match_parameters(self,paramClass):
        
        # set controller parameters
        if paramClass.d_weighted.shape[1] != len(self.action):
            raise ValueError("Error! Mis-match in dimensions of controller and RL parameters")
        # for each control parameter (i.e. lattice lengths)
        
        # note: I need to ensure the d_weighted line up wll with actions... mismatch?

        for i in range(paramClass.d_weighted.shape[1]):
            # for each neighbour
            # for j in range(len(self.action)-1): # this -1 needs to go
            #     # load the neighbour action
            #     paramClass.d_weighted[i, j] = self.action["Agent " + str(i)]["Neighbour Action " + str(j)]
            for j in range(len(self.action)): # this -1 needs to go
                # load the neighbour action
                if i != j:
                    paramClass.d_weighted[i, j] = self.action["Agent " + str(i)]["Neighbour Action " + str(j)]        
                    #paramClass.d_weighted[i, j] = float(self.action["Agent " + str(i)]["Neighbour Action " + str(j)].replace("Option ",""))
        
    def update_q_table(self):
        
        # note: for the "next state", we will use the "invereted" actions (i.e. what if the neighbour used the select states). 
        #   this assseses the future rewards of selecting, as there is a larger consensus at play.
        #   what about max rewards
        
        # for each agent
        for i in range(self.nAgents):
            # and its neighbour
            #for j in range(self.nAgents-1):
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
        
        if self.Q_update_count > 10:
            self.Q_update_count = 0
            
            data = convert_to_json_serializable(self.data)

            with open(file_path, 'w') as file:
                json.dump(data, file)

            
  
        
        #self.next_action = np.argmax(self.Q[self.next_state,:])
        #self.Q[self.state, self.action] += np.multiply(self.learn_rate, self.reward + self.discount*self.Q[self.next_state, self.next_action] - self.Q[self.state, self.action])
        
        
#%% test

# nState = each agent has 1 state, a vector of control parameters
# nAction = each agent has nAgent-1 actions, the control parameter corresponding to each neighbour
# the Q-table keeps track of all these, so it has 1 "meta state" and nAgent * nAction-1 "meta actions"

# here, we input nState = number of agents, nAction = number of actions to choose from 

#nAgents = 10
#learning_agent = q_learning_agent(nAgents)   
#test = learning_agent.action 

# initialize state-action
#np.zeros((len(self.params),len(self.params))) 

# randomly select indicies         
#[random.randint(0, nAction) for _ in range(nState)]          
            
        
        