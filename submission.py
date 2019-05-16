#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#update log 1.change queruy_file read method from re to a function.
#update 2019-04-23-23:50 *treat symbol error 
#@alexchan
"""
Cated on Tue Apr 23 17:57:16 2019

@author: jihuiliang
"""

import numpy as np
import pandas as pd
import re
import time

'''
State_File ='./toy_example/State_File'
Symbol_File='./toy_example/Symbol_File'
Query_File ='./toy_example/Query_File'
State_File ='./dev_set/State_File'
Symbol_File='./dev_set/Symbol_File'
Query_File ='./dev_set/Query_File'
'''

def classification_unk(s1,s2,s1_index):
    if s2=='/':
        if 'U' in s1:
            return 'UnitNumber'
        if 'Lot' in s1:
            return 'StreetNumber'
        if 'Shp' in s1:
            return 'CommercialUnitType'
    if s2=='-':
        if s1.isdigit():
            return 'StreetNumber'
    if s1_index==0 and s1.isupper() and s2.isupper():
            return 'EntityName' 

    return ''

def split_symbole(line):
    pattern=[',','&','-','/','(',')']
    L=line.split()
    symbole_list=[]
    
    for i in range(len(L)):
        s=''
        for j in range(len(L[i])):
            if L[i][j] in pattern:
                if s !='':
                    symbole_list.append(s)
                    s=''
               
                symbole_list.append(L[i][j])
            else:
                s+=L[i][j]
        if s !='':
            symbole_list.append(s)    
    return symbole_list

def viterbi_algorithm(State_File, Symbol_File, Query_File):
    state_list = []
    with open(State_File) as f:
        for line in f.readlines():
            state_list.append(line.strip())
    # print(state_list)

    num_states = int(state_list[0])
    states = state_list[1:num_states+1]
    state_matrix = np.zeros((num_states,num_states))

    for i in range(num_states+1,len(state_list)):
        m, n, k = state_list[i].split()
        m1 = int(m)
        n1 = int(n)
        k1 = int(k)
        state_matrix[m1][n1] = k1
    # print(state_matrix)

    #probability of transition between states
    state_prob = {}
    for i in range(0,num_states):
        state_prob[states[i]] = {} 
        for j in range(0,num_states):
            state_prob[states[i]][states[j]] = (state_matrix[i][j]+1)/(sum(state_matrix[i])+num_states-1)
            if states[i] == 'END' or states[j] == 'BEGIN':
                state_prob[states[i]][states[j]] = 0
            if states[i] == 'BEGIN' and states[j] == 'END':
                state_prob[states[i]][states[j]] = 0        
    # print(state_prob)

    #process the symbol file
    symbol_list = []
    with open(Symbol_File) as f:
        for line in f.readlines():
            symbol_list.append(line.strip())

    num_symbols = int(symbol_list[0])
    # print(num_symbols)
    symbols = symbol_list[1:num_symbols+1]

    #the prob for symbols in symbol_list
    sym_dic = {i:{} for i in states[:-2] }
    for i in range(num_symbols+1,len(symbol_list)):
        m, n, k = symbol_list[i].split()
        m1 = int(m)
        n1 = int(n)
        k1 = int(k)
        sym_dic[states[m1]][symbols[n1]] = k1
    # print(sym_dic)
    res = []
    #process the Query file
    #pattern = re.compile(r"[A-Za-z0-9.]+|[,&-/()]")
    q_list = []
    with open(Query_File) as f:
        for line in f.readlines():
            q_list = split_symbole(line)
            #print(q_list,"88888888888888")
            #to calculate the emission
            emission_prob = {}
            for i in states:
                emission_prob[i] = {}
                for j in q_list:
                    if i in sym_dic.keys():
                        if j in sym_dic[i].keys():
                            emission_prob[i][j] = (sym_dic[i][j]+1)/(sum(sym_dic[i].values())+len(symbols)+1)
                        else:
                            emission_prob[i][j] =1/(sum(sym_dic[i].values())+len(symbols)+1)
                    else:
                        emission_prob[i][j] = 0.0
                    # print(emission_prob)
            path = np.zeros((len(q_list),num_states))
            cur_prob = np.zeros((len(q_list),num_states))
            for i in states:  
                cur_prob[0][states.index(i)]=state_prob['BEGIN'][i]*emission_prob[i][q_list[0]]
        #     print(cur_prob)

            for i in range(1,len(q_list)):
                for cur_state in states:
                    for last_state in states:
                        value = cur_prob[i-1][states.index(last_state)]*state_prob[last_state][cur_state]*emission_prob[cur_state][q_list[i]]
                        if value >  cur_prob[i][states.index(cur_state)]:
                            cur_prob[i][states.index(cur_state)] = value
                            path[i][states.index(cur_state)] = states.index(last_state)

        #     print(path)
        #     print(cur_prob)
            state_before_last = np.argmax(cur_prob[-1,:])
            v = np.log(cur_prob[-1][state_before_last]*state_prob[states[state_before_last]]['END'])
            opt_list = []
            opt_list.append(states.index('END'))
            opt_list.append(state_before_last)
            n = len(path) - 1
            for i in range(len(path)-1,0,-1):
                t = int(opt_list[-1])
        #         print(t)
                opt_list.append(int(path[i][t]))
                n -= 1
            opt_list.append(states.index('BEGIN'))
            opt_list.reverse()
            opt_list.append(v)
            res.append(opt_list)
#             print(opt_list)
    return res

def top_k_viterbi(State_File, Symbol_File, Query_File, k):
    
    state_list = []
    with open(State_File) as f:
        for line in f.readlines():
            state_list.append(line.strip())
    # print(state_list)

    num_states = int(state_list[0])
    states = state_list[1:num_states+1]
    state_matrix = np.zeros((num_states,num_states))

    for i in range(num_states+1,len(state_list)):
        m, n, u = state_list[i].split()
        m1 = int(m)
        n1 = int(n)
        k1 = int(u)
        state_matrix[m1][n1] = k1
    # print(state_matrix)

    #probability of transition between states
    state_prob = {}
    for i in range(0,num_states):
        state_prob[states[i]] = {} 
        for j in range(0,num_states):
            state_prob[states[i]][states[j]] = (state_matrix[i][j]+1)/(sum(state_matrix[i])+num_states-1)
            if states[i] == 'END' or states[j] == 'BEGIN':
                state_prob[states[i]][states[j]] = 0
            if states[i] == 'BEGIN' and states[j] == 'END':
                state_prob[states[i]][states[j]] = 0        
    # print(state_prob)

    #process the symbol file
    symbol_list = []
    with open(Symbol_File) as f:
        for line in f.readlines():
            symbol_list.append(line.strip())

    num_symbols = int(symbol_list[0])
    # print(num_symbols)
    symbols = symbol_list[1:num_symbols+1]

    #the prob for symbols in symbol_list
    sym_dic = {i:{} for i in states[:-2] }
    for i in range(num_symbols+1,len(symbol_list)):
        m, n, u = symbol_list[i].split()
        m1 = int(m)
        n1 = int(n)
        k1 = int(u)
        sym_dic[states[m1]][symbols[n1]] = k1
    # print(sym_dic)
    res = []
    k = int(k)
    #process the Query file
    #pattern = re.compile(r"[A-Za-z0-9.]+|[,&-/()]")
    q_list = []
    with open(Query_File) as f:
        for line in f.readlines():
            q_list = split_symbole(line)
            #to calculate the emission
            emission_prob = {}
            for i in states:
                emission_prob[i] = {}
                for j in q_list:
                    if i in sym_dic.keys():
                        if j in sym_dic[i].keys():
                            emission_prob[i][j] = (sym_dic[i][j]+1)/(sum(sym_dic[i].values())+len(symbols)+1)
                        else:
                            emission_prob[i][j] =1/(sum(sym_dic[i].values())+len(symbols)+1)
                    else:
                        emission_prob[i][j] = 0.0
            '''
            
            we use two list separately store prob and path
            delta[T][S][K] 

            T is the length of observations
            S is the states without 'BEGIN' and 'END'
            K is num of top_k
            path[T][S] store k list, it is the path from the first to the state_before_last
            
            '''
            delta = []
            path = []
            
            for i in range(len(q_list)):
                delta.append({})
                path.append({})
            
            for i in states[:-2]:#begin->si
                delta[0][i] = [0.]*k 
                delta[0][i][0] = state_prob['BEGIN'][i]*emission_prob[i][q_list[0]]
                path[0][i] = [[states.index('BEGIN')]]*k
#             print(delta)
            '''
            loop to find the top k 
            list(map)+[0]*k is to avoid while there is not enough k results
            after sort and add
            put the result into the delta and path
            '''
            for t in range(1,len(q_list)):
                for i in states[:-2]:#cur_state
                    temp = [] # to store the prob and path
                    for j in states[:-2]:#last state
                        for idx in range(k):
                            p = delta[t-1][j][idx]*state_prob[j][i]*emission_prob[i][q_list[t]]
                            pa = path[t-1][j][idx]+[states.index(j)]
                            temp.append([p,pa])
                    temp = sorted(temp, key=lambda ele: (-ele[0],ele[1]))
    #                 print('-1-',temp)

                    delta[t][i] = list(map(lambda e: e[0],temp))+[0.]*k
                    path[t][i] = list(map(lambda e: e[1],temp))+[-1]*k
                    delta[t][i] = delta[t][i][0:k]
                    path[t][i] = path[t][i][0:k]
            '''

            p is the prb
            res_p is the path from 'begin' to hiden states to 'end'
            use opt to store the prob and path 
            and sort to find the top_k
            
            put the path and prob into 1 list and store to a result list
            '''
    #         print('-2-',delta)
            opt = []
            res_p = []
            for i in delta[-1]:
                for idx in range(k):
                    if delta[-1][i][idx] == 0:
                        break
                    p = np.log(delta[-1][i][idx]*state_prob[i]['END'])
                    res_p = path[-1][i][idx]+[states.index(i)]
                    res_p = res_p + [states.index('END')]
                    opt.append([p,res_p,idx])
            opt = sorted(opt,key = lambda e: (-e[0],e[1]))
    #         print('-3-',opt)
            for i in range(k):
                res.append(opt[i][1]+[opt[i][0]])
#                print(opt[i][1]+[opt[i][0]])
    return res




def viterbi_algorithm1(State_File, Symbol_File, Query_File):
    state_list = []
    with open(State_File) as f:
        for line in f.readlines():
            state_list.append(line.strip())
        #print(state_list)

    num_states = int(state_list[0])
    states = state_list[1:num_states+1]
    #print(states)
    state_matrix = np.zeros((num_states,num_states))

    for i in range(num_states+1,len(state_list)): # read state1 -state2 frequency
        m, n, k = state_list[i].split()
        m1 = int(m)
        n1 = int(n)
        k1 = int(k)
        state_matrix[m1][n1] = k1
    #print(state_matrix)

    #probability of transition between states
    state_prob = {}
    for i in range(0,num_states):
        state_prob[states[i]] = {} 
        for j in range(0,num_states):
            state_prob[states[i]][states[j]] = (state_matrix[i][j]*1.18+1)/(sum(state_matrix[i])*1.18+num_states-1)
            if states[i] == 'END' or states[j] == 'BEGIN':
                state_prob[states[i]][states[j]] = 0
##            if states[i] == 'BEGIN' and states[j] == 'END':
##                state_prob[states[i]][states[j]] = 0        
   # print(state_prob)

    #process the symbol file
    symbol_list = []
    with open(Symbol_File) as f:
        for line in f.readlines():
            symbol_list.append(line.strip())

    num_symbols = int(symbol_list[0])
    # print(num_symbols)
    symbols = symbol_list[1:num_symbols+1]

    #the prob for symbols in symbol_list
    sym_dic = {i:{} for i in states[:-2] }
    for i in range(num_symbols+1,len(symbol_list)):
        m, n, k = symbol_list[i].split()
        m1 = int(m)
        n1 = int(n)
        k1 = int(k)
        sym_dic[states[m1]][symbols[n1]] = k1
    #print(sym_dic)

        
    res = []
    prb = []
    #process the Query file
    #pattern = re.compile(r"[A-Za-z0-9.]+|[,&-/()]")
    q_list = []
    with open(Query_File) as f:
        for line in f.readlines(): #each time treat one query
            q_list = split_symbole(line)
            #to calculate the emission
            emission_prob = {}
            for state in states:
                emission_prob[state] = {}
            for state in states:
                #print
                for j in range(len(q_list)):
                    if state in sym_dic.keys():
                        if q_list[j] in sym_dic[state].keys():
                            emission_prob[state][q_list[j]] = (sym_dic[state][q_list[j]]*1.2+1)/(sum(sym_dic[state].values())*1.2+(len(symbols)+1))
                        else:
##                            if j==0:
##                                for S_0 in ['Postcode',',','/','-','(',')','&','LevelName']:
##                                    emission_prob[S_0][q_list[j]] = 0
                            if j==0 and q_list[j].isalpha():
                                emission_prob['UnitNumber'][q_list[j]] = 0
                                emission_prob['StreetNumber'][q_list[j]] = 0
                                emission_prob['SubNumber'][q_list[j]] = 0
                                emission_prob['Postcode'][q_list[j]] = 0
                                                                
                            elif j<len(q_list)-1:
                                unk_type=classification_unk(q_list[j],q_list[j+1],j)
                                if unk_type !='':
                                    #print(emission_prob)
                                    emission_prob[unk_type][q_list[j]] = 1
                            emission_prob[state][q_list[j]] = 0.002/(sum(sym_dic[state].values())+(len(symbols)+1))       
                    else: #begin and end state
                         emission_prob[state][q_list[j]] = 0.0
            #print("------------=",emission_prob)
            path = np.zeros((len(q_list),num_states))
            cur_prob = np.zeros((len(q_list),num_states))
            #inital probability
            for i in states:
                cur_prob[0][states.index(i)]=state_prob['BEGIN'][i]*emission_prob[i][q_list[0]]
                
                #cur_prob[0][states.index(i)]=(1/num_states)*emission_prob[i][q_list[0]]
        #     print(cur_prob)

            for i in range(1,len(q_list)):
                for cur_state in states:
                    for last_state in states:
                        value = cur_prob[i-1][states.index(last_state)]*state_prob[last_state][cur_state]*emission_prob[cur_state][q_list[i]]
                        if value >  cur_prob[i][states.index(cur_state)]:
                            cur_prob[i][states.index(cur_state)] = value
                            path[i][states.index(cur_state)] = states.index(last_state)

        #     print(path)
        #     print(cur_prob)
            state_before_last = np.argmax(cur_prob[-1,:])
            v = np.log(cur_prob[-1][state_before_last]*state_prob[states[state_before_last]]['END'])
            opt_list = []
            opt_list.append(states.index('END'))
            opt_list.append(state_before_last)
            n = len(path) - 1
            for i in range(len(path)-1,0,-1):
                t = int(opt_list[-1])
        #         print(t)
                opt_list.append(int(path[i][t]))
                n -= 1
            opt_list.append(states.index('BEGIN'))
            opt_list.reverse()
            opt_list.append(v)
            res.append(opt_list)
            prb.append(v)
            #print(opt_list)
    return res

def advanced_decoding(State_File, Symbol_File, Query_File): #, do not change the heading of the function
    
    res=viterbi_algorithm1(State_File, Symbol_File, Query_File)
    return res


'''
a=top_k_viterbi(State_File, Symbol_File, Query_File, 4)
for i in range(len(a)):
    print(a[i])

a=viterbi_algorithm(State_File, Symbol_File, Query_File)
for i in range(len(a)):
    print(a[i])
'''    

