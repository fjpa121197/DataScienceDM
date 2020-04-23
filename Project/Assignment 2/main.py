# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:05:51 2020

@author: Francisco Parrilla
"""
"""
The following code is for assignment 2. All the UCT code provided can be found in uct_func_class.py file, slight 
modification has been made in that file, please refer to it if needed.

This file consists of many iterations using a decision tree classifier to take an action to which child node expand,
it does not replace the MCTS algorithm at any point, it just "helps" which chiild node to expand.

"""
import pandas as pd
from sklearn import tree
from datetime import datetime
import uct_func_class as uct
startTime = datetime.now()


if __name__ == "__main__":
    
    models = []  #Where the classifier are going to be stored
    data_tree = []# Where the data from each game is going to be stored
    mcts = [] #Where the games score using only MCTS default will be stored
    tree_tree = [] #Where the games score using MCTS Tree will be stored
    
    #The game will be played for 500 iterations
    for idz,x in enumerate(range(501)):
        
        print("Iteration %s" %(idz + 1))
        #Load data that was collected for assignment 1
        if idz == 0:
            game_data = pd.read_csv(r"data_t0.csv")
        else:
            game_data = data_tree[idz - 1]
        
        #Separate x values and y values from the last game that will be used to train a classifier
        x_values = game_data.drop(["player_action"],axis = 1)
        y_values = game_data["player_action"]
        
        #Every 10 iteratorions, the last classifier will be tested, so player 1 will have last classifier and player 2
        #will be assigned classifier 1-9.
        check = idz - 10
        if idz%10 == 0 and idz != 0:
            
            iteration = int(idz / 10)
            wins_mcts = ties_mcts = 0 
            
            for idm,m in enumerate(range(10)):
    
                wins_tree_vs_tree = loss_tree_vs_tree = ties_tree_vs_tree = 0
                #print("===================================================")
                #print("Player 1 using classifer %s  and player 2 using classifier %s" %(idz, idm + check + 1))
                #print("50 games being played")
                for game in (range(50)):
                    
                    #Check is created so the last classifier plays against the previous 10 classifiers
                    _,winner_tree= uct.UCTPlayGame(models[-1],models[idm + check]) #Call UCTPlayGame and use models
                    
                    #Check if there were wins and who won for winner_tree
                    if winner_tree == 2:
                        loss_tree_vs_tree += 1
                    if winner_tree == 1:
                        wins_tree_vs_tree += 1
                    if winner_tree == 0:
                        ties_tree_vs_tree += 1
                        
                    _,winner_mcts = uct.UCTPlayGame(None,None) #Call UCTPlayGame using no models
                    
                    #Check if there were wins and ties for winner_mcts    
                    if winner_mcts == 2 or winner_mcts == 1:
                        wins_mcts += 1
                    else:
                        ties_mcts += 1
                        
                #print("Player 1 won %s games, player 2 won %s games, there were %s ties" %(wins_tree_vs_tree,loss_tree_vs_tree,ties_tree_vs_tree))
                tree_tree.append((idz,idm+check+1,wins_tree_vs_tree,loss_tree_vs_tree,ties_tree_vs_tree)) #Append results
            mcts.append((iteration,wins_mcts,ties_mcts)) #Append results
        
        #We want to run the analysis for the classifier 500, but not append to our list of models
        if idz == 500:
            break
        
        clf = tree.DecisionTreeClassifier() #Define classifier (decision tree)
        clf.fit(x_values,y_values) #Train classifier with data from previous game
        models.append(clf)    #Save classifier to models list
        data=[] #Where the data from current games will be stored
            
        #Loop to generate data using same models
        for y in range(300):
          game,_= uct.UCTPlayGame(clf,clf) #Game using same models
          #Append data from current games to list
          for idx,x in enumerate(game):
            data.append(game[idx])
                
            
        #Create dataframe with data from the last 300 games    
        data_df = pd.DataFrame(data, columns = ["cell_0", "cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "player", "player_action"])
        data_tree.append(data_df) #Append dataframe to list


    #Test the last model (model 500) vs all the saved models for last time
    best_clf = models[-1] #Get last model from list
    opponents = models[0:-1] #Rest of the models, all but the last one
    score = [] #Where data will be stored
    
    #Loop through all opponents
    for idx,clf in enumerate(opponents):
        wins = loss = ties = 0
        
        #Play 50 games for each classifier
        for game in range(50):
              _,winner_tree= uct.UCTPlayGame(best_clf,opponents[-idx]) #Player 1 (Last classifier), player 2 opponents
             
              #Record scores
              if winner_tree == 1:
                  wins += 1
              elif winner_tree == 2:
                  loss +=1
              else:
                  ties += 1
        
        #Append score to list
        score.append((500,499-idx,wins,loss,ties))         
    
    #Save to csv files all the scores lists
    results_all = pd.DataFrame(score,columns = ['classifier best','opponent', 'wins','loss','ties'])        
    results_all.index += 1
    results_all.to_csv("results_all.csv",index = False)
    results_mcts = pd.DataFrame(mcts,columns = ['iteration','wins','ties'])
    results_mcts.index += 1 
    results_mcts.to_csv("results_mcts.csv",index = False)
    results_tree_tree = pd.DataFrame(tree_tree,columns = ['latest classifier','opponent classifier','wins','loss','ties'])
    results_tree_tree.index += 1 
    results_tree_tree.to_csv("results_tree_tree.csv",index = False)

    data_example = pd.DataFrame()
    
    for data in data_tree:
        data_example = data_example.append(data,ignore_index = True)
        
    data_example.to_csv("data_classifiers_example.csv",index = False)   
        
    print(datetime.now() - startTime)
    
