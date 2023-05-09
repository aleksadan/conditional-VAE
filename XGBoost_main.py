"""
XGBoost Predictions script. 

Pipeline to predict whether a molecule is active or not using a XGBoost classifier 
with ECFP featurization.

Before running the script, make sure that you have the XGBoost_functions.py, since the 
functions from that script will be used for this analysis. Additionally, make sure that
you have the file.sdf from the screening. 
 
The output will be saved in three .csv files:
    1. test_xgb.csv
    2. val_xgb.csv
    3. scores_xgb.csv

Steps:
    1. Load primary screening data (change the property and threshold)
    2. Divide the molecules into active and inactive molecules  
    3. Split the data into a training, validation and training dataset 
    4. Calculate ECFPs for training set
    5. Train the XGBoost model
    6. Optimize the XGBoost model with HyperOpt
    7. Obtain label predictions and feature importance from the XGBoost model
    8. Save
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from XGBoost_functions import *


###############################################################################



def main():
    #load and divide the data  
    mols_active, mols_inactive = load_data('file.sdf', 'Property', threshold)
    
    #split the data and featurize with ECFP
    x_train, y_train, x_val, y_val,x_test, y_test, mols_test, mols_val = split(mols_active, mols_inactive)
    
    #create an XGBoost model using the default parameters
    default_auc, default_model = baseline(x_train, y_train, x_val, y_val)
    
    #optimize the XGBoost model
    best_params = optimize(x_train, y_train, x_val, y_val)
    
    #predict the labels and indices of the most important features according to XGBoost
    y_val_labeled, y_test_labeled, ind_scores = predictions(best_params, x_train, y_train, x_val, y_val, x_test, y_test)
    
    #save the validation and test datasets
    save_results(mols_test, y_test, y_test_labeled, 'test_xgb.csv')
    save_results(mols_val, y_val, y_val_labeled, 'val_xgb.csv')
    
    #save the indices of the important features
    save_scores(ind_scores, 'scores_xgb.csv')
    

if __name__ == "__main__":
    main()