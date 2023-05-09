"""
Executing the Conditional Variational Autoencoder script. 

Pipeline to evaluate the accuracy of the XGBoost classifier using a conditional variational  
autoencoder (CVAE). The reconstruction error of the autoencoder is compared with a binary vector
which takes the value = 1 every time the classifier made a mistake. 

Before running the script, make sure that you have the CVAE_functions.py and CVAE.py and
files, since the functions from that script will be used for this analysis. Additionally, 
make sure that you have the 'val_xgb.csv', 'test_xgb.csv' and 'scores_xgb.csv' files created 
in XGBoost_predictions.py. 
 

Steps:
    1. Load the validation and test datasets
    2. Decrease the number of features in the x arrays according to which features 
       XGBoost considered important  
    3. Create a dataset for training the autoencoder
    4. Create a CVAE
    5. Calculate the reconstruction error vector
    6. Determine where the XGBoost model made mistakes in label predictions with a 
       vector called false_labels
    7. Compare the reconstruction error vector with the vector false_labels
"""

import pandas as pd
import numpy as np
from rdkit import Chem
import tensorflow as tf
from typing import *
from CVAE import *
from CVAE_functions import *


def main():
        
    #loading the data
    x_test, y_test, y_test_pred, scores = data('test_xgb.csv', 'scores_xgb.csv')
    x_val, y_val, y_val_pred, scores = data('val_xgb.csv','scores_xgb.csv')
    
    #reducing the number of features
    x_val= pick_out_scores(x_val, scores)
    x_test= pick_out_scores(x_test, scores)
    
    #creating a dataset for training
    data_val = create_dataset(x_val, y_val_pred)
    
    dimensions = len(scores)
    
    #creating, compliling and fitting the model
    autoencoder = condVAE(dimensions)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(data_val,
                    epochs=100,
                    shuffle=True,
                    )
    
    #reconstruct the test dataset using the built autoencoder
    x_pred = autoencoder((x_test, y_test_pred))
    
    #calculating the reconstruction error vector
    rec_error = reconstruction_error(x_test, x_pred) 
    
    #checking where the XGBoost model was wrong
    false_labels = check_predictions(y_test_pred, y_test)
    
    #checking if a correlation between the reconstruction error and false_labels exists 
    roc_auc = score(false_labels, rec_error)
    
    print(f'The ROC AUC is {roc_auc}')

if __name__ == "__main__":
    main()
