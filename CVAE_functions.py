import pandas as pd
import numpy as np
from rdkit import Chem
import tensorflow as tf
from typing import *
from sklearn.metrics import *
from fingerprints import *

# ##############################################################################

def data(
        file_name:str, 
        sec_filename:str
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.core.series.Series]:
    """loads .csv files
    
    Uses pandas dataframe to load .csv files and retrieve SMILES, the corresponding 
    labels and predicted labels. Uses the ecfp() function from fingerprints.py to 
    featurize the molecules with ECFPs. Uses pandas dataframe to load .csv files 
    and retrieve important feature indices according to XGBoost. 

    Args:
        file_name:                  name of file with SMILES, labels and 
                                    predicted labels
        sec_filename:               name of file with important feature indices  
        
        

    Returns:
        Array of features (number of molecules, 1024), labels, predicted labels 
        and important feature indices

    """
    #read as a pandas dataframe 
    df = pd.read_csv(file_name)
    smiles = df['Smiles']
    y = df['Active/Inactive']
    y_pred = df['Predicted labels']
    a = pd.read_csv(sec_filename)
    scores = a['relevant ecfps']
    
    #retrieve the molecules from SMILES and featurize them with ECFPs
    mols = [Chem.MolFromSmiles(x) for x in smiles]   
    x = ecfp(mols)
    
    #return as arrays
    y_pred = np.array(y_pred, dtype=np.float32)
    y_pred = y_pred.reshape(-1,1)
    y = np.array(y, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    
    
    return x, y, y_pred, scores 

# -----------------------------------------------------------------------------#

def pick_out_scores(
        x: np.ndarray,
        scores: pd.core.series.Series
        ) -> np.ndarray:
    """Selects the features which are the most important according to XGBoost
    
    Uses indices obtained after using feature_importances_ from XGBoost to return 
    a feature vector consisting only of the most important feautres.
    
    Args:
        x:                  (number of molecules, 1024), array of features 
        scores:             indices of important features 



    Returns:
        new array with fewer features for the same molecules
    """
    
    #create result container
    x_new = []
    #iterate through the columns to retrieve the important features
    rows, columns = np.shape(x)
    for a in range(rows):
        b = [x[a, i] for i in scores]
        x_new.append(b)    
    x = np.array(x, dtype=np.float32)
    
    return x_new

# -----------------------------------------------------------------------------#

def create_dataset(
        x: np.ndarray,
        y: np.ndarray, 
        batch_size:int = 64 
        ) -> tf.data.Dataset:
    """Creates a TensorFlow Dataset to feed into the autoencoder
    
    Args:
        x:                  array of features 
        scores:             array of labels 

    

    Returns: 
        A TensorFlow Dataset consisitng of features and labels
    """
    
    #reshaping the arrays to fit into the dataset
    num_rows, num_columns = np.shape(x)
    x = np.reshape(x, (-1, num_columns))
    y = np.asarray(y, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.reshape(y, (len(y),1))
    
    #creating the dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    
    return dataset

# -----------------------------------------------------------------------------#

#functions for comparison of XGBoost with the autoencoder
def reconstruction_error(
        x: np.ndarray,
        x_pred: np.ndarray
        ) -> np.ndarray:
    """Calculates the reconstruction error 
    
    Feeds array into the autoencoder and calcualtes the reconstruction error
    
    Args:
        x:             array of features 
        y:             array of labels 

    Returns:
        Vector consisitng of reconstruction errors for each of the molecules
    """
    
    #create result container
    rec_error = []
    
    #calculate the reconstruction error for each row and create a vector 
    #of reconstruction errors
    bce = tf.keras.losses.binary_crossentropy
    for i in range(len(x)):
        a = bce(x[i], x_pred[i])
        rec_error.append(a)
    rec_error = np.array(rec_error)
   
    return rec_error

# -----------------------------------------------------------------------------#

def check_predictions(
        y_pred: np.ndarray,
        y: np.ndarray
        ) -> np.ndarray:
    """Compares XGBoost predictions with labels
    

    Args:
        y_pred:             array of predicted labels 
        y:                  array of labels 

    Returns:
        A binary array, with 1 meaning that the XGBoost classifier was wrong
    """
    #create result container
    false_labels = np.ones(len(y))
    
    #iterate through the labels  
    #if XGBoost predicted the wrong label, value = 1
    #if XGBoost predicted the wrong label, value = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            false_labels[i] = 0
        else:
            continue
    
    return false_labels

# -----------------------------------------------------------------------------#

#checking if there's a correlation
def score(
        false_labels: np.ndarray, 
        rec_error: np.ndarray
        ) -> float:
    """Checks whether a correlation exists between the reconstruction error and 
    XGBoost's false predictions
    
    Computes the ROC AUC between the two vecotrs. Both vectors must have the 
    same length. 
    
    Args:
        false_labels:               binary vector of false predictions
        rec_error:                  reconstruction error vector
        


    Returns:
        a ROC AUC value
    -------
    roc_auc : TYPE
        DESCRIPTION.

    """
    roc_auc = roc_auc_score(false_labels, rec_error)
    return roc_auc
