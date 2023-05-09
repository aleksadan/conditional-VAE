import rdkit
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from typing import *
from fingerprints import *

###############################################################################

def load_data(
        filename: str, 
        prop: str, 
        threshold: int
        ) -> Tuple[List[rdkit.Chem.rdchem.Mol], List[rdkit.Chem.rdchem.Mol]]:
    """Loads the data and divides it into active and inactive molecules
    
    Uses a specific property measured in the primary screening and the corresponding 
    threshold to divide the molecules into active and inactive

    Args:
        filename:           name of file with the primary screening data
        prop:               name of the property  
        threshold:          value of the threshold                  
        
     
    
    Returns:
        Two lists of rdkit.Chem.rdchem.Mol, which contain active and inactive 
        molecules, respectively.
    

    """
    #create result containers
    mols_active = []
    mols_inactive = []
    
    #use a supplier to iterate through the .sdf file
    supply = Chem.SDMolSupplier(filename)
    for i in supply:
        try:
            mol = next(supply) 
        except:
            mol = None
        if mol is not None:
            #retrieve the value of the desired property of the molecule 
            try:
                props = mol.GetDoubleProp(prop)
                if props > threshold:
                    mols_active.append(mol)
                else:
                    mols_inactive.append(mol)
            except:
                pass
            
    return mols_active, mols_inactive

#-----------------------------------------------------------------------------#

def split(
        mols_active: List[rdkit.Chem.rdchem.Mol], 
        mols_inactive: List[rdkit.Chem.rdchem.Mol]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray, List[rdkit.Chem.rdchem.Mol], List[rdkit.Chem.rdchem.Mol]]:
    """Splits the data and featurizes it
    
    Uses train_test_split from sklearn to split the molecules into three datasets:
    train, validation and test. Uses a function from functions.py to featurize the 
    molecules with ECFPs. Retrieves SMILES for the validation and test dataset which will 
    be important at the end of the XGBoost_predicitions.py script when saving the 
    data from the XGBoost model.
    
    Args:
        mols_active:           list of active molecules
        mols_inactive:         list of inactive molecules   



    Returns:
        Train, validation and test datasets, with each consisting of an array 
        of features and an array of labels, and two lists of SMILES (for the validation 
        and test set)

    """
    #create a list of all molecules and their corresponding labels
    y_inactive_mols = [0]*len(mols_inactive)
    y_active = [1]*len(mols_active)
    mols = mols_active + mols_inactive 
    y = y_active + y_inactive_mols
    y = np.asarray(y)
    y = np.reshape(y, (len(y),1))  
    
    #create train, validation and test dataset
    x_train, x_test, y_train, y_test = train_test_split(mols, y, test_size=0.2, random_state=42, stratify = y)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify = y_test)
    
    #retrieve SMILES for the test and valisation dataset
    mols_test = [Chem.MolToSmiles(x) for x in x_test] 
    mols_val = [Chem.MolToSmiles(x) for x in x_val]
    
    #featurize the molecules 
    x_train = ecfp(x_train)
    x_val = ecfp(x_val)
    x_test = ecfp(x_test)
    
    #make sure the outputs are arrays of the same data-type
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    x_val = np.array(x_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    
    
    return x_train, y_train, x_val, y_val, x_test, y_test, mols_test, mols_val

#-----------------------------------------------------------------------------#

def baseline(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray) -> Tuple[int, XGBClassifier]:
   """Creates a default XGBoost model
        
   Creates a XGBoost classifier using default parameters and computes the 
   Area Under the Receiver Operating Characteristic Curve (ROC AUC) from the prediction scores.
        
   Args:
       x_train:           array of features from the training set
       y_train:           array of labels from the training set
       x_val:             array of features from the validation set
       y_val:             array of labels from the validation set
        
        

    Returns:
        The ROC AUC value from the prediciton scores and the default XGBoost classifier
   """

   #create the XGBoost model
   model = XGBClassifier(n_jobs=6) 
   
   #fit the XGBoost model on the training set
   model.fit(x_train, y_train,
             verbose=False)
   
   #retrieve predictions for the validation set
   y_pred = model.predict_proba(x_val)
   
   #compute the ROC AUC
   auc = roc_auc_score(y_val, y_pred[:,1])
   print('The ROC AUC of the default XGBoost model is: ' + str(auc))
   
   return auc, model
    


#-----------------------------------------------------------------------------#

def optimize(
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_val: np.ndarray, 
        y_val: np.ndarray
        ) -> Dict : 
    """Optimizes the hyperparameters for the XGBoost Classifier
    
    Uses HyperOpt and the average between the ROC AUC value and the average precision (AP)
    to find the best hyperparameters for the XGBoost classifier. 
   
    Args:
        x_train:           array of features from the training set
        y_train:           array of labels from the training set
        x_val:             array of features from the validation set
        y_val:             array of labels from the validation set
    


    Returns:
        The best hyperparameters for the XGBoost classifier
        
    """
    
    #initialize the range of hyperparameter values
    space={
           'max_depth': hp.quniform("max_depth", 1, 9, 1), 
           'gamma': hp.uniform ('gamma',1e-8, 1.0), 
           'eta' : hp.quniform('eta', 1e-8, 0.5, 0.025), 
           'reg_lambda' : hp.uniform('reg_lambda',  1e-8, 1.0), 
           'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1), 
    }
    
    #define the objective function
    def objective(args):
        args["max_depth"] = int(args["max_depth"]) 
        args["min_child_weight"] = int(args["min_child_weight"])
        model_opt = XGBClassifier(n_estimators = 300,
                                  scale_pos_weight = 40,
                                  n_jobs=12,
                                  **args)
        
        #fit the XGBoost model with the new hyperparameters on the training set
        evaluation = [( x_val, y_val)]
        model_opt.fit(x_train, y_train,
                eval_set=evaluation,
                early_stopping_rounds=30,
                verbose=False)
        
        #retrieve predictions for the validation set
        preds = model_opt.predict_proba(x_val)[:,1]
        
        #calculate the score with ROC AUC and AP 
        roc_auc = roc_auc_score(y_val, preds)
        precision = average_precision_score(y_val, preds) 
        score = (roc_auc + precision) / 2
        
        return 1 - score
        
    #use the fmin function from Hyperopt to find the best hyperparameters
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=30,
                trials=trials)
    print (best) 
    
    return best

#-----------------------------------------------------------------------------#

def predictions(
        best: Dict, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_val: np.ndarray, 
        y_val: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """Predicts the labels and finds important features 
    
    Uses the hyperparameters from optimize() to predict the labels for the validation
    and test datasets. Uses feature_importances_ from XGBoost to find the important features
    
    Args:
        x_train:           array of features from the training set
        y_train:           array of labels from the training set
        x_val:             array of features from the validation set
        y_val:             array of labels from the validation set
        x_test:            array of features from the test set
        y_test:            array of labels from the test set
        
        
    
    Returns:
        The predicted labels for the validation and test datasets as well as the 
        indices of the important features.

    """
    #create the XGBoost model using the optimized hyperparameters
    model_best = XGBClassifier(n_estimators = 300,
                               scale_pos_weight = 40,
                               n_jobs=12,
                            max_depth = int(best['max_depth']),
                            gamma = best['gamma'],
                            eta = best['eta'],
                            reg_lambda = best['reg_lambda'],
                            min_child_weight = int(best['min_child_weight']))
    
    #fit the optimized XGBoost model on the training set
    model_best.fit(x_train, y_train)
    
    #predict labels for the validation and test  set 
    y_val_labeled = model_best.predict(x_val)
    y_test_labeled = model_best.predict(x_test)
    
    #retrieve the importance of all the features according to XGBoost
    scores = model_best.feature_importances_
    
    #find the indices of the important features  
    ind = np.where(scores>0)
    
    return y_val_labeled, y_test_labeled, ind

#-----------------------------------------------------------------------------#

def save_scores(
        scores: np.ndarray, 
        filename: str
        ):
    """Saves the indices of important feature 
    
    Uses pandas to store the important feature indices in a .csv file
    

    Args:
        scores:           indices of important features 
        filename:         name of .csv file in which the results will be saved
        
    
    """
    scores = scores[0]
    
    #store in a pandas dataframe and save
    props = {'relevant ecfps': scores[:]}
    df = pd.DataFrame(data=props)
    df.to_csv(filename)  

#-----------------------------------------------------------------------------#

def save_results(
        smiles: List[str],
        y: np.ndarray, 
        y_labeled: np.ndarray,
        filename: str
        ):
    """
    
    Args:
        smiles:           list of SMILES of given the molecules
        y:                corresponding array of labels 
        y_labeled:        array of predicted labels from the XGBoost model
        filename:         name of .csv file in which the results will be saved

    """
    
    #make the labels one dimensional
    y = y.flatten()
    
    #store in a pandas dataframe and save
    props = {'Smiles': smiles[:],
             'Active/Inactive': y[:],
             'Predicted labels': y_labeled[:]}
    df = pd.DataFrame(data=props)
    df.to_csv(filename)






