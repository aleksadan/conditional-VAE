# conditional-VAE
A conditional variational autoencoder which analyses an XGBoost classifier

## Folder description
Python files containing "main" in their name are meant to be used as command line tools. 
Python files containing "functions" in their name contain helper functions to execute the scripts. 
[CVAE.py:](CVAE.py) contains the class for a conditional variational autoencoder. 

## File documentation
-[XGBoost_main.py:](XGBoost_main.py) Generates an XGBoost classifier and produces predictions.
-[CVAE_main.py:](CVAE_main.py) Generates a conditional variational autoencoder and analyses the predictions from the XGBoost model.
-[XGBoost_functions.py:](XGBoost_functions.py) Contains the functions used to prepare the data, generate and train the model, as well as get predicitons in [XGBoost_main.py:](XGBoost_main.py).
-[CVAE_functions.py:](CVAE_functions.py) Contains the functions used to prepare and analasye the data in [CVAE_main.py:](CVAE_main.py).
-[CVAE.py:](CVAE.py) Containes the contains the class for the conditional variational autoencoder. 
