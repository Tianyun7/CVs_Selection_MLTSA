This is a repository related to a dissertation of a master degree in UCL. Copyright belongs to Tianyun QI.

The topic of this project is Combining Machine Learning and Molecular Dynamics Simulations for the Selection of Relevant Collective Variables.

CODE SCRIPT DESCRIPTION:

  CV_from_MD.py : The script used to calculate the distances between atoms and create collective variables (CVs).
  
  GPCR_CV_creation.py : The script used to generate CVs for the application of the machine learning transition state analysis (MLTSA).
  
  models.py : The script containing wrappers to easily train and test the machine learning models, and other functions useful for MLTSA.
  
  mltsa.py : The script applying MLTSA and calculating the accuracy drop for the Multilayer Perceptron (MLP) model.
  
  mltsa_GBDT.py : The script applying MLTSA and calculating the feature importance for the Gradient Boosting Decision Tree (GBDT) model.
  
  analyze_results_figures_GBDTMLP.ipynb : The script to analyse and visualise the results.
  
  PCA_GPCR.py and mltsa_PCA_GBDT.py : The scripts used for the PCA method in GBDT model. This part of work is is ongoing and was not included in the dissertation, but it's meaningful for the selection of relevant collective variables. As a result, the ode for this part will be updated later.
