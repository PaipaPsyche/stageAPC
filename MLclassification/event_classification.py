# IMPORTS
#utils
import numpy as np
import pandas as pd
import pickle as pickle

#learning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



# ERROR ASSESSMENT FUNCITON
#given the real categories and the predicted probabilities of being a proton shower(PS)
#finds a threshold value to classify probability events that optimizes the proton detection
#and mantains the photon misclassification rate under a certain value

# y_test   real classes
# y_pred  predicted classes
# proton_tol  tolerance of miclassificated PS
# photon_loss_tol  tolerance for misclassifed photons
# resol resolution in the probability space, number of steps steps is defined as 1/resol

def error_assessment(y_test,y_pred,proton_tol=1E-4,photon_loss_tol=0.5,resol = 1E-4):
    # number of steps
    steps = int(1/resol)+1
    thresholds = np.linspace(0,1,steps)


    protons_passed = [] # values of (1-recall) . proportion of protons misclassificated
    photons_npassed = [] # values of (1-specificity). proportion of photons misclassificated

    # better threshold : [threshold value, 1-recall , 1 - spec.]
    ideal_th = [0,1,1]

    # is ther at least one candidate for threshold?
    found  = False


    for th in thresholds:
        # create an array with the classes decided with thresshold = th
        # if probability above th, then the class is 1. if else , 0.
        y_c = [1  if  v > th else 0 for v in y_pred.reshape(-1)]

        #true negatives, false positives, false negatives, true positives
        tn, fp, fn, tp = confusion_matrix(y_test.reshape(-1),y_c).ravel()


        pr_pass = fn/(fn+tp) # 1 -recall
        ph_pass = fp/(tn+fp) # 1 - specificity

        # the better threshold is only updated if tolerance values are respected and the candidate minimizes
        # at least one value (1-recall) or (1-specificity)
        if( ph_pass<photon_loss_tol and pr_pass < proton_tol and (ph_pass<ideal_th[2] or pr_pass < ideal_th[1])):
            ideal_th = [th,pr_pass,ph_pass]
            found = True

        protons_passed.append(pr_pass)
        photons_npassed.append(ph_pass)

    # if there is not at least one candidate
    if(not found):
        print("Warning! No minimal threshold conditions found in all the threshold sweep")
    else:
        print("The best threshold is in {} where {}% of protons are misclassified and {}% of photons are miscalssified".format(ideal_th[0],round(ideal_th[1]*100,2),round(ideal_th[2]*100,2)))
    return {
        "thresholds":thresholds,
        "protons_accepted":protons_passed,
        "photons_rejected":photons_npassed,
        "best_threshold":ideal_th
    }



#LOGISTIC PROBABILITY THRESHOLD MODEL
#Object that uses a Logisitc Regression CV in its core to generate a classifier and
#the optimal threshold value for classfying future events

class Logistic_probability_threshold:
    def __init__(self , **params):
        self.model = LogisticRegressionCV(**params) # model used
        self.threshold = 0.5 #threshold value initialized to 0.5 default
        self.trained = False #tells if model was already trained



    def fit_model(self , xx , yy , test_s=0.3):

        #splitting in train and test sample. the proportion of the test sample is set on "test_size"
        x_train,x_test,y_train,y_test = train_test_split(xx,yy,test_size=test_s)
        #train the model
        self.model.fit(x_train,y_train)

        #get the probability of classification of the second class (positive case: 1, associated with a proton shower)
        y_pred = self.model.predict_proba(x_test)[:,1]

        #used error assesment function to find the threshold
        err_ass =  error_assessment(y_test,y_pred,resol=2E-4)
        # threshold
        ideal_th = err_ass["best_threshold"][0]


        if(1>ideal_th and ideal_th>0):
            print("Optimal threshold set at {}".format(ideal_th))
            self.threshold = ideal_th
        else:
            print("Optimal threshold not found, 0.5 assigned as default")
            self.threshold = 0.5

        self.trained = True


        ## for plotting
        #thresholds = err_ass["thresholds"]
        #protons_passed = err_ass["protons_accepted"]
        #photons_npassed = err_ass["photons_rejected"]




    def predict(self,xx):
        if(self.trained):
            pred = self.model.predict_proba(xx)[:,1]
            return np.array([1 if p >= self.threshold else 0 for p in pred ])
        else:
            print("Model needs to be sucessfully trained first")


    def evaluate(self,xx,yy):
        y_pred = self.predict(xx)
        tn, fp, fn, tp = confusion_matrix(yy.reshape(-1),y_pred).ravel()


        pr_pass = fn/(fn+tp) # 1 -recall
        ph_pass = fp/(tn+fp) # 1 - specificity
        print("{}% of protons are misclassified and {}% of photons are miscalssified".format(round(pr_pass*100,2),round(ph_pass*100,2)))
        acc = np.sum(y_pred == yy)/len(yy)
        print("Total accuracy {}%".format(acc))
        return [acc,pr_pass,ph_pass]

    def give_coefs(self):

        if(self.trained):
            list_coef = [self.threshold,self.model.intercept_[0]]
            list_coef = list_coef + list(self.model.coef_[0])
            return list_coef
        else:
            print("Model needs to be sucessfully trained first")

    def save_pkl(self,filename):
        with open(filename,"wb") as f:
            pkl.dump(self,f)
