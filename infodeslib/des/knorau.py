# Author: Firuz Juraev <f.i.juraev@gmail.com>

import numpy as np 
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics import accuracy_score 
from sklearn.decomposition import PCA 
from matplotlib import pyplot as plt  

from infodeslib.des.utils import * 


class KNORAU: 
    """
    Late Fusion version of k-Nearest Oracles Union (KNORA-U). 
    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base classifiers. 
    DFP : Boolean (Default = False)
        Determines if the dynamic frienemy pruning is applied. 
        
    """
    
    def __init__(self, pool_classifiers=None, feature_subsets=None, 
                 k=7, DFP=False, plot=False, X_dsel=None, y_dsel=None, colors=None): 
        
        self.pool_classifiers = pool_classifiers 
        self.feature_subsets  = feature_subsets
        self.k                = k 
        self.DFP              = DFP
        self.plot             = plot
        self.X_dsel           = X_dsel 
        self.y_dsel           = y_dsel 
        self.colors           = colors 
         
     
    """
    Counting how many different classes in region of competence (RoC)
    If the number of classes in RoC, this RoC is indecision region 
    """
    def get_indecision_region(self, X, y):   
        num_classes = len(set(y.tolist()))  
        if self.plot: 
            print("Num classes: {}".format(num_classes))

        if num_classes > 1: 
            return True 
        else: 
            return False
    
    """
    For the given new test example, the funtion returns k neighbors
    
    """
    def plot_region(self, X, y, query):
        pca = PCA(n_components = 2) 
        pca.fit(self.X_dsel) 
        
        dsel_pca =  pca.transform(self.X_dsel) 
        neighbors_pca = pca.transform(X) 
        query_pca = pca.transform(query) 
        
        plt.scatter(dsel_pca[:, 0], dsel_pca[:, 1], c = self.y_dsel.map(self.colors), marker='.', label=self.colors)  
        plt.scatter(neighbors_pca[:, 0], neighbors_pca[:, 1], c = y.map(self.colors), marker='o', s=50) 
        plt.scatter(query_pca[:, 0], query_pca[:, 1], marker='X', s=50) 
        plt.show() 
    
    
    """
    For the given new test example, the funtion returns k neighbors
    
    """
    def get_region_of_competence(self, query): 
        nbrs    = NearestNeighbors(n_neighbors=self.k, metric="minkowski").fit(self.X_dsel) 
        indices = nbrs.kneighbors(query.values.reshape(1, -1), return_distance=False)   
        
        indecision_region = self.get_indecision_region(self.X_dsel.iloc[indices[0]], self.y_dsel.iloc[indices[0]]) 
        
        return self.X_dsel.iloc[indices[0]], self.y_dsel.iloc[indices[0]], indecision_region 
    
    
    """
    The competence of the base classifiers is simply estimated as the
    number of samples in the region of competence that it
    correctly classified.
    """ 
    def estimate_competence(self, roc, roc_labels):  
        num_modalities = len(self.feature_subsets)

        competence_acc_list = [] 
        average_correct_probs_list = [] 

        for i in range(num_modalities): 
            probability = self.pool_classifiers[i].predict_proba(roc[self.feature_subsets[i]]) 
            preds = np.argmax(probability, axis=1)
            probs = np.max(probability, axis=1)
            average_correct_probs = 0 

            corrects = 0 

            for j in range(len(roc_labels)): 
                if preds[j] == roc_labels.iloc[j]: 
                    average_correct_probs += probs[j]
                    corrects = corrects + 1 

            accuracy = round(accuracy_score(roc_labels, preds), 3)
            
            if corrects > 0: 
                average_correct_probs = round(average_correct_probs / corrects, 3) 

            competence_acc_list.append(accuracy)
            average_correct_probs_list.append(average_correct_probs) 
            
            if self.plot: 
                print("Model {} Accuracy: {} (Conf: {})".format(i, accuracy, average_correct_probs))
            
        self.correct_probs = average_correct_probs_list 
        
        return competence_acc_list 
    
    
    """
    If the query is in indecision region, we estimate the competence differently
    If there 3 classes in indecision region, the classifiers that classify at least one sample 
    in each class correctly will be selected 
    """    
    def estimate_indecision_competence(self, roc, roc_labels): 
        num_modalities = len(self.feature_subsets) 
        classes = set(roc_labels.tolist())  

        competence_acc_list = []  
        average_correct_probs_list = []  

        rocs_by_classes = [] 
        roc['target'] = roc_labels     

        for c in classes: 
            rocs_by_classes.append(roc[roc['target'] == c][:])         

        for i in range(num_modalities):        
            corrects_list = [] 
            for iroc in rocs_by_classes:
                corrects = 0 
                probability = self.pool_classifiers[i].predict_proba(iroc[self.feature_subsets[i]])  
                preds = np.argmax(probability, axis=1)
                probs = np.max(probability, axis=1)
                average_correct_probs = 0 

                for j in range(len(iroc['target'])): 
                    if preds[j] == iroc['target'].iloc[j]: 
                        average_correct_probs += probs[j]
                        corrects = corrects + 1 

                corrects_list.append(corrects) 

            if 0 not in corrects_list: 
                preds = self.pool_classifiers[i].predict(roc[self.feature_subsets[i]])
                probability = self.pool_classifiers[i].predict_proba(roc[self.feature_subsets[i]])   
                probs = np.max(probability, axis=1) 

                accuracy = round(accuracy_score(roc_labels, preds), 3) 
                competence_acc_list.append(accuracy)

                average_correct_probs = round(sum(probs) / sum(probs), 3)  
                average_correct_probs_list.append(average_correct_probs) 

            else: 
                competence_acc_list.append(0) 
                average_correct_probs_list.append(0) 
                
            
            self.correct_probs = average_correct_probs_list
            
        return competence_acc_list
    
    
    """
    Select the base classifiers for the classification of the query sample. 
     
    """ 
    def select(self, competences):    
        selected_models_indices = [] 
        weights_list            = [] 

        while(len(selected_models_indices) <= 0): 
            for i in range(len(competences)): 
                if competences[i] > 0: 
                    selected_models_indices.append(i) 
                    weights_list.append(competences[i])

            if len(selected_models_indices) == 0: 
                for i in range(len(competences)):
                    selected_models_indices.append(i)
                    weights_list.append(1) 

        
        if self.plot: 
            print("Selected models indices: {} (weights: {})".format(selected_models_indices, weights_list))

        self.selected_models_indices = selected_models_indices   
        self.selected_models_weights = weights_list  
    
    
    """
    Gives final prediction.  
     
    """  
    def predict(self, query, y_true): 
        predictions = [] 
        
        # 1) define region of competence 
        roc, y_roc, indecision = self.get_region_of_competence(query)
        
        # plot option
        if self.plot:  
            self.plot_region(roc, y_roc, query) 
        
        # 2) estimate competence 
        if self.DFP: 
            if indecision: 
                competences = self.estimate_indecision_competence(roc, y_roc)  
            else: 
                competences = self.estimate_competence(roc, y_roc) 
        else: 
            competences = self.estimate_competence(roc, y_roc)
        
        # 3) select models 
        self.select(competences) 
        
        # 4) predict 
        for j in range(len(self.selected_models_indices)): 
            pred = self.pool_classifiers[self.selected_models_indices[j]].predict_proba(query[self.feature_subsets[self.selected_models_indices[j]]])
            prediction = [value * self.selected_models_weights[j] for value in pred[0]] 
    #         prediction = [value * self.correct_probs[j] for value in prediction] 

            predictions.append(prediction)

        prediction, conf, conf_list = soft_vote_predictions(predictions, self.plot) 

 
        if self.plot: 
            print("[True Label]: {}".format(y_true)) 
            print("========================================================") 
        
        return prediction, conf, conf_list   