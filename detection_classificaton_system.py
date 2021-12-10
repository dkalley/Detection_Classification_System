import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from joblib import dump, load
import time
from process import detailed_preprocess

class DetectionSystem():    
    def __init__(self, model=DecisionTreeClassifier, stage2=IsolationForest, preprocess=detailed_preprocess):
        # model pipeline values
        self.preprocess_ = preprocess
        self.model_ = model
        self.stage2 = stage2
        self.predictions = []      
        self.time = []
        self.predictions_stage2 = []

        # Metrics
        self.confusion_matrix_ = (0,0,0,0)
        self.confusion_matrix_stage2_ = (0,0,0,0)
        self.base_metrics_ = (0,0,0,0)
        self.base_metrics_stage2_ = (0,0,0,0)
        self.subtype_accuracy = {}
        self.avg_time = -1
        self.avg_time_stage2 = -1

    def fit(self, X, y):
        self.model_ = self.model_(X,y)
        return self.model_

    def predict_and_time(self, X):
        avg_time = -1
        time_s = time.process_time_ns() 

        x = self.preprocess(X)
        pred = self.model_.predict(x)

        time_f = time.process_time_ns()
        avg_time = time_f - time_s

        # Temporarily store prediction
        self.predictions.append(pred)
        self.time.append(avg_time)

        return pred, avg_time

    def predict(self, X,stage2=False):
        avg_time = -1
        time_s = time.process_time_ns() 
        x = self.preprocess(X)
        pred = self.model_.predict(x)

        time_f = time.process_time_ns()
        avg_time = (time_f - time_s)/len(x.index)

        # Temporarily store prediction
        self.predictions = pred
        self.avg_time = avg_time

        if stage2:
            self.predict_stage2(X)

        return pred

    def predict_stage2(self, X):
        X = self.preprocess_(X)
        X['label'] = self.predictions
        
        if len(X[X['label']=='Benign'].index) == 0:
            self.predictions_stage2 = []
            self.avg_time_stage2 = 0
            return 
        avg_time = -1
        time_s = time.process_time_ns() 

        pred = self.stage2.fit_predict(X[X['label']=='Benign'].drop(['label'],axis=1))

        time_f = time.process_time_ns()
        avg_time = (time_f - time_s)/len(X.index)

        # Translate the output and save
        X.loc[X['label'] == 'Benign', 'label'] = ['Benign' if p == 1 else 'Malicious' for p in pred]
        self.predictions_stage2 = list(X['label'])
        self.avg_time_stage2 = avg_time

        return

    def calc_average_time(self):
        self.avg_time = sum(self.time)/len(self.time)
        return self.avg_time

    def calc_average_time_stage2(self):
        self.avg_time_stage2 = sum(self.time_stage2)/len(self.time_stage2)
        return self.avg_time_stage2

    def preprocess(self, X):
        return self.preprocess_(X)

    def save(self,filepath,filename):
        dump(self.model_, filepath+filename+'.joblib')
    
    def load(self,filepath):
        self.model_ = load(filepath)

    def load_stage2(self,filepath):
        self.stage2 = load(filepath)

    def save_raw_data(self,filepath,model_name,stage2_model,preprocessing,label_type,append=False):
        # Create dataframe of raw data
        row = {'model' : model_name}
        row['stage2_model'] = stage2_model
        row['preprocessing'] = preprocessing
        row['label-type'] = label_type

        # Basic Metrics
        row['A'] = self.base_metrics_[0]
        row['P'] = self.base_metrics_[1]
        row['R'] = self.base_metrics_[2]
        row['F1'] = self.base_metrics_[3]

        # Confusion Matrix
        row['TP'] = self.confusion_matrix_[0]
        row['TN'] = self.confusion_matrix_[1]
        row['FP'] = self.confusion_matrix_[2]
        row['FN'] = self.confusion_matrix_[3]

        # Basic Metrics For stage 2
        row['A-stage2'] = self.base_metrics_stage2_[0]
        row['P-stage2'] = self.base_metrics_stage2_[1]
        row['R-stage2'] = self.base_metrics_stage2_[2]
        row['F1-stage2'] = self.base_metrics_stage2_[3]

        # Confusion Matrix for stage 2
        row['TP-stage2'] = self.confusion_matrix_stage2_[0]
        row['TN-stage2'] = self.confusion_matrix_stage2_[1]
        row['FP-stage2'] = self.confusion_matrix_stage2_[2]
        row['FN-stage2'] = self.confusion_matrix_stage2_[3]

        # Subtype accuracies
        row['subtypes'] = self.subtype_accuracy

        # Time and all predictions
        row['avg_time'] = self.avg_time
        row['avg_time_stage2'] = self.avg_time_stage2
        row['predictions'] = self.predictions
        row['stage2_predictions'] = self.predictions_stage2

        data = pd.DataFrame()
        data = data.append(row,ignore_index=True)

        # Determine mode to save data
        m = 'w'
        if append:
            m = 'a'

        # Save data
        data.to_csv(filepath, index=False, mode=m, header=(not append))

    # Benign is a negative
    # Malicious is positive
    # All predictions not malicious are considered negative
    def confusion_matrix(self,actual,stage2=False):
        TP, TN, FP, FN = 0, 0, 0, 0

        if stage2: 
            pred = self.predictions_stage2
        else:
            pred = self.predictions

        for p, y in zip(pred, actual):
            if p not in ['benign','Benign'] and y not in ['benign','Benign']:
                TP = TP + 1
            if p in ['benign','Benign'] and y in ['benign','Benign']:
                TN = TN + 1  
            if p not in ['benign','Benign'] and y in ['benign','Benign']:
                FP = FP + 1
            if p in ['benign','Benign'] and y not in ['benign','Benign']:
                FN = FN + 1

        if stage2:
            self.confusion_matrix_stage2_ = (TP,TN,FP,FN)
        else:
            self.confusion_matrix_ = (TP,TN,FP,FN)

        return (TP,TN,FP,FN)

    def basic_metrics(self, stage2=False):
        if stage2: 
            (TP,TN,FP,FN)= self.confusion_matrix_stage2_
        else:
            (TP,TN,FP,FN)= self.confusion_matrix_
        A, P, R, F1 = 0, 0, 0, 0

        # Accuracy
        A = ((TP+TN)/(TP+TN+FP+FN)) if TP+TN+FP+FN != 0 else 0

        # Precision
        P = (TP/(TP+FP)) if TP+FN != 0 else 0

        # Recall
        R = (TP/(TP+FN)) if TP+FN != 0 else 0

        # # F1 Score
        F1 = (2*(P*R)/(P+R)) if P+R != 0 else 0

        if stage2:
            self.base_metrics_stage2_ = (A,P,R,F1)
        else:
            self.base_metrics_ = (A,P,R,F1)

        return (A,P,R,F1)

    def calc_subtype_accuracy(self, subtypes, actual):
        # Initialize dictionaries for determine results
        occurrence = {}
        correct_predictions = {}
        for subtype in subtypes:
            occurrence[subtype] = 0
            correct_predictions[subtype] = 0

        for prediction, actual in zip(self.predictions,actual):
            occurrence[actual] = occurrence[actual] + 1
            if prediction == actual:
                correct_predictions[actual] = correct_predictions[actual] + 1

        self.subtype_accuracy = {subtype:(correct_predictions[subtype]/occurrence[subtype] if occurrence[subtype] != 0 else -1) for subtype in subtypes}

        return self.subtype_accuracy 
