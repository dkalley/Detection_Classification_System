# This script is used to test all models from the model.py file
# The evaluation will look at the different levels of preprocessing from the process.py
#   the different label types (Basic (Benign,Malicious), Detailed (Attack subytpes))
#   as well as stage 2 for the basic label models using kmeans and isolation forest

from detection_classification_system import DetectionSystem
from models import *
from process import detailed_preprocess
from process import basic_preprocess

train_new_models = False

# Will simulate incoming packets and iterate the test set one datapoint at a time
# If false, the system will give an average time of predicting on the entire test set at once
timing_analysis = False

model_names = ['knn','nb','dt','rf','gb','mlp','kmeans','if']
models = [knn, nb,dt,rf,gb,mlp,kmeans,isofor]


# Begin Testing Block
X = pd.read_csv('data/TestData.csv')
Y = pd.read_csv('data/TestLabel.csv')
Yd = pd.read_csv('data/TestDetailLabel.csv')

# Loop over all models being evaluated
for model_name,model in zip(model_names,models):
    # This flag will create an output file once per model
    first_flag = True
    # Loop over the different pre-processing methods
    for preprocessing,processing_label in zip([detailed_preprocess,basic_preprocess],['','-min']):

        # Loop over the different label types
        for label_type, stage2 in zip(['Basic','Detailed'], [True,False]):
            if label_type == 'Detailed' and model_name in ['kmeans','if']:
                continue

            processing_string = ''
            if processing_label == '':
                processing_string = 'Detailed Pre-Processing'
            else:
                processing_string = 'Basic Pre-Processing'
            print('Evaluating ' + model_name + ' on ' + label_type + ' labels using ' + processing_string)

            # If the model needs to be trained
            if train_new_models:
                # Load the training data
                x = pd.read_csv('data/TrainData.csv')

                if label_type == 'Basic':
                    y = pd.read_csv('data/TrainLabel.csv')
                    
                    # Reduce dataset for KNN to save on time 
                    if model_name == 'knn':
                        x['label'] = y['label']
                        x = x.sample(n=10000, random_state=1)
                        y = x['label']
                        x = x.drop(['label'],axis=1)
                else:
                    y = pd.read_csv('data/TrainDetailLabel.csv')
                    if model_name == 'knn':
                        x['detailed-label'] = y['detailed-label']
                        x = x.sample(n=1000, random_state=1)
                        y = x['detailed-label']
                        x = x.drop(['detailed-label'],axis=1)


                # Instantiate the detection sysem and fit the model
                sys = DetectionSystem(model=model,preprocess=preprocessing)
                if processing_label == '-min':
                    x = x.drop(['history_len','new_history','conn_state'],axis=1)

                sys.fit(x,y.values.ravel())

                # Save the fit model
                sys.save('models/', model_name + label_type + processing_label)

                # Clean up memory
                del x
                del y

            # Else load the trained model from a joblib file
            else:
                sys = DetectionSystem(preprocess=preprocessing)
                sys.load('models/'+model_name+label_type+processing_label+'.joblib')

            if label_type == 'Basic':
                y = Y
            else:
                y = Yd

            # Loop over different stage 2 modesl
            for stage2_model in ['kmeans','if']:

                # If stage2 load the stage 2 model being used
                if stage2:
                    if stage2_model == 'kmeans':
                        sys.stage2 = KMeans(n_clusters=2, random_state=0) 
                    else:
                        sys.stage2 = IsolationForest()

                # If timing analysis loop over each datapoint and time each run
                if timing_analysis:
                    pred, all_times = [], []
                    for index, row in X.iterrows():
                        # Returns prediction and average time
                        p, _ = sys.predict_and_time(row.to_frame().T.values.reshape(1,-1))
                        pred.append(p)
                    print('Average Time: ' + str(sys.calc_average_time()))

                # Else predict on the whole dataset
                else:
                    pred = sys.predict(X,stage2=stage2)

                # Calculate metrics
                print(model_name)
                if stage2:
                    print(stage2_model)
                print('(TP,TN,FP,FN): ' + str(sys.confusion_matrix(y.values.ravel())))
                print('(A,P,R,F1): ' + str(sys.basic_metrics()))
                if stage2:
                    print('(TP,TN,FP,FN): ' + str(sys.confusion_matrix(y.values.ravel(),stage2)))
                    print('(A,P,R,F1): ' + str(sys.basic_metrics(stage2)))
                
                if label_type == 'Detailed':
                    print('Accuracies: ' + str(sys.calc_subtype_accuracy(['Benign','Attack','PartOfAHorizontalPortScan','DDoS','Okiru','C&C','C&C-HeartBeat'], Yd.values.ravel())))
                print()

                append_flag = True
                if first_flag:
                    append_flag = False
                    first_flag = False


                # Save raw data appending to results csv, if stage 2 include the model name otherwise do not
                if stage2:
                    sys.save_raw_data('evaluation'+ model_name+ '.csv',model_name,stage2_model,processing_label,label_type,append=append_flag)
                else:
                    sys.save_raw_data('evaluation'+ model_name+ '.csv',model_name,None,processing_label,label_type,append=append_flag)

                # Break if not testing stage 2
                if not stage2:
                    break