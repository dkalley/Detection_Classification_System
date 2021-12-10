# Detection_Classificaton_System
Advanced Computer Networks Final Project

The data subdirectory holds the datasets used in the project
	The train and test sets are divided into 3 files each. The data, label, and detailed labels.
  
The Main system is in detection_classification_system.py.
	The system is a class that can be instantiated with 
	DetectionSystem(model=[model function], stage2=[model function], preprocess=[preprocessing function]):
	  The model function and preprocessing functions should be functions that the system will call to execute the desired functionality. 
    For the models it will be the creation of the model with the correct parameters. For preprocessing it should take in a dataframe like object and accept one in return. 
	DetectionSystem.fit(X,y):
    will fit the model.
	DetectionSystem.predict(X):
    will predict on X.
	DetectionSystem.save(filepath, filename):
    will save the model at ‘filepath/filename’ .
	DetectionSystem.load(filepath):
    will load a model at ‘filepath’.
  DetectionSystem.load_stage2(filepath):
    will load a stage 2 model at ‘filepath’.
  DetectionSystem.save_raw_data(filepath,model_name,stage2_model,preprocessing,label_type):
    will save the all performance results to filepath for a given model and stage 2 model. 
    It will record the preprocessing procedure name as well as if its for detection or classification. 

evaluate.py  is the testing script used to produce all results.

Models.py holds the functions for all models used in the project.

Process.py holds the two preprocessing methods used in the project. 
