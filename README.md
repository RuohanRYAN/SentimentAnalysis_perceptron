# SentimentAnalysis_perceptron

This program trains a perceptron model on the hotel review in the chicago area and classifies each review into truthful/deceitful and positive/negative. 

Run perceplearn3.py to train the model using the data under op_spam_training_data folder. The program trains both vanilla perceptron model and an average perceptron model. It writes results into "vanillamodel.txt" and "averagemodel.txt". 

Run percepclassify3.py model/path input/path to test on the input file. model/path is the path to the model parameter obtained from training, either a "vanillamodel.txt" or "averagemodel.txt". The program will write the result into a file called "percepoutput.txt" in the following format:

deceitful/truthful positive/negative path/to/review 
