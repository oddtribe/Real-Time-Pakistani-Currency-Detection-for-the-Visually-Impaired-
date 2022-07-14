import numpy as np
import pyttsx3
from collections import Counter

classes_to_inds = {  '1000_back': 0,
		             '1000_front': 1,
		             '100_back': 2,
		             '100_front': 3,
		             '10_back': 4,
		             '10_front': 5,
		             '20_back': 6,
		             '20_front': 7,
		             '5000_back': 8,
		             '5000_front': 9,
		             '500_back': 10,
		             '500_front': 11,
		             '50_back': 12,
		             '50_front': 13,
		             'background': 14   }
inds_to_classes = {ind:currency_side for currency_side,ind in classes_to_inds.items()}


# Classifies a frame 
def classify_frame(frame, interpreter, input_details, output_details):
	frame = np.expand_dims(frame, axis=0)

	# Run inference
	interpreter.set_tensor(input_details[0]['index'], frame)
	interpreter.invoke()

	probs = interpreter.get_tensor( output_details[0]['index'] )[0]
	print(probs)
	
	class_ind = np.argmax(probs)
	class_name = inds_to_classes[class_ind]

	print(class_ind, class_name, '\n')
	return class_name
	
	
# Returns currency denomination e,g passing "50_back" would return 50	
def denomination(currency_side):  
    dash = currency_side.index('_')   
    return int(currency_side[:dash])	
    
# Majority votes
def majority_voting(frames_preds):
    majority = Counter(frames_preds.split(' '))
    return majority.most_common()[0][0]

# Count algorithm 
def count(frames_preds):
    # Splitting different classes predictions based on intervals
    frames_preds_intervals = frames_preds.split('  ')   
    # print(frames_preds_intervals)
    
    # Discarding those frame predictions which were not constant for at least 20 frames
    reliable_frames_preds = list( filter(lambda x: len(x.split(' ')) > 5  ,frames_preds_intervals) )    # Filtering those classes predictions only whose predictions were stable for 40 frames at least
#     print(reliable_frames_preds)
    
    # Discarding those frames which were predicted background 
    notes_preds_frames = list( filter(lambda x: 'background' not in x, reliable_frames_preds) )   # Filtering classes that were not background
#     print(notes_preds_frames)
    
    # Majority voting the frames predictions
    currencies_sides = list( map(lambda x: majority_voting(x), notes_preds_frames) )
    print(currencies_sides)
    
    # Getting only the denomination as 50 is the information needed if note predicted is either 50_back or 50_front
    currencies = list( map(lambda x: denomination(x), currencies_sides) )
    
    # We got our notes, just sum them
    count = sum(currencies)
        
    return count    

# Text to speech for augmenting reality of user
def speak_amount(amount):
	engine = pyttsx3.init()
	sentence = "Total amount is {} rupees".format(amount)
	print(sentence)
	engine.say(sentence)
	engine.runAndWait()
