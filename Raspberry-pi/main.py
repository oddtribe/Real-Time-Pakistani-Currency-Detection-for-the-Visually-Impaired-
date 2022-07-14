import picamera
import RPi.GPIO as gpio
import numpy as np
from tflite_runtime.interpreter import Interpreter 
from utils import *



# Initializing important variables
model_path = './fourth-data-augmented-background-model.tflite'
prev_pred = ''           # Necessary for counting to work
frames_preds = ''        # Necessary for counting to work
start_limit = 1          # How many times should button be pressed to start
end_limit = 2            # How many times should button be pressed to end
button_pressed = 0       # Keeps track of how many times button has been pressed
height, width = 224, 224  



# Initialize our model tflite interpreter
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details)
# print(output_details)



# Setting up button
def button_callback(channel):
	global button_pressed 
	button_pressed += 1
	print('button pushed and pressed', button_pressed, 'times')

gpio.setmode(gpio.BOARD)  # Use physical pin numbering
gpio.setup(10, gpio.IN, pull_up_down=gpio.PUD_DOWN)  # set pin 10 to be an input pin and set initial value to be pulled low (off)
gpio.add_event_detect(10, gpio.RISING, callback=button_callback)  # Setup event on pin 10 rising edge



print('Press the button to start')
while True:
	
	if button_pressed == start_limit:            # Start the process only if button is pressed required number of times
		
		with picamera.PiCamera() as camera:
			camera.resolution = (height, width)
			camera.start_preview()
			
			# Start Reading frames until the button is not pressed again required number of times
			while button_pressed < end_limit:  
				
				# Input frame
				frame = np.empty((height, width, 3), dtype=np.uint8)   
				camera.capture(frame, 'rgb')
				frame = frame.astype(np.float32)    # Because model understands float32 only, doing it while capturing from camera results in unwanted results of model so that's why doing it here
				np.set_printoptions(suppress=True)

				# Classify
				class_pred = classify_frame(frame, interpreter, input_details, output_details)

				# Display class name
				camera.annotate_text = "{}".format(class_pred)

				# Counting Logic
				if class_pred != prev_pred:
					frames_preds += '  ' + class_pred
				else:
					frames_preds += ' ' + class_pred
				prev_pred = class_pred 
			
				
			camera.stop_preview()                       
			
			# As button has been pressed according to end criteria and need to wrap up so count now
			total_amount = count(frames_preds)

			# Output total_amount through sound
			speak_amount(total_amount)
			
			gpio.cleanup()
			break
