# ai assistant
# pip install speechrecognition
# sudo apt-get install python3-pyaudio
# pip install pyttsx3
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate
# sudo apt-get update
# sudo apt-get install python-picamera python3-picamera

import speech_recognition as sr
import pyttsx3 
import torch
from transformers import pipeline
from gpiozero import Button
from picamera import PiCamera
from time import sleep

# initialize camera
camera = PiCamera()
camera.resolution = (1024, 768)
# initialize camera button
cambutton = Button(31)
imagenum = 0
# initialize video button
vidbutton = Button(32)
vidstate = 0
vidnum = 0
# initialize ai button
aibutton = Button(29)
#############TTS and STT#########################
# Initialize the recognizer 
r = sr.Recognizer() 

# Function to convert text to
# speech
def SpeakText(command):
	
	# Initialize the engine
	engine = pyttsx3.init()
	engine.say(command) 
	engine.runAndWait()
	
# ai part 1
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

while(1): 
	
    if aibutton.is_pressed():
        

        # Exception handling to handle
        # exceptions at the runtime
        try:
            
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level 
                r.adjust_for_ambient_noise(source2, duration=0.2)
                
                #listens for the user's input 
                audio2 = r.listen(source2)
                
                # Using google to recognize audio
                humanVoice = r.recognize_google(audio2)
                humanVoice = humanVoice.lower()

                print(humanVoice)
                
                # ai part 2
                messages = [
                    {
                        "role": "system",
                        "content": "You are a friendly artificially intelligent assistant",
                    },
                    {"role": "user", "content": humanVoice},
                ]
                prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

                ai_response = outputs[0]["generated_text"]
                
                SpeakText(ai_response)

            
                
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            print("unknown error occurred")
        sleep(5)
    elif cambutton.is_pressed():
        camera.start_preview()
        sleep(2)
        camera.capture(f'img_{imagenum}.jpg')
        sleep(5)
    elif vidbutton.is_pressed() and vidstate == 0:
        vidstate = 1
        camera.start_recording(f'vid_{vidnum}.h264')
    elif vidbutton.is_pressed() and vidstate == 1:
        vidstate = 0
        camera.stop_recording()
        vidnum = vidnum + 1
    else: 
        continue
    