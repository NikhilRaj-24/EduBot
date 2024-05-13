import speech_recognition as sr

import os
os.environ['GOOGLE_API_KEY'] = "AIzaSyBr2HptnjoeSV_QebETtbJJRL7AY_2XlHk"



r = sr.Recognizer()

def record_text():
    while 1:
        try:
            with sr.Microphone(device_index=1) as source2:
                print("Speak Anything :")
                r.adjust_for_ambient_noise(source2)
                audio = r.listen(source2)
                try:
                    print("Recognizing...")
                    text = r.recognize_google(audio)
                    print(f"{Colors.PURPLE}You said: {text} {Colors.RESET}")
                    triggerKeyword = "quit"
                    if text and triggerKeyword in text:
                        break
                    
                except :
                    print("Sorry could not recognize what you said")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("unknown error occured")
        except KeyboardInterrupt:
            print("Recording ended by user.")
            break


def mic():
    mic = sr.Microphone(device_index=1)
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = r.listen(source)
        print("Got it! Now to recognize it...")
        text = r.recognize_google(audio)
        print(f"You said: {text}")
if __name__ == "__main__":
    mic()