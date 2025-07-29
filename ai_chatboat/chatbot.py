# --- 1. Import necessary libraries ---
import pyttsx3  # For text-to-speech functionality
import speech_recognition as sr  # For recognizing voice command
import datetime  # For getting the current date and time
import wikipedia  # For accessing Wikipedia articles
import webbrowser  # For opening websites
import os  # For interacting with the operating system (e.g., shutdown)
import ctypes  # For system commands like locking the screen on Windows

# --- 2. Initialize the Text-to-Speech Engine ---
engine = pyttsx3.init('sapi5')  # sapi5 is the Microsoft Speech API, used on Windows
voices = engine.getProperty('voices')
# You can check the available voices by printing the `voices` variable
# To set a specific voice, use its ID. voices[0] is typically male, voices[1] is female.
engine.setProperty('voice', voices[0].id)

# --- 3. Create the function for the chatbot to speak ---
def speak(audio):
    """This function takes text and speaks it."""
    print(f"Chatbot: {audio}")  # Prints the chatbot's response to the terminal for clarity
    engine.say(audio)  # Queues the audio to be spoken
    engine.runAndWait()  # Speaks the queued audio and waits until it's finished

# --- 4. Create the greeting function ---
def wishMe():
    """Greets the user based on the time of day."""
    hour = int(datetime.datetime.now().hour)  # Gets the current hour (0-23)

    if hour >= 0 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")

    speak("I am your AI Chatbot. How may I help you?")

# --- 5. Create the function to take and recognize voice commands ---
def takeCommand():
    """Listens for a command via microphone and returns it as text."""
    r = sr.Recognizer()  # Initialize the recognizer
    with sr.Microphone() as source:
        print("Listening...")
        # These settings help improve recognition accuracy
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)  # Captures audio from the microphone

    try:
        print("Recognizing...")
        # Use Google's online speech recognition service
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")
    except Exception as e:
        # This block runs if recognition fails for any reason
        print(e)  # Optional: print the error
        speak("Sorry, I didn't catch that. Could you please say it again?")
        return "None"  # Return "None" to signify that no command was understood

    return query.lower()  # Return the command in lowercase for easy handling



# --- 6. Main execution block ---

if __name__ == "__main__":
    wishMe()  # Greet the user once when the script starts
    
    # Infinite loop to continuously listen for commands
    while True:
        query = takeCommand().lower()  # Listen for a command and convert to lowercase

        # --- Logic for executing tasks based on the query ---

        # Command to search Wikipedia
        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            try:
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia")
                speak(results)
            except Exception as e:
                speak(f"Sorry, I could not find any results for {query} on Wikipedia.")

        # Command to open websites
        elif 'open youtube' in query:
            speak("Opening YouTube...")
            webbrowser.open("youtube.com")

        elif 'open google' in query:
            speak("Opening Google...")
            webbrowser.open("google.com")

        # Command to get the time
        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%I:%M %p") # e.g., "05:30 PM"
            speak(f"The time is {strTime}")

        # System command to lock the screen
        elif 'lock screen' in query or 'lock the computer' in query:
            speak("Locking the screen.")
            ctypes.windll.user32.LockWorkStation()

        # Command to exit the program
        elif 'exit' in query or 'quit' in query or 'stop' in query:
            speak("Goodbye! Shutting down.")
            break  # This command breaks out of the infinite 'while' loop
        
