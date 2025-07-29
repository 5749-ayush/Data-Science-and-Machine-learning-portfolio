# Voice-Activated AI Chatbot

## Objective
A voice-activated AI chatbot using Python, capable of responding to commands, conducting searches, and interacting with the system.

## Setup Instructions

**1. Prerequisites:**
- Python 3.8 or above
- A working microphone

**2. Create a Virtual Environment:**
```bash
# Navigate to the project directory
cd path/to/this/folder

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


## To start the chatbot, run the chatbot.py script from your terminal:
## python chatbot.py

# ## Available Commands
# "Search Wikipedia for [your query]"
# "Open YouTube" / "Open Google"
# "What is the time?"
# "Lock screen" / "Lock the computer"
# "Shutdown the system" (requires confirmation)
# "Restart the system" (requires confirmation)
# "Exit" / "Quit" / "Stop" (to end the program)



List of All Available Voice Commands

Here is everything you can ask your chatbot to do, based on the code we have built.

General Commands
"What is the time?" - Tells you the current time (e.g., "07:30 PM").
"How are you?" - The chatbot will give a friendly response.
"Who are you?" - The chatbot will describe itself.

Web & Search Commands
"Search Wikipedia for [your topic]" - Searches Wikipedia and reads a two-sentence summary.
Example: "Search Wikipedia for the planet Mars."
"Open YouTube" - Opens YouTube in your web browser.
"Open Google" - Opens Google's homepage.
"Open Stack Overflow" - Opens the Stack Overflow website.

System Commands
"Lock screen" or "Lock the computer" - Locks your Windows computer.
"Shutdown the system" - Will ask for confirmation. You must say "yes" to proceed with the shutdown.
"Restart the system" - Will ask for confirmation. You must say "yes" to proceed with the restart.
Stopping the Program
"Exit", "Quit", or "Stop" - Shuts down the chatbot program.