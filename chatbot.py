import os
import google.generativeai as genai
from textwrap import indent
from dotenv import load_dotenv

# Load your API key securely (avoid hardcoding)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the GenerativeModel and ChatSession
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def to_readable_format(text):
    # Split the text into lines
    lines = text.split('\n')

    # Initialize an empty string to store the formatted text
    formatted_text = ""

    # Iterate over each line
    for line in lines:
        # Remove leading and trailing whitespace
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if the line starts with '**' or '*'
        if line.startswith('**'):
            # Add bold formatting for headers
            formatted_text += f"\n**{line.strip('*')}**\n"
        elif line.startswith('*'):
            # Add bullet points for items
            formatted_text += f"- {line.strip('*').strip()}\n"

    return formatted_text.strip()  # Remove leading/trailing whitespace

def chat_loop(user_input):
  """Continuously interacts with the user and provides responses using Gemini"""
  while True:
    if user_input.lower() == "quit":
      break

    # Send user input to the chat session and get response
    response = chat.send_message(user_input)
    return to_readable_format(response.text)
