# app.py
from flask import Flask, request, jsonify, session, render_template
from flask_session import Session
import numpy as np
from sentence_transformers import SentenceTransformer
import os, re, json, faiss, csv
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
# --- Initialization ---
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['SECRET_KEY'] = os.urandom(24)
Session(app)

# --- Configure Gemini and Load Models ---
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
except Exception as e:
    print(f"Could not configure Gemini API: {e}")
    gemini_model = None

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('knowledge_base.index')
with open('knowledge_base.json', 'r') as f:
    text_chunks = json.load(f)

@app.route('/')
def home():
    # When a user visits the root, clear the session to start fresh
    # This is useful for testing; you might remove it in production.
    session.clear()
    return render_template('index.html')

# --- Helper Functions (Unchanged) ---
def find_relevant_context(query, k=3):
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return [text_chunks[i] for i in indices[0] if i != -1]

def is_valid_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

def is_valid_phone(phone):
    return bool(re.match(r"^\+?1?\d{9,15}$", phone))

def get_llm_response(context, query):
    if not gemini_model:
        raise ConnectionError("Gemini model is not configured.")
    prompt = f"""
    You are a helpful and versatile AI assistant for Occams Advisory. Your primary goal is to answer questions about the company using the provided context.
    1. If the user's question is about Occams Advisory, answer it using ONLY the context below.
    2. If the context does not contain the answer, reply exactly: "I do not have information on that topic based on the website content."
    3. If the user's question is a general knowledge question NOT related to Occams Advisory, you may answer it.

    Context: --- {' '.join(context)} ---
    Question: {query}
    Answer:"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        raise ConnectionError("API call to Gemini failed.")

def wants_onboarding(message: str) -> bool:
    onboarding_triggers = ["start onboarding", "sign me up", "get started", "work with you", "register", "onboard"]
    return any(trigger in message.lower() for trigger in onboarding_triggers)

def interpret_yes_no(message: str):
    m = message.strip().lower()
    if m in ["yes", "y", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead"]:
        return "yes"
    if m in ["no", "n", "nope", "nah", "not now", "later"]:
        return "no"
    return None

def save_details_to_csv(data):
    filename = 'onboarding_details.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        fieldnames = ['timestamp', 'name', 'email', 'phone', 'session_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'name': data.get('name'), 'email': data.get('email'),
            'phone': data.get('phone'), 'session_id': session.sid
        }
        writer.writerow(entry)
    print(f"Successfully saved data to {filename}")

# --- NEW: Chat History Logging ---
def log_chat_message(session_id, sender, message):
    """Appends a message to the chat history log file."""
    with open('chat_history.log', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] [Session: {session_id}] {sender}: {message}\n")

# --- Main Chat Route ---
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'response': "Please provide a message."})

    # --- NEW: Initialize session and log user message ---
    if 'onboarding_state' not in session:
        session['onboarding_state'] = 'idle'
        session['user_data'] = {}
        session['awaiting_onboarding_confirmation'] = False
    
    session_id = session.sid # Get unique session ID
    log_chat_message(session_id, 'USER', user_message)

    state = session['onboarding_state']
    awaiting_confirmation = session['awaiting_onboarding_confirmation']
    response_text = ""

    # --- ONBOARDING SEQUENCE ---
    if state == 'awaiting_name':
        session['user_data']['name'] = user_message
        session['onboarding_state'] = 'awaiting_email'
        response_text = f"Thanks, {user_message.split(' ')[0]}! What's your email address?"
    elif state == 'awaiting_email':
        if is_valid_email(user_message):
            session['user_data']['email'] = user_message
            session['onboarding_state'] = 'awaiting_phone'
            response_text = "Got it. And finally, what's a good phone number?"
        else:
            response_text = "That doesn't look like a valid email. Could you please try again?"
    elif state == 'awaiting_phone':
        if is_valid_phone(user_message):
            session['user_data']['phone'] = user_message
            session['onboarding_state'] = 'complete'
            session['awaiting_onboarding_confirmation'] = False
            response_text = "Perfect, you're all set! We'll be in touch soon. Can I help with anything else?"
            save_details_to_csv(session['user_data'])
        else:
            response_text = "That phone number doesn't look valid. Please try again."

    # --- GENERAL Q&A and ONBOARDING INITIATION ---
    else:
        # Check if a user who is ALREADY onboarded tries to sign up again
        if state == 'complete' and (wants_onboarding(user_message) or interpret_yes_no(user_message) == "yes"):
             response_text = "It looks like you've already completed the onboarding process. How else can I help you today?"
        
        # Handle yes/no replies to the onboarding nudge
        elif awaiting_confirmation and interpret_yes_no(user_message) == "yes":
            session['awaiting_onboarding_confirmation'] = False
            session['onboarding_state'] = 'awaiting_name'
            response_text = "Great! Let's get started. What's your full name?"
        elif awaiting_confirmation and interpret_yes_no(user_message) == "no":
            session['awaiting_onboarding_confirmation'] = False
            response_text = "No problem. Is there anything else I can help you with?"
        
        # Handle explicit requests to onboard
        elif wants_onboarding(user_message):
            session['onboarding_state'] = 'awaiting_name'
            response_text = "Great! Let's get started. What's your full name?"

        # Default to Q&A
        else:
            try:
                context = find_relevant_context(user_message)
                response_text = get_llm_response(context, user_message)
                
                # --- MODIFIED: Only ask to onboard if not already complete ---
                if state != 'complete':
                    response_text += "\n\nWould you like to get started with our onboarding process?"
                    session['awaiting_onboarding_confirmation'] = True
                else:
                    session['awaiting_onboarding_confirmation'] = False # Ensure flag is off for completed users

            except Exception as e:
                print(f"Error during Q&A: {e}")
                response_text = "I'm having some trouble connecting right now. Please try again in a moment."

    # --- NEW: Log bot response before sending ---
    log_chat_message(session_id, 'BOT', response_text)
    session.modified = True
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)