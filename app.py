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
    session.clear()
    return render_template('index.html')

# --- Helper Functions (Mostly Unchanged) ---
def find_relevant_context(query, k=5):
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
        raise ConnectionError("Simulating API failure")
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        raise ConnectionError("API call to Gemini failed.")

def wants_onboarding(message: str) -> bool:
    onboarding_triggers = ["start onboarding", "sign me up", "get started", "work with you", "register", "onboard"]
    return any(trigger in message.lower() for trigger in onboarding_triggers)

def interpret_yes_no(message: str):
    m = message.strip().lower()
    yes_words = ["yes", "y", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead"]
    no_words = ["no", "n", "nope", "nah", "not now", "later", "that's all", "im good", "i'm good"]
    if m in yes_words:
        return "yes"
    if m in no_words:
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

def log_chat_message(session_id, sender, message):
    with open('chat_history.log', 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] [Session: {session_id}] {sender}: {message}\n")

# --- Main Chat Route ---
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'response': "Please provide a message."})

    if 'onboarding_state' not in session:
        session['onboarding_state'] = 'idle'
        session['user_data'] = {}
        session['awaiting_onboarding_confirmation'] = False
    
    session_id = session.sid
    log_chat_message(session_id, 'USER', user_message)

    state = session['onboarding_state']
    awaiting_confirmation = session['awaiting_onboarding_confirmation']
    response_text = ""

    # --- ONBOARDING SEQUENCE ---
    if state == 'awaiting_name':
        session['user_data']['name'] = user_message
        session['onboarding_state'] = 'awaiting_email'
        response_text = f"Thanks, {user_message.split(' ')[0]}! What is your professional email address?"
    elif state == 'awaiting_email':
        if is_valid_email(user_message):
            session['user_data']['email'] = user_message
            session['onboarding_state'] = 'awaiting_phone'
            response_text = "Excellent. And finally, what is a good contact number?"
        else:
            response_text = "That doesn't appear to be a valid email format. Could you please try again?"
    elif state == 'awaiting_phone':
        if is_valid_phone(user_message):
            session['user_data']['phone'] = user_message
            session['onboarding_state'] = 'complete'
            session['awaiting_onboarding_confirmation'] = False
            response_text = "Thank you. Your onboarding is complete, and a member of our team will be in touch shortly. Is there anything else I can assist you with today?"
            save_details_to_csv(session['user_data'])
        else:
            response_text = "That does not seem to be a valid phone number. Please provide a valid contact number."

    # --- GENERAL Q&A and ONBOARDING INITIATION ---
    else:
        if state == 'complete':
            if interpret_yes_no(user_message) == "no":
                response_text = "Understood. Thank you for your time, and have a great day."
            elif wants_onboarding(user_message):
                response_text = "It looks like you've already completed the onboarding process. If you need to update your details, please contact our team directly. How else can I help?"
            else:
                # If the user asks another question, fall through to Q&A
                pass 
        
        # This block will now execute if response_text is still empty
        if not response_text:
            if awaiting_confirmation and interpret_yes_no(user_message) == "yes":
                session['awaiting_onboarding_confirmation'] = False
                session['onboarding_state'] = 'awaiting_name'
                response_text = "Great! To begin, may I have your full name?"
            elif awaiting_confirmation and interpret_yes_no(user_message) == "no":
                session['awaiting_onboarding_confirmation'] = False
                response_text = "Of course. Please let me know if you have any other questions."
            elif wants_onboarding(user_message):
                session['onboarding_state'] = 'awaiting_name'
                response_text = "Great! To begin, may I have your full name?"
            else:
                try:
                    context = find_relevant_context(user_message)
                    llm_answer = get_llm_response(context, user_message)
                    response_text = llm_answer
                    
                    # --- MODIFIED: Only nudge if the answer was successful ---
                    if "I do not have information on that topic" not in llm_answer and state != 'complete':
                        response_text += "\n\nI hope that was helpful. Would you like to get started with our onboarding process?"
                        session['awaiting_onboarding_confirmation'] = True
                    else:
                        session['awaiting_onboarding_confirmation'] = False

                except Exception as e:
                    print(f"Error during Q&A: {e}")
                    if context:
                        response_text = f"I am currently unable to connect to the AI model, but I found this information from the website that might help:\n\n> \"{context[0]}\""
                    else:
                        response_text = "I am currently unable to connect to the AI model and could not find relevant information on the website for your query."

    log_chat_message(session_id, 'BOT', response_text)
    session.modified = True
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)