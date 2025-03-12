import os
import logging
import absl.logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all TensorFlow logs

# Suppress ABSL logs
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppresses ABSL warnings

# Suppress Werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Suppresses Werkzeug debug messages

# Import other libraries after suppressing logs
import nltk
import pickle
import json
import random
import numpy as np
import torch
import subprocess
import re
import black
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Download necessary NLTK data
nltk.download('popular')
lemmatizer = WordNetLemmatizer()

# Load the chatbot model and data with error handling
def load_chatbot_model():
    try:
        required_files = ['model.h5', 'data.json', 'texts.pkl', 'labels.pkl']
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} not found. Please train the model first.")

        model = load_model('model.h5')
        intents = json.load(open('data.json'))
        words = pickle.load(open('texts.pkl', 'rb'))
        classes = pickle.load(open('labels.pkl', 'rb'))

        # Compile the model to build metrics
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(f"Loaded chatbot model with {len(words)} words and {len(classes)} classes.")
        return model, intents, words, classes
    except Exception as e:
        print(f"Error loading chatbot model: {e}")
        return None, None, None, None

chatbot_model, intents, words, classes = load_chatbot_model()

# Load the GPT-Neo model and tokenizer once
def load_gpt_model():
    try:
        model_name = "EleutherAI/gpt-neo-125M"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading GPT-Neo model: {e}")
        return None, None

gpt_model, gpt_tokenizer = load_gpt_model()

# Chatbot functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, max_length=2):
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(max_length, dtype=int)  # Ensure correct length
    for i, w in enumerate(sentence_words):
        if i >= max_length:
            break  # Stop if we exceed the max length
        if w in words:
            bag[i] = 1
    return bag

def predict_class(sentence, model):
    if model is None or words is None or classes is None:
        return []

    p = bow(sentence, words)
    
    # Ensure shape matches model input
    p = np.array([p])
    if p.shape[1] != model.input_shape[1]:
        print(f"Error: Model expects input shape ({model.input_shape[1]}), but received ({p.shape[1]})")
        return []

    res = model.predict(p)[0]
    ERROR_THRESHOLD = 0.25

    # Debug: Print lengths of res and classes
    print(f"Length of res: {len(res)}, Length of classes: {len(classes)}")
    print(f"Classes: {classes}")

    # Filter results above the threshold
    results = []
    for i, r in enumerate(res):
        if r > ERROR_THRESHOLD:
            # Map the index to the corresponding intent using the classes dictionary
            intent = None
            for key, value in classes.items():
                if value == i:  # Match the index to the value in the dictionary
                    intent = key
                    break
            if intent:
                results.append({"intent": intent, "probability": float(r)})
            else:
                print(f"Warning: No intent found for index {i}")

    # Sort results by probability in descending order
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results

def get_response(ints, intents_json):
    if not ints:
        return "I'm not sure I understand."
    
    tag = ints[0]['intent']  # Get the predicted intent
    print(f"Predicted intent tag: {tag}")  # Debug: Print the predicted intent tag

    for i in intents_json['intents']:
        print(f"Checking intent: {i['tag']}")  # Debug: Print each intent being checked
        if i['tag'] == tag:
            print(f"Found matching intent: {i['tag']}")  # Debug: Print the matching intent
            print(f"Responses for this intent: {i['responses']}")  # Debug: Print the responses
            return random.choice(i['responses'])  # Return a random response for the intent
    
    print("No matching intent found.")  # Debug: Print if no matching intent is found
    return "I don't have a response for that."

def chatbot_response(msg):
    if chatbot_model is None:
        return "Chatbot model is not loaded. Please check the logs for errors."
    ints = predict_class(msg, chatbot_model)
    print(f"Predicted intents: {ints}")  # Debug: Print predicted intents
    return get_response(ints, intents)

# GPT-Neo Code Generation
def generate_code(prompt, model, tokenizer, max_length=300):
    if model is None or tokenizer is None:
        return "GPT-Neo model is not loaded. Please check the logs for errors."
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = inputs.ne(tokenizer.pad_token_id).float()  # Create attention mask
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # Set pad token ID
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Format Code using Black
def format_code(code):
    try:
        return black.format_str(code, mode=black.FileMode())
    except Exception as e:
        return f"Error formatting code: {e}\n\nOriginal Code:\n{code}"

# Detect Dependencies in Code
def detect_dependencies(code):
    matches = re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", code, re.MULTILINE)
    return list(set(matches))

# Test Code Execution
def test_code(code):
    try:
        temp_file = "temp.py"
        with open(temp_file, "w") as f:
            f.write(code)
        result = subprocess.run(["python", temp_file], capture_output=True, text=True, timeout=5)
        os.remove(temp_file)
        return (result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr)
    except subprocess.TimeoutExpired:
        return False, "Code execution timed out."
    except Exception as e:
        return False, str(e)

# Flask App
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg', '').strip()
    if not user_text:
        return jsonify({"error": "No message provided."})
    
    if "generate code" in user_text.lower():
        if gpt_model is None or gpt_tokenizer is None:
            return jsonify({"error": "GPT-Neo model is not loaded. Please check the logs for errors."})
        
        code_prompt = user_text.replace("generate code", "").strip()
        generated_code = generate_code(code_prompt, gpt_model, gpt_tokenizer)
        formatted_code = format_code(generated_code)
        dependencies = detect_dependencies(formatted_code)
        success, output = test_code(formatted_code)
        return jsonify({
            "code": formatted_code,
            "dependencies": dependencies,
            "test_success": success,
            "test_output": output
        })
    else:
        if chatbot_model is None:
            return jsonify({"error": "Chatbot model is not loaded. Please check the logs for errors."})
        return jsonify({"response": chatbot_response(user_text)})

if __name__ == "__main__":
    app.run(debug=False)  # Set debug=False to disable Flask debug mode