from flask import Flask, render_template, request, jsonify
from utils import generate_response
import json

app = Flask(__name__)

# Define the available LLM models
llm_models = ["gpt-3.5-turbo", "gpt-4", "Llama-2-70b-chat", "Falcon-40b-instruct"]

@app.route('/')
def index():
    return render_template('index.html', llm_models=llm_models)

@app.route('/generate_response', methods=['POST'])
def get_response():
    data = request.json
    query = data['query']
    chat_history = []
    model_choice = data['model_choice']
    
    # Call the generate_response function from utils
    response, updated_chat_history = generate_response(query, chat_history, model_choice)
    
    return {"response": response, "chat_history": updated_chat_history}

if __name__ == '__main__':
    app.run(debug=True)
