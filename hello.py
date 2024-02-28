from flask import Flask, render_template
from flask_socketio import SocketIO
import openllm
import asyncio

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')
llm = openllm.LLM('ugshanyu/mongol-mistral-3')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('generate')
def handle_generate(message):
    async def generate_text():
        
        async for generation in llm.generate_iterator(message['text']):
            socketio.emit('generation', {'text': generation.outputs[0].text})

    asyncio.run(generate_text())

if __name__ == '__main__':
    socketio.run(app)
