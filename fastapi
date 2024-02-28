from fastapi import FastAPI, WebSocket
import openllm

app = FastAPI()
llm = openllm.LLM('ugshanyu/mongol-mistral-3')

@app.websocket("/generate")
async def generate(websocket: WebSocket):
    await websocket.accept()
    async for generation in llm.generate_iterator('What is the meaning of life?'):
        await websocket.send_text(generation.outputs[0].text)
