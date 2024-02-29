from aiohttp import web
import socketio
import openai_llm as openllm

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

llm = openllm.LLM('ugshanyu/mongol-mistral-3')

@sio.event
async def connect(sid, environ):
    print('User connected:', sid)

@sio.event
async def disconnect(sid):
    print('User disconnected:', sid)

@sio.event
async def my_event(sid, message):
    print("User said" + message['data'])
    async for generation in llm.generate_iterator(message['data']):
        await sio.emit('my_response', {'data': generation.outputs[0].text}, room=sid)

if __name__ == '__main__':
    web.run_app(app, port=8080)
