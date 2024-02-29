from aiohttp import web
import socketio
import openllm

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
    sio.emit('my_response', {'data': "<end>"}, room=sid)

@sio.event
async def all(sid, message):
    print("User said: " + message['data'])
    generated_parts = []
    async for generation in llm.generate_iterator(message['data']):
        generated_parts.append(generation.outputs[0].text)
    full_generated_text = ''.join(generated_parts)
    await sio.emit('my_response', {'data': full_generated_text}, room=sid)

if __name__ == '__main__':
    web.run_app(app, port=8080)
