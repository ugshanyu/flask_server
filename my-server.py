from aiohttp import web
import socketio

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

@sio.event
async def connect(sid, environ):
    print('User connected:', sid)

@sio.event
async def disconnect(sid):
    print('User disconnected:', sid)

@sio.event
async def my_event(sid, message):
    print('User said' + message['data'], sid)
    await sio.emit('my_response', {'data': 'Hello!'}, room=sid)

if __name__ == '__main__':
    web.run_app(app, port=8080)
