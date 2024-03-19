from aiohttp import web
import socketio
import openllm
import aiohttp
import ast
import asyncio
import datetime
import json
import datetime
import aiohttp_cors  # Import the aiohttp_cors library


sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

llm = openllm.LLM('ugshanyu/mongol-mistral-3')

def get_top_keys(sentence, keys, top_n=2):
    sentence = sentence.lower()
    key_info = []
    for key in keys:
        count = sentence.count(key.lower())
        position = sentence.find(key.lower())
        key_info.append((key, count, position))
    key_info.sort(key=lambda x: (-x[1], x[2]))
    top_keys = [key for key, _, _ in key_info[:top_n]]
    return top_keys

async def fetch_info_dict():
    url = 'https://huggingface.co/datasets/ugshanyu/TungalagTamir/raw/main/test'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                response_text = await response.text()
                try:
                    return ast.literal_eval(response_text)
                except ValueError:
                    raise Exception("Failed to convert response to dictionary")
            else:
                raise Exception(f"Failed to fetch info_dict from API: {response.status}")

info_dict = {}

async def load_data(app):
    global info_dict
    global string_list
    info_dict = await fetch_info_dict()
    string_list = list(info_dict.keys())
    server_ready_event.set()

app.on_startup.append(load_data)

@sio.event
async def connect(sid, environ):
    print('User connected:', sid)

@sio.event
async def disconnect(sid):
    print('User disconnected:', sid)

async def save_message(user_id, message, generated_message_id):
    save_url = 'http://52.221.164.159/save_message'
    current_date = datetime.datetime.now().isoformat()  # Convert to ISO 8601 format
    data = {
        "userId": user_id,
        "id": generated_message_id,
        "message": message,
        "date": current_date  # Use the ISO-formatted string
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(save_url, json=data) as response:
            if response.status == 200 or response.status == 201:
                print("Message saved successfully")
            else:
                print(f"Failed to save message: {response.status}")

@sio.event
async def my_event(sid, message):
    print("User said: " + message['data'])
    print("User id: " + message['id'])
    generated_message_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + message['id']
    asyncio.create_task(save_message(message['id'], message['data'], generated_message_id))

    prompt = """<s>[INST] Чи бол хиймэл оюунтай ухаалаг туслах бөгөөд чамайг Astvision технологийн баг бүтээж байна. Чиний нэрийг MongolGPT гэдэг. \n Өгүүллэгийг уншаад асуултад хариул. \n Хэрэв өгүүллэгт багтаагүй хамааралгүй асуулт асуувал мэдэхгүй гэж хариул. \n Хүний талаар сайн муу гэж дүгнэлт гаргаж болохгүй. \n Хэрэв хэрэглэгч хэрвээ "Cайн уу", "баярлалаа" гэх мэт энгийн харилцаа өрнүүлэхийг хүсвэл хэрэглэгчтэй эелдгээр харилцаа өрнүүл.\n Өгүүллэг: """

    if(message['id'] == "Usion"):
        prompt = "<s>[INST]" + message['data'] + "[/INST]"
    else:
        input_string = message['data']
        top_keys = get_top_keys(input_string, string_list, top_n=1)
        print(f"The top 2 keys for '{input_string}' are {top_keys}")
        for key in top_keys:
            prompt += info_dict[key] + "\n\n"
        prompt += "Асуулт: " + input_string + " [/INST]"
        print(prompt)
    generated = ""
    async for generation in llm.generate_iterator(
        prompt,
        max_new_tokens=1024,
        temperature=0.5,
        top_p=0.95
    ):
        await sio.emit('my_response', {'data': generation.outputs[0].text}, room=sid)
        generated += generation.outputs[0].text
    asyncio.create_task(save_message(message['id'], prompt + "\n" + generated, generated_message_id))
    await sio.emit('my_response', {'data': "<end>", 'message_id': generated_message_id}, room=sid)

@sio.event
async def all_at_once(sid, message):
    print("User said: " + message['data'])
    generated_message_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + message['id']
    asyncio.create_task(save_message(message['id'], message['data'], generated_message_id))

    prompt = "<s>[INST]" + message['data'] + "[/INST]"

    responses = ""
    async for generation in llm.generate_iterator(
        prompt,
        max_new_tokens=1024,
        temperature=0.5,
        top_p=0.95
    ):
        responses += generation.outputs[0].text

    # combined_response = "\n".join(responses)
    asyncio.create_task(save_message("school", prompt + "\n" + responses, generated_message_id))
    await sio.emit('my_response_all_at_once', {'data': responses, 'message_id': generated_message_id}, room=sid)


async def get_all_keys(request):
    return web.json_response({"keys": list(info_dict.keys())})

# app.router.add_get('/all_keys', get_all_keys)

cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})

# Add your routes and configure CORS for each route
resource = cors.add(app.router.add_resource("/all_keys"))
cors.add(resource.add_route("GET", get_all_keys))

server_ready_event = asyncio.Event()

async def send_initial_message(app):
    await server_ready_event.wait()
    await sio.emit('my_response', {'data': 'Hi'})

app.on_startup.append(send_initial_message)

if __name__ == '__main__':
    web.run_app(app, port=8080)
