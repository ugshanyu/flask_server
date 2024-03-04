from aiohttp import web
import socketio
#import openllm
#import difflib
import aiohttp


sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

# llm = openllm.LLM('ugshanyu/mongol-mistral-3')

# def most_similar(input_str, string_list):
#     # Calculate similarity scores for each string in the list
#     similarity_scores = [difflib.SequenceMatcher(None, input_str, s).ratio() for s in string_list]
    
#     # Find the index of the string with the highest similarity score
#     max_index = similarity_scores.index(max(similarity_scores))
    
#     # Return the most similar string and its similarity score
#     return string_list[max_index], similarity_scores[max_index]

async def fetch_info_dict():
    url = 'https://raw.githubusercontent.com/ugshanyu/flask_server/main/hello.json'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to fetch info_dict from API: {response.status}")

info_dict = {}

string_list = list(info_dict.keys())


# @sio.event
# async def connect(sid, environ):
#     print('User connected:', sid)

# @sio.event
# async def disconnect(sid):
#     print('User disconnected:', sid)

# @sio.event
# async def my_event(sid, message):
#     print("User said: " + message['data'])
#     print("User id: " + message['id'])

#     prompt = ""
    
#     if(message['id'] == "Usion"):
#         prompt = "<s>[INST]" + message['data'] + "[/INST]"
    
#     elif(message['id'] == "transparent"):
#         input_string = message['data']
#         most_similar_string, score = most_similar(input_string, string_list)
#         prompt = info_dict[most_similar_string] + input_string + " [/INST]"
#         print(f"The most similar string to '{input_string}' is '{most_similar_string}' with a similarity score of {score:.2f}")


#     async for generation in llm.generate_iterator(
#         prompt,
#         max_new_tokens=512,
#         temperature=0.5,
#         top_p=0.95
#     ):
#         await sio.emit('my_response', {'data': generation.outputs[0].text}, room=sid)
#     await sio.emit('my_response', {'data': "<end>"}, room=sid)

# @sio.event
# async def all(sid, message):
#     print("User said: " + message['data'])
#     generated_parts = []
#     async for generation in llm.generate_iterator(
#         message['data'],
#         max_new_tokens=512,
#         temperature=0.5,
#         top_p=0.95
#     ):
#         generated_parts.append(generation.outputs[0].text)
#     full_generated_text = ''.join(generated_parts)
#     await sio.emit('my_response', {'data': full_generated_text}, room=sid)

@sio.event
async def update_info_dict():
    global info_dict
    global string_list
    try:
        info_dict = await fetch_info_dict()
        string_list = list(info_dict.keys())  # Update the string_list as well
        # await sio.emit('info_dict_updated', {'success': True}, room=sid)
        print("Info_dict updated successfully")
        print(info_dict.keys())
        #print all values
        for key in info_dict:
            print(info_dict[key])
    except Exception as e:
        # await sio.emit('info_dict_updated', {'success': False, 'error': str(e)})
        print(f"Error updating info_dict: {e}")

if __name__ == '__main__':
    web.run_app(app, port=8080)
