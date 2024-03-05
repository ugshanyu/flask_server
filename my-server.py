from aiohttp import web
import socketio
import openllm
#import difflib
import aiohttp
import ast
import time


sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

llm = openllm.LLM('ugshanyu/mongol-mistral-3')

# def most_similar(input_str, string_list):
#     # Calculate similarity scores for each string in the list
#     similarity_scores = [difflib.SequenceMatcher(None, input_str, s).ratio() for s in string_list]
    
#     # Find the index of the string with the highest similarity score
#     max_index = similarity_scores.index(max(similarity_scores))
    
#     # Return the most similar string and its similarity score
#     return string_list[max_index], similarity_scores[max_index]


def get_top_keys(sentence, keys, top_n=2):
    # Convert the sentence to lowercase for case-insensitive matching
    sentence = sentence.lower()
    
    # Create a list to store the count and first occurrence position of each key
    key_info = []

    # Count the occurrences of each key in the sentence and record their first occurrence
    for key in keys:
        count = sentence.count(key.lower())
        position = sentence.find(key.lower())
        key_info.append((key, count, position))

    # Sort the keys first by count in descending order, then by position in ascending order
    key_info.sort(key=lambda x: (-x[1], x[2]))

    # Get the top N keys
    top_keys = [key for key, _, _ in key_info[:top_n]]

    return top_keys


async def fetch_info_dict():
    
    # url = 'https://raw.githubusercontent.com/ugshanyu/flask_server/main/hello.json'
    url = 'https://huggingface.co/datasets/ugshanyu/TungalagTamir/raw/main/test'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                response_text = await response.text()
                try:
                    print(ast.literal_eval(response_text))
                    return ast.literal_eval(response_text)
                except ValueError:
                    raise Exception("Failed to convert response to dictionary")
            else:
                raise Exception(f"Failed to fetch info_dict from API: {response.status}")


info_dict = {}

string_list = list(info_dict.keys())


@sio.event
async def connect(sid, environ):
    print('User connected:', sid)

@sio.event
async def disconnect(sid):
    print('User disconnected:', sid)

@sio.event
async def my_event(sid, message):
    print("User said: " + message['data'])
    print("User id: " + message['id'])

    prompt = """<s>[INST] Нийтлэлийг уншаад асуултан хариул. Хэрэв нийтлэлд багтаагүй хамааралгүй асуулт асуувал мэдэхгүй гэж хариул. Хүний талаар сайн муу гэж дүгнэлт гаргаж болохгүй
    Нийтлэл: """
    
    if(message['id'] == "Usion"):
        prompt = "<s>[INST]" + message['data'] + "[/INST]"
    
    elif(message['id'] == "transparent"):
        input_string = message['data']

        #based on the input string, get the top 2 keys from the info_dict
        top_keys = get_top_keys(input_string, string_list, top_n=1)
        print(f"The top 2 keys for '{input_string}' are {top_keys}")

        # Get the information for the top 2 keys
        for key in top_keys:
            prompt += info_dict[key] + "\n\n"
        
        prompt += "Асуулт: " + input_string + " [/INST]"
        print(prompt)

        # most_similar_string, score = most_similar(input_string, string_list)
        # prompt = info_dict[most_similar_string] + input_string + " [/INST]"
        # print(f"The most similar string to '{input_string}' is '{most_similar_string}' with a similarity score of {score:.2f}")

    async for generation in llm.generate_iterator(
        prompt,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.95
    ):
        await sio.emit('my_response', {'data': generation.outputs[0].text}, room=sid)
    await sio.emit('my_response', {'data': "<end>"}, room=sid)

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
async def update_info_dict(sid, message):
    global info_dict
    global string_list
    try:
        info_dict = await fetch_info_dict()
        string_list = list(info_dict.keys())  # Update the string_list as well
        # await sio.emit('info_dict_updated', {'success': True}, room=sid)
        print("Info_dict updated successfully")
        print(string_list)
        
        #print all values
        # for key in info_dict:
        #     print(info_dict[key])
    except Exception as e:
        # await sio.emit('info_dict_updated', {'success': False, 'error': str(e)})
        print(f"Error updating info_dict: {e}")

if __name__ == '__main__':
    web.run_app(app, port=8080)
