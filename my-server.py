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
    url = 'https://huggingface.co/datasets/ugshanyu/TungalagTamir/resolve/main/test'
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
import torch
import numpy as np
import nltk
import re
import PyPDF2
import socketio
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os

nltk.download('punkt')

def load_pdf_text(pdf_path):
    """Loads text content from each page of a PDF."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        page_texts = [page.extract_text() for page in reader.pages]
    return page_texts

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    pattern = re.compile('[^а-яА-ЯүөҮӨ,. 1 2 3 4  5 6 7 8 9 0 a-zA-Z\s]')
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def embedding(pdf_path):
    embedding_file_path = os.path.splitext(pdf_path)[0] + ".npy"
    
    if os.path.exists(embedding_file_path):
        return embedding_file_path
    else:
        page_texts = load_pdf_text(pdf_path)
        cleaned_text = ' '.join(clean_text(page_text) for page_text in page_texts)
        sentences = nltk.sent_tokenize(cleaned_text)
        max_chunk_length = 1000
        
        text_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_length:
                current_chunk += sentence + " "
            else:
                text_chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            text_chunks.append(current_chunk.strip())
        
        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        model = AutoModelForMaskedLM.from_pretrained("intfloat/multilingual-e5-large")
        
        chunk_embeddings = [encode_text([chunk], tokenizer, model) for chunk in text_chunks]
        chunk_embeddings_array = np.vstack([embedding.numpy() for embedding in chunk_embeddings])
        np.save(embedding_file_path, chunk_embeddings_array)
    
    return embedding_file_path

def encode_text(texts, tokenizer, model):
    encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        input_ids = encoded_texts['input_ids']
        attention_mask = encoded_texts['attention_mask']
        outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # Get the embeddings from the output
        pooled_embedding = torch.mean(embeddings, dim=1)  # Mean pooling over tokens
        normalized_embedding = F.normalize(pooled_embedding, p=2, dim=1)
    return normalized_embedding

def mongolchat(text):
    sio = socketio.Client()
    response_data = None

    @sio.event
    def connect():
        pass

    @sio.event
    def disconnect():
        pass

    @sio.on('*')
    def catch_all(event, data):
        if 'data' in data:
            nonlocal response_data
            response_data = data['data']
            sio.disconnect()

    sio.connect('http://202.70.34.27:8081')
    sio.emit('all_at_once', {"data": text, "id": 'all'})
    while response_data is None:
        time.sleep(1)

    return response_data

def ask_mongol_chat(question, pdf_path):
    embedding_file_path = embedding(pdf_path)
    chunk_embeddings_arrays = np.load(embedding_file_path)
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModelForMaskedLM.from_pretrained("intfloat/multilingual-e5-large")
    
    page_texts = load_pdf_text(pdf_path)
    cleaned_text = ' '.join(clean_text(page_text) for page_text in page_texts)
    sentences = nltk.sent_tokenize(cleaned_text)
    max_chunk_length = 1000
        
    text_chunks = []
    current_chunk = ""
        
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += sentence + " "
        else:
            text_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        
    if current_chunk:
        text_chunks.append(current_chunk.strip())

    question_embedding = encode_text([question], tokenizer, model)   
    cos_similarities = []
    for chunk_embedding in chunk_embeddings_arrays:
        chunk_embedding_tensor = torch.tensor(chunk_embedding)
        cos_sim = F.cosine_similarity(question_embedding, chunk_embedding_tensor)
        cos_similarities.append(cos_sim.item())
    closest_chunk_indices = sorted(range(len(cos_similarities)), key=lambda i: cos_similarities[i], reverse=True)[:3]

    closest_sections = [text_chunks[index] for index in closest_chunk_indices]
    closest_sections_str = '\n\n'.join(closest_sections)
    template_content = "Чи бол хиймэл оюунтай ухаалаг эелдэг, сайхан сэтгэлтэй туслах. \n\n Дээрх агуулгыг уншаад асуултад хариул. \n Хэрэв өгүүллэгт багтаагүй эсвэл хамааралгүй асуулт асуувал мэдэхгүй гэж хариул. Хүний талаар сайн муу гэж дүгнэлт гаргаж болохгүй. Хэрэв асуулт нь асуулт биш бол харилцан яриа өрнүүл.  Асуулт: {question} "
    output_value = closest_sections_str + '\n\n' + template_content.format(question=question)

    return output_value

# pdf_path = "Аствишин.pdf"
# embedding_file_path = embedding(pdf_path)

@sio.event
async def my_event(sid, message):
    print("User said: " + message['data'])
    print("User id: " + message['id'])
    generated_message_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + message['id']
    asyncio.create_task(save_message(message['id'], message['data'], generated_message_id))
    detail_link = ""
    prompt = """<s>[INST] Чи бол хиймэл оюунтай ухаалаг эелдэг, сайхан сэтгэлтэй туслах. Чиний нэрийг MongolGPT гэдэг. Чамайг Аствишн компани бүтээсэн. \n\n Өгүүллэгийг уншаад асуултад хариул. \n Хэрэв өгүүллэгт багтаагүй эсвэл хамааралгүй асуулт асуувал мэдэхгүй гэж хариул. Хүний талаар сайн муу гэж дүгнэлт гаргаж болохгүй. Хэрэв асуулт нь асуулт биш бол харилцан яриа өрнүүл. \n Өгүүллэг: """
    addition = ""
    if message['id'] == "Usion":
        prompt = "<s>[INST]" + message['data'] + "[/INST]"
    elif message['id'] == "Ast":
        pdf_path = "Аствишин.pdf"
        embedding_file_path = embedding(pdf_path)
        prompt = ask_mongol_chat(message['data'], pdf_path)
    else:
        input_string = message['data']
        if 'keys' in message and message['keys']:
            keys = message['keys']
            for key in keys:
                if key in info_dict:
                    print(f"Mentioned key'{key}'")
                    prompt += info_dict[key]['value'] + "\n\n"
                    if info_dict[key]["entityRegistryNumber"] != "null":
                        detail_link = f"https://shilen.gov.mn/legal-entity/{info_dict[key]['entityRegistryNumber']}?type=INTRODUCTION"
                else:
                    top_keys = get_top_keys(input_string, string_list, top_n=1)
                    print(f"The top key for '{input_string}' is {top_keys}")
                    for top_key in top_keys:
                        prompt += info_dict[top_key]['value'] + "\n\n"
                    break
        else:
            top_keys = get_top_keys(input_string, string_list, top_n=1)
            print(f"The top key for '{input_string}' is {top_keys}")
            for key in top_keys:
                prompt += info_dict[key]['value'] + "\n\n"
            if(info_dict[top_keys[0]]['entityRegistryNumber']) != "null":
                detail_link = f"https://shilen.gov.mn/legal-entity/{info_dict[top_keys[0]]['entityRegistryNumber']}?type=INTRODUCTION"
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
    # if 'keys' in message and message['keys'] == ['МОНЦЕО']:
    #     await sio.emit('my_response', {'data': "<<https://old.shilen.gov.mn/organization/2772787>>", 'message_id': generated_message_id}, room=sid)
    if (detail_link != ""):
        new_detail_link = f'\n Та дэлгэрэнгүй мэдээллийг <a href="{detail_link}"?type=INTRODUCTION" target="_blank">энэ холбоос руу</a> орж үзнэ үү.'
        await sio.emit('my_response', {'data': new_detail_link, 'message_id': generated_message_id}, room=sid)    
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

    @sio.event
async def all_at_once_userId(sid, message):
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

    asyncio.create_task(save_message("chatbot", prompt + "\n" + responses, generated_message_id))
    await sio.emit('my_response_all_at_once', {'user_id': message['user_id'], 'data': responses, 'message_id': generated_message_id}, room=sid)


async def get_all_keys(request):
    return web.json_response({"keys": list(info_dict.keys())})

# app.router.add_get('/all_keys', get_all_keys)

async def fetch_info_dict_route(request):
    global info_dict
    info_dict = await fetch_info_dict()
    print("updated")
    return web.json_response(info_dict)


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

# Add your routes and configure CORS for each route
fetch_info_dict_resource = cors.add(app.router.add_resource("/fetch_info_dict"))
cors.add(fetch_info_dict_resource.add_route("GET", fetch_info_dict_route))


server_ready_event = asyncio.Event()

async def send_initial_message(app):
    await server_ready_event.wait()
    await sio.emit('my_response', {'data': 'Hi'})

app.on_startup.append(send_initial_message)

if __name__ == '__main__':
    web.run_app(app, port=8080)
