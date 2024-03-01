from aiohttp import web
import socketio
import openllm
import difflib


sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

llm = openllm.LLM('ugshanyu/mongol-mistral-3')

def most_similar(input_str, string_list):
    # Calculate similarity scores for each string in the list
    similarity_scores = [difflib.SequenceMatcher(None, input_str, s).ratio() for s in string_list]
    
    # Find the index of the string with the highest similarity score
    max_index = similarity_scores.index(max(similarity_scores))
    
    # Return the most similar string and its similarity score
    return string_list[max_index], similarity_scores[max_index]


string_list = ["Хан-Уул", "Сумъяабазар"]
info_dict = {"Хан-Уул": """ <s>[INST] Нийтлэлийг уншаад асуултан хариул. Хэрэв нийтлэлд багтаагүй асуулт асуувал мэдэхгүй гэж хариул. Хүний талаар сайн муу гэж дүгнэлт гаргаж болохгүй
    Нийтлэл: "Хан-Уул Дүүрэг нь Монгол улсын нийслэл Улаанбаатар хотын 9 дүүргийн нэг юм. 16 хороонд хуваагддаг. 2022 оны байдлаар хүн амын тоо  238,511. Одоогийн засаг дарга Ж.Алдаржавхлан.
Үе үеийн Хотын засаг даргуудын шийдвэр гаргаж Хан-Уул дүүрэгт газар олгосон байдал Сумъяабазар 1639 (хамгийн их), Энхболд 1097,Мөнхбаяр 467, Бат-Үүл 336, Батболд 296, Амарсайхан 242, Билэгт 82, Батбаяр 82, Батбаясгалан 28 (хамгийн бага) удаа шийдвэр гаргаж Хан-Уул дүүрэгт газар олгосон байна.

Онд шийдвэр гаргасан огноо 2019 онд 72 хамгийн бага ширхэг, 2020 онд 748 хамгийн их ширхэг, 2021 онд 190 ширхэг, 2022 онд 608 ширхэг, 2023 онд 291 ширхэг Хан-Уул дүүрэгт газар олгогдсон байна.

Хан-Уул дүүргийн хамгийн том газар эзэмшигч нь "Нийслэлийн Өмчит Улаанбаатар хөрөнгө оруулалт менежемент ХХК" бөгөөд Үндэсний төв цэнгэлдэх хүрээлэн байгуулах зориулалтаар 14.920.000 м2  хэмжээтэй газартай. Уг газарыг олгосон Засаг дарга нь  Батбаясгалан юм.
    Асуулт: """,


"Сумъяабазар": """<s>[INST] Нийтлэлийг уншаад асуултан хариул. Хэрэв нийтлэлд багтаагүй асуулт асуувал мэдэхгүй гэж хариул. Хүний талаар сайн муу гэж дүгнэлт гаргаж болохгүй
    Нийтлэл: "Сумъяабазар нь 2020-2023 он хүртэл 4н жил Улаанбаатар хотын засаг даргаар ажиллаж байсан. Тэрээн нийт 2871 удаа шийдвэр гаргаж газар олгож байсан. Тэрээр хамгийн том хэмжээтэй газарыг Эмээлт эко аж үйлдвэрийн паркд "Үйлдвэрлэлийн технологийн парк" чиглэлрээр 2,355,124 м2 хэмжээтэй газар олгосон.
    Сумъяабазар нь хамгийн их шийдвэрийг 2021 онд 1190 удаа гаргаж газар олгосон байна. 2020 онд 568 удаа, 2022 онд 677 удаа, 2023 онд 436 удаа тус тус шийдвэр гаргаж газар олгосон байна.
    Дүүргүүдэд газар олгосон газарын тоо
        1) Хан-Уул дүүрэгт хамгийн их буюу 1639 удаа шийдвэр гаргаж газар олгосон байна,
        2) Чингэлтэй дүүрэгт 88,
        3) Сонгинхайрхан дүүрэгт 417
        4) Баянзүрх дүүрэгт 322
        5) Баянгол дүүрэгт 179
        6) Сүхбаатар дүүрэгт 143
        7) Налайх дүүрэгт 42
        8) Багахангай дүүрэгт 24
        9) Багануур дүүрэгт 16 буюу хамгийн бага удаа газар олгосон байна.

    Асуулт: """}



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

    prompt = ""
    
    if(message['id'] == "Usion"):
        prompt = "<s>[INST]" + message['data'] + "[/INST]"
    
    elif(message['id'] == "transparent"):
        input_string = message['data']
        most_similar_string, score = most_similar(input_string, string_list)
        prompt = info_dict[most_similar_string] + input_string + " [/INST]"
        print(f"The most similar string to '{input_string}' is '{most_similar_string}' with a similarity score of {score:.2f}")


    async for generation in llm.generate_iterator(
        prompt,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.95
    ):
        await sio.emit('my_response', {'data': generation.outputs[0].text}, room=sid)
    await sio.emit('my_response', {'data': "<end>"}, room=sid)

@sio.event
async def all(sid, message):
    print("User said: " + message['data'])
    generated_parts = []
    async for generation in llm.generate_iterator(
        message['data'],
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.95
    ):
        generated_parts.append(generation.outputs[0].text)
    full_generated_text = ''.join(generated_parts)
    await sio.emit('my_response', {'data': full_generated_text}, room=sid)

if __name__ == '__main__':
    web.run_app(app, port=8080)
