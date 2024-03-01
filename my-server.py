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

Онд шийдвэр гаргасан огноо => 2019 онд 72 хамгийн бага ширхэг, 2020 онд 748 хамгийн их ширхэг, 2021 онд 190 ширхэг, 2022 онд 608 ширхэг, 2023 онд 291 ширхэг Хан-Уул дүүрэгт газар олгогдсон байна.

Хан-Уул дүүргийн хамгийн том газар эзэмшигч нь "Нийслэлийн Өмчит Улаанбаатар хөрөнгө оруулалт менежемент ХХК" бөгөөд Үндэсний төв цэнгэлдэх хүрээлэн байгуулах зориулалтаар 14.920.000 м2  хэмжээтэй газарыг Батбаясгалан даргын зөвшөөрөлөөр авж байжээ. 
    Асуулт: """,


"Сумъяабазар": """<s>[INST] Нийтлэлийг уншаад асуултан хариул. Хэрэв нийтлэлд багтаагүй асуулт асуувал мэдэхгүй гэж хариул. Хүний талаар сайн муу гэж дүгнэлт гаргаж болохгүй
    Нийтлэл: "Сумъяабазар нь 2020-2023 он хүртэл 4н жил Улаанбаатар хотын засаг даргаар ажиллаж байсан.
        Тэрээн нийт 2871 удаа шийдвэр гаргаж газар олгож байсан. Шийдвэр гаргасан тоог нь жилээр нь жагсаавал
        2020 онд 568 
        2021 онд 436
        2022 онд 1190
        2023 онд 677

        Дүүргүүдэд газар олгосон тоо
        1) Хан-Уул дүүрэгт хамгийн их буюу 1639, Чингэлтэй дүүрэгт 88 удаа тус тус газар олгосон байна
        2) Сонгинхайрхан дүүрэгт 417
        3) Баянзүрх дүүрэгт 322
        4) Баянгол дүүрэгт 179
        5) Сүхбаатар дүүрэгт 143
        6) Налайх дүүрэгт 42
        7) Багахангай дүүрэгт 24
        8) Багануур дүүрэгт 16 буюу хамгийн бага,  

        Хамгийн хэмээтэй газар олгосон 5н байгуулга.

        1) Эмээлт эко аж үйлдвэрийн парк 2,355,124 м2 газар => Үйлдвэрлэлийн технологийн парк => чиглэлээр олгов
        2) Хорихоос өөр төрлийн ял эдлүүлэх албанд => 1,500,000 м2 => Төрийн захиргааны байгууллага => чиглэлээр олгов
        3) Нийслэлийн засаг даргын Тамгын газарт=> 896,616 м2 => Орон сууцны цогцолбор => чиглэлээр олгов
        4) Хил хамгаалах ерөнхий газарт => 857,212 м2 => Улсын батлан хамгаалах болон аюулгүй байдлыг хангах => чиглэлээр олгов
        5) Нийслэлийн орон сууцны корпораци XK => 792,309 м2 газар => Орон сууцны цогцолбор => чиглэлээр олгов" 
    
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
        prompt = info_dict[most_similar_string] + input_string + " [INST]"
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
