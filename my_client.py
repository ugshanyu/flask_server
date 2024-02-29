import socketio
import time

ended = True

# Create a Socket.IO client
sio = socketio.Client()

# Event handler for connection
@sio.event
def connect():
    print('Connected to server')

# Event handler for disconnection
@sio.event
def disconnect():
    print('Disconnected from server')

# Event handler for receiving a response
@sio.event
def my_response(data):
    global ended
    token = data['data']
    
    if token == "<end>":
        ended = True
        print()  # Move to a new line after the message is complete
    else:
        print(token, end='', flush=True)  # Print each token without a newline and flush the output


# Connect to the server
sio.connect('https://f9qudi4ybmg8n1-8080.proxy.runpod.net/')

# Continuously send messages to the server based on user input
while True:
    while not ended:
        time.sleep(0.1)  # Wait for the 'ended' flag to become True

    user_input = input("=============================================================================\n")
    user_input = "<s>[INST]" + user_input + "[/INST]"
    print("-----------------------------------------------------------------------------\n")

    prompt = f"""<s>[INST] Нийтлэлийг уншаад асуултан хариул. Хэрэв нийтлэлд багтаагүй асуулт асуувал мэдэхгүй гэж хариул. Хүний талаар сайн муу гэж дүгнэлт гаргаж болохгүй
    Нийтлэл: "Нийслэлийн засаг даргаар ажиллаж байх хугацаандаа Бат-Үүл 2012-2016 онуудад хотыг даргаар ажиллаж байх үедээ 2012 онд 1, 2013 онд 262, 2014 онд 554, 2015 онд 633, 2016 онд 438 удаа 5н жилийн хугацаан ажиллаж байхдаа нийт 1888 удаа шийдвэр гаргаж газар эзэмших болон ашиглах эрх шинээр олгосон байна." 
    
    Асуулт: {user_input} [/INST]"""
    if user_input.lower() == 'exit':
        break

    ended = False
    sio.emit('my_event', {'data': prompt})
    # sio.emit('my_event', {'data': user_input})

# Disconnect from the server
sio.disconnect()
