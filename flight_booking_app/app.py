import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from dotenv import load_dotenv
import sqlite3
from openai import AzureOpenAI
import json
from emb import chat_with_functions, run_emb, text_to_speech

load_dotenv()

app = Flask(__name__)

client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

from crud_tools import list_bookings, create_booking, update_booking, delete_booking
from emb import text_to_speech, run_emb, search_flights_meta

save_chat = ""

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    bookings = list_bookings()
    return render_template('index.html', bookings=bookings)

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        citizen_id = request.form['citizen_id']
        ticket_price = request.form['ticket_price']
        airplane_code = request.form['airplane_code']
        create_booking(name, email, phone, citizen_id, ticket_price, airplane_code)
        return redirect(url_for('index'))
    return render_template('create.html')

@app.route('/update/<id>', methods=['GET', 'POST'])
def update(id):
    conn = get_db_connection()
    booking = conn.execute('SELECT * FROM bookings WHERE id = ?', (id,)).fetchone()
    conn.close()

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        citizen_id = request.form['citizen_id']
        ticket_price = request.form['ticket_price']
        airplane_code = request.form['airplane_code']
        update_booking(id, name, email, phone, citizen_id, ticket_price, airplane_code)
        return redirect(url_for('index'))
    return render_template('update.html', booking=booking)

@app.route('/delete/<id>')
def delete(id):
    delete_booking(id)
    return redirect(url_for('index'))

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chat_api', methods=['POST'])
def chat_api():
    user_input = request.json.get('message')
    global save_chat
    functions = [
        {
            "name": "list_bookings",
            "description": "Liệt kê tất cả đặt vé",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "create_booking",
            "description": "Tạo đặt vé mới",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "citizen_id": {"type": "string"},
                    "airplane_code": {"type": "string"},
                },
                "required": ["name", "email", "phone", "citizen_id", "airplane_code"],
            },
        },
        {
            "name": "update_booking",
            "description": "Cập nhật đặt vé theo id",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "citizen_id": {"type": "string"},
                    "airplane_code": {"type": "string"},
                },
                "required": ["id", "name", "email", "phone", "citizen_id", "airplane_code"],
            },
        },
        {
            "name": "delete_booking",
            "description": "Xóa đặt vé theo id",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                },
                "required": ["id"],
            },
        },
        {
            "name": "check_plane",
            "description": "Xem thông tin các chuyến bay, kiểm tra lịch trình của các chuyến bay",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    ]

    messages = [{"role": "user", "content": user_input}]
    save_chat = save_chat + "\n -" + user_input

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    message = response.choices[0].message

    if message.function_call:
        func_call = message.function_call
        func_name = func_call.name
        args = json.loads(func_call.arguments)

        if func_name == "list_bookings":
            result = list_bookings()
            save_chat = ''
            if result is None or len(result) == 0:
              content ="Bạn chưa đặt vé nào"
            else:
              content = f"Danh sách đặt vé:\n" + "\n".join([str(b) for b in result])
        elif func_name == "create_booking":
            plane_info = search_flights_meta("hãy lấy thông tin máy bay " + args.get("airplane_code"))
            if plane_info.get("flight_id") != args.get("airplane_code"):
                content = "chuyến bay của bạn không tồn tại. xin hay kiểm tra lại thông tin"
            else:
                content = create_booking(
                    args.get("name"),
                    args.get("email"),
                    args.get("phone"),
                    args.get("citizen_id"),
                    str(plane_info.get("price_economy")),
                    args.get("airplane_code"),
                )
                save_chat = ''
        elif func_name == "update_booking":
            plane_info = search_flights_meta("hãy lấy thông tin máy bay " + args.get("airplane_code"))
            if plane_info.get("flight_id") != args.get("airplane_code"):
                content = "chuyến bay của bạn không tồn tại. xin hay kiểm tra lại thông tin"
            else:
                content = update_booking(
                    args.get("id"),
                    args.get("name"),
                    args.get("email"),
                    args.get("phone"),
                    args.get("citizen_id"),
                    str(plane_info.get("price_economy")),
                    args.get("airplane_code"),
                )
                save_chat = ''
        elif func_name == "delete_booking":
            content = delete_booking(args.get("id"))
            save_chat = ''
        elif func_name == "check_plane":
            content = run_emb(user_input)
            save_chat = ''
        else:
            content = "Không hỗ trợ thao tác này."
            save_chat = ''
        file = text_to_speech(content)

        return jsonify({"response": content, "audio_id": file})
    else:
        content = message.content
        file = text_to_speech(content)
        save_chat = ''
        return jsonify({"response": content, "audio_id": file})

@app.route('/table')
def table():
    bookings = list_bookings()
    return render_template('table.html', bookings=bookings)


@app.route('/get_audio')
def get_audio():
    audio_id = request.args.get('id')
    # Xác định đường dẫn file audio theo audio_id
    output_dir = "./audio"
    path = os.path.join(output_dir, audio_id)  # ví dụ
    try:
        return send_file(path, mimetype='audio/wav')
    except Exception as e:
        print(f"Error sending audio file {path}: {e}")
        return "Audio không tồn tại", 404

# Add this global variable for RAG chat history
chat_rag_history = []

# New endpoint for RAG chat
@app.route('/chat_rag', methods=['POST'])
def chat_rag():
    user_input = request.json.get('message')
    global chat_rag_history
    response, chat_rag_history = chat_with_functions(user_input, chat_rag_history)
    audio_id = text_to_speech(response)
    return jsonify({"response": response, "audio_id": audio_id})

if __name__ == '__main__':
    app.run(debug=True)
