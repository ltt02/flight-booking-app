import sqlite3
import uuid

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def list_bookings():
    conn = get_db_connection()
    bookings = conn.execute('SELECT * FROM bookings').fetchall()
    conn.close()
    return [dict(b) for b in bookings]

def create_booking(name, email, phone, citizen_id, ticket_price, airplane_code):
    if not (name and email and phone and citizen_id and ticket_price and airplane_code):
        return "Vui lòng điền đầy đủ thông tin."
    if "@" not in email:
        return "Email không hợp lệ."
    try:
        price = float(ticket_price)
        if price < 0:
            return "Giá vé phải là số dương."
    except:
        return "Giá vé không hợp lệ."

    conn = get_db_connection()
    conn.execute(
        'INSERT INTO bookings (id, name, email, phone, citizen_id, ticket_price, airplane_code) VALUES (?,?, ?, ?, ?, ?, ?)',
        (str(uuid.uuid4()), name, email, phone, citizen_id, price, airplane_code)
    )
    conn.commit()
    conn.close()
    return f"Đã tạo đặt vé cho {name}."

def update_booking(id, name, email, phone, citizen_id, ticket_price, airplane_code):
    conn = get_db_connection()
    booking = conn.execute('SELECT * FROM bookings WHERE id = ?', (id,)).fetchone()
    if not booking:
        return f"Không tìm thấy đặt vé với id={id}."
    if not (name and email and phone and citizen_id and ticket_price and airplane_code):
        return "Vui lòng điền đầy đủ thông tin."
    if "@" not in email:
        return "Email không hợp lệ."
    try:
        price = float(ticket_price)
        if price < 0:
            return "Giá vé phải là số dương."
    except:
        return "Giá vé không hợp lệ."

    conn.execute(
        'UPDATE bookings SET name=?, email=?, phone=?, citizen_id=?, ticket_price=?, airplane_code =? WHERE id=?',
        (name, email, phone, citizen_id, price, airplane_code, id)
    )
    conn.commit()
    conn.close()
    return f"Đã cập nhật đặt vé id={id}."

def delete_booking(id):
    conn = get_db_connection()
    booking = conn.execute('SELECT * FROM bookings WHERE id = ?', (id,)).fetchone()
    if not booking:
        return f"Không tìm thấy đặt vé với id={id}."
    conn.execute('DELETE FROM bookings WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    return f"Đã xóa đặt vé id={id}."
