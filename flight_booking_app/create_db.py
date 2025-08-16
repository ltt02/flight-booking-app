import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS bookings (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL,
    citizen_id TEXT NOT NULL,
    ticket_price REAL NOT NULL,
    airplane_code TEXT NOT NULL
)
''')

conn.commit()
conn.close()

print("Database and table created.")
