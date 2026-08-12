from database import get_connection


connection = get_connection()

if connection.is_connected():

    print("VisionID database connected successfully.")

connection.close()