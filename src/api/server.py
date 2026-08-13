from fastapi import FastAPI
from src.database import get_connection


app = FastAPI(
    title="VisionID API",
    description="Face Recognition and Identity Management API",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "project": "VisionID",
        "message": "VisionID API is running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


@app.get("/users")
def get_users():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
    """
    SELECT id, name, email, student_id
    FROM users
    """
)

    users = cursor.fetchall()

    cursor.close()
    connection.close()


    result = []

    for user in users:

        result.append({
    "id": user[0],
    "name": user[1],
    "email": user[2],
    "student_id": user[3]
})


    return {
        "count": len(result),
        "users": result
    }