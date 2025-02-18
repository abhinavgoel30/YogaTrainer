from fastapi import FastAPI
import process_image  # Import the endpoints module

app = FastAPI()

# Include the routes from endpoints.py
app.include_router(process_image.router)

@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}
