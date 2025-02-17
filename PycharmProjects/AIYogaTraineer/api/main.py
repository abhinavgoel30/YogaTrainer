import os

import uvicorn
from fastapi import FastAPI
import process_image  # Import the endpoints module

app = FastAPI()

# Include the routes from endpoints.py
app.include_router(process_image.router)

@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=8000)
