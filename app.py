from fastapi.staticfiles import StaticFiles
import uuid
import os
import shutil
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from deepface import DeepFace

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# staticディレクトリを配信
app.mount("/static", StaticFiles(directory="static"), name="static")

# static/uploadsディレクトリを自動作成
os.makedirs("static/uploads", exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

 