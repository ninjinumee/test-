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

# 特徴量抽出関数 (ここは実際の実装に合わせて変更が必要)
def get_embedding(image_path):
    # ここに特徴量抽出のロジックを実装
    # 例: ONNXモデルを使用
    # 画像を読み込み
    img = Image.open(image_path)
    # 画像をリサイズ (DeepFaceのモデルに合わせる)
    img = img.resize((112, 112))
    # 画像をNumPy配列に変換
    img_np = np.array(img)
    # モデルに入力するために前処理
    # 例: 色空間をRGBからBGRに変換
    img_np = img_np[:, :, ::-1]
    # 特徴量を抽出
    # ここに特徴量抽出のロジックを実装
    # 例: ONNXモデルを使用
    # 特徴量はNumPy配列として返す
    return img_np # 実際の特徴量抽出ロジックをここに実装

# コサイン類似度計算関数
def cosine_similarity(emb1, emb2):
    # 特徴量がNumPy配列であることを確認
    if not isinstance(emb1, np.ndarray) or not isinstance(emb2, np.ndarray):
        raise ValueError("特徴量はNumPy配列である必要があります。")
    
    # 特徴量の次元が一致することを確認
    if emb1.shape != emb2.shape:
        raise ValueError("特徴量の次元が一致しません。")
    
    # コサイン類似度を計算
    # ユークリッド距離を使用
    distance = np.linalg.norm(emb1 - emb2)
    # 距離が小さいほど類似度が高いため、1から引いて類似度を計算
    similarity = 1 - distance / np.sqrt(2) # ユークリッド距離の最大値はsqrt(2)
    return similarity

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/verify", response_class=HTMLResponse)
def verify(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # 画像を一時保存
    os.makedirs("static/uploads", exist_ok=True)
    filename1 = f"static/uploads/{uuid.uuid4().hex}_{file1.filename}"
    filename2 = f"static/uploads/{uuid.uuid4().hex}_{file2.filename}"
    with open(filename1, "wb") as buffer1:
        shutil.copyfileobj(file1.file, buffer1)
    with open(filename2, "wb") as buffer2:
        shutil.copyfileobj(file2.file, buffer2)
    # ONNX ArcFace特徴量抽出
    with open(filename1, "rb") as f1:
        emb1 = get_embedding(f1)
    with open(filename2, "rb") as f2:
        emb2 = get_embedding(f2)
    sim = cosine_similarity(emb1, emb2)
    threshold = 0.5
    is_same = sim > threshold
    # DeepFace判定
    try:
        df_result = DeepFace.verify(img1_path=filename1, img2_path=filename2, model_name="ArcFace", enforce_detection=False)
        df_similarity = 1 - df_result["distance"]  # DeepFaceは距離が小さいほど近い
        df_is_same = df_result["verified"]
        df_distance = df_result["distance"]
    except Exception as e:
        df_similarity = None
        df_is_same = None
        df_distance = None
    result = {
        "onnx_arcface": {
            "similarity": f"{sim:.4f}",
            "is_same": is_same
        },
        "deepface": {
            "similarity": f"{df_similarity:.4f}" if df_similarity is not None else "エラー",
            "distance": f"{df_distance:.4f}" if df_distance is not None else "エラー",
            "is_same": df_is_same
        },
        "img1_path": "/" + filename1,
        "img2_path": "/" + filename2
    }
    return templates.TemplateResponse("index.html", {"request": request, "result": result}) 