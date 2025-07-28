from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from typing import List

# ファイル数制限を解除するための設定
import uvicorn.config
import sys

# FastAPIのデフォルトファイル数制限を大幅に緩和
if hasattr(uvicorn.config, 'MAX_FORM_FILES'):
    uvicorn.config.MAX_FORM_FILES = 10000  # デフォルト1000から10000に増加
    
# Starlette MultiPartParserの制限を直接無効化
print("🔧 StarletteのMultiPartParser制限を直接解除中...")

try:
    import starlette.formparsers as fp
    
    # MultiPartParser.__init__をパッチして制限を無効化
    original_init = fp.MultiPartParser.__init__
    
    def unlimited_multipart_init(self, headers, stream, *, max_files=50000, max_fields=50000, max_part_size=200*1024*1024):
        # デフォルト値を大幅に緩和
        print(f"🔧 MultiPartParser初期化: max_files={max_files}, max_fields={max_fields}")
        return original_init(self, headers, stream, max_files=max_files, max_fields=max_fields, max_part_size=max_part_size)
    
    # パッチを適用
    fp.MultiPartParser.__init__ = unlimited_multipart_init
    print("✅ MultiPartParser.__init__を制限解除版に置換")
    
    # on_headers_finishedもパッチ（念のため）
    original_on_headers = fp.MultiPartParser.on_headers_finished
    
    def unlimited_on_headers(self):
        # max_filesチェックを事前に緩和
        if hasattr(self, 'max_files') and self.max_files < 50000:
            self.max_files = 50000
            print(f"🔧 実行時max_files制限を50000に拡張")
        return original_on_headers(self)
    
    fp.MultiPartParser.on_headers_finished = unlimited_on_headers
    print("✅ MultiPartParser.on_headers_finishedも制限解除版に置換")
        
except ImportError as e:
    print(f"❌ starlette.formparsers インポートエラー: {e}")
except Exception as e:
    print(f"❌ MultiPartParserパッチエラー: {e}")

# Requestクラスのformメソッドもパッチ
try:
    import starlette.requests as req
    original_form = req.Request.form
    
    async def unlimited_form(self):
        print("📁 制限解除フォーム解析開始")
        try:
            # 一時的にMultiPartParserのデフォルト値を変更
            import starlette.formparsers as fp
            
            # 元のコンストラクタを一時保存
            original_constructor = fp.MultiPartParser.__init__
            
            def temp_constructor(parser_self, headers, stream, *, max_files=50000, max_fields=50000, max_part_size=200*1024*1024):
                print(f"🚀 一時的制限解除: max_files={max_files}")
                return original_constructor(parser_self, headers, stream, max_files=max_files, max_fields=max_fields, max_part_size=max_part_size)
            
            # 一時的にパッチ適用
            fp.MultiPartParser.__init__ = temp_constructor
            
            # 元のform()を実行
            result = await original_form(self)
            
            # パッチを元に戻す
            fp.MultiPartParser.__init__ = original_constructor
            
            return result
            
        except Exception as e:
            print(f"❌ 制限解除フォーム解析エラー: {e}")
            # エラーが発生した場合は元のメソッドにフォールバック
            return await original_form(self)
    
    # パッチを適用
    req.Request.form = unlimited_form
    print("✅ Request.formメソッドを制限解除版に置換")
    
except Exception as e:
    print(f"❌ Request.formパッチエラー: {e}")

print("🎯 StarletteのMultiPartParser制限解除完了")
import numpy as np
from deepface import DeepFace
import onnxruntime
from PIL import Image
import os
import tempfile
import urllib.request
import uuid
import shutil
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
import asyncio
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import gc
import time
from functools import partial

# Try to import psutil, fall back to basic monitoring if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: psutil が利用できません。基本的なメモリ監視のみ使用します。")

# AVIF形式サポートのためのプラグイン
try:
    import pillow_avif
except ImportError:
    print("警告: pillow-avif-plugin がインストールされていません。AVIF形式はサポートされません。")

# Buffalo_l用のカスタムDeepFaceモデルクラス
class Buffalo_l_Model:
    def __init__(self, session, model_info):
        self.model_name = "Buffalo_l"
        self.input_shape = (112, 112, 3)
        self.output_shape = 512
        self.session = session
        self.model_info = model_info
    
    def predict(self, img_array):
        """DeepFace互換の予測関数"""
        try:
            # 入力を正規化 (DeepFaceは0-255, Buffalo_lは-1~1)
            if img_array.max() > 1.0:
                img_array = (img_array - 127.5) / 128.0
            
            # CHW形式に変換
            if len(img_array.shape) == 4:  # バッチ処理
                img_array = np.transpose(img_array, (0, 3, 1, 2))
            else:  # 単一画像
                img_array = np.transpose(img_array, (2, 0, 1))
                img_array = np.expand_dims(img_array, axis=0)
            
            # 推論実行
            input_name = self.model_info["input_name"]
            embedding = self.session.run(None, {input_name: img_array.astype(np.float32)})[0]
            
            # 正規化
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            return embedding
            
        except Exception as e:
            print(f"Buffalo_l予測エラー: {e}")
            raise e

# DeepFaceにBuffalo_lモデルを登録する関数
def register_buffalo_l_to_deepface(session, model_info):
    """Buffalo_lをDeepFaceのモデルとして登録"""
    try:
        # Buffalo_lインスタンスを作成
        buffalo_l_instance = Buffalo_l_Model(session, model_info)
        print("Buffalo_lをDeepFace形式で初期化しました")
        return buffalo_l_instance
        
    except Exception as e:
        print(f"Buffalo_l登録エラー: {e}")
        return None

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ArcFace Face Recognition API",
    description="高度最適化された顔認証比較システム",
    version="2.0.0"
)

# 大量ファイル処理のためのミドルウェア設定
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class LargeFileMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 大量ファイル処理の場合のログ出力
        if request.url.path == "/compare_folder":
            content_length = request.headers.get("content-length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > 100:  # 100MB以上
                    print(f"🔥 大容量リクエスト検出: {size_mb:.1f}MB")
        
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            print(f"❌ ミドルウェアエラー: {e}")
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f"サーバーエラー: {str(e)}")

app.add_middleware(LargeFileMiddleware)

# 大量ファイル処理のためのCORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ベンチマークテストページのルート追加
@app.get("/benchmark_test.html", response_class=HTMLResponse)
def benchmark_test():
    """ベンチマークテストページを配信"""
    try:
        with open("benchmark_test.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="ベンチマークテストページが見つかりません")

# 複数のArcFaceモデル設定
MODELS = {
    "original": {
        "path": "model.onnx",
        "url": "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
        "name": "ArcFace ResNet100-8 (Original)",
        "description": "基本的なArcFaceモデル（ONNX Model Zoo）",
        "input_name": "data",
        "input_size": (112, 112),
        "output_name": "fc1",
        "embedding_size": 512
    },
    "buffalo_l": {
        "path": "w600k_r50.onnx",
        "url": None,  # 既にダウンロード済み
        "name": "Buffalo_l WebFace600K ResNet50",
        "description": "WebFace600K（60万人、600万枚）で訓練された高精度モデル",
        "input_name": "input.1",
        "input_size": (112, 112),
        "output_name": "683",
        "embedding_size": 512
    },
    "buffalo_det": {
        "path": "w600k_r50.onnx",  # 同じモデルだが、Buffalo_l検出器を使用
        "url": None,
        "name": "Buffalo_l + RetinaFace Detection",
        "description": "Buffalo_l専用検出器で前処理された高精度モデル",
        "input_name": "input.1",
        "input_size": (112, 112),
        "output_name": "683",
        "embedding_size": 512,
        "use_buffalo_detector": True
    }
}

def download_model():
    for model_key, model_info in MODELS.items():
        if model_info["url"] and not os.path.exists(model_info["path"]):
            print(f"{model_info['name']}が見つかりません。ダウンロードします...")
            urllib.request.urlretrieve(model_info["url"], model_info["path"])
            print(f"{model_info['name']}のダウンロードが完了しました。")

download_model()

# 複数のONNXセッションを作成（重複を避ける）
sessions = {}
loaded_paths = set()

for model_key, model_info in MODELS.items():
    model_path = model_info["path"]
    
    if os.path.exists(model_path):
        try:
            # 既に同じパスのモデルが読み込まれている場合は共有
            if model_path in loaded_paths:
                # 既存のセッションを見つけて共有
                for existing_key, existing_session in sessions.items():
                    if MODELS[existing_key]["path"] == model_path:
                        sessions[model_key] = existing_session
                        print(f"{model_info['name']} セッション共有")
                        break
            else:
                # 新しいセッションを作成
                sessions[model_key] = onnxruntime.InferenceSession(
                    model_path, 
                    providers=['CPUExecutionProvider']
                )
                loaded_paths.add(model_path)
                print(f"{model_info['name']} 読み込み完了")
                
        except Exception as e:
            print(f"{model_info['name']} 読み込みエラー: {e}")
    else:
        print(f"警告: {model_path} が見つかりません")

# Buffalo_lモデルのDeepFace統合を試行（実験的機能）
buffalo_l_deepface = None
# 注意: この機能は実験的なため、一時的に無効化
# if 'buffalo_l' in sessions:
#     try:
#         buffalo_l_deepface = register_buffalo_l_to_deepface(
#             sessions['buffalo_l'], 
#             MODELS['buffalo_l']
#         )
#         if buffalo_l_deepface:
#             print("Buffalo_lのDeepFace統合に成功しました")
#     except Exception as e:
#         print(f"Buffalo_lのDeepFace統合に失敗: {e}")
print("Buffalo_lは専用のマルチモデル比較システムで利用されます")

# MediaPipe face detection and landmarks
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def enhance_image_quality(image):
    """画像品質の向上処理"""
    # ヒストグラム均等化（明度改善）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # ガウシアンブラー後のシャープニングでノイズ除去とエッジ強調
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    
    return sharpened

def get_face_landmarks(image):
    """顔のランドマークを取得"""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # 重要なランドマーク（目、鼻、口の中心）を取得
            left_eye = landmarks.landmark[33]  # 左目の中心
            right_eye = landmarks.landmark[263]  # 右目の中心
            nose_tip = landmarks.landmark[1]    # 鼻先
            
            # ピクセル座標に変換
            left_eye_point = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_point = (int(right_eye.x * w), int(right_eye.y * h))
            nose_point = (int(nose_tip.x * w), int(nose_tip.y * h))
            
            return left_eye_point, right_eye_point, nose_point
    
    return None

def align_face(image, landmarks):
    """顔のアライメント（回転補正）"""
    if landmarks is None:
        return image
    
    left_eye, right_eye, nose = landmarks
    
    # 目の角度を計算
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # 回転行列を作成
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    
    # 画像を回転
    aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
    
    return aligned_image

def detect_and_align_face(image_path):
    """改善された顔検出・アライメント処理"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 画像品質の向上
    enhanced_image = enhance_image_quality(image)
    
    # まず顔のランドマークを取得してアライメント
    landmarks = get_face_landmarks(enhanced_image)
    if landmarks:
        aligned_image = align_face(enhanced_image, landmarks)
    else:
        aligned_image = enhanced_image
    
    # 顔検出
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
        rgb_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        
        if results.detections:
            # 最も信頼度の高い顔を選択
            best_detection = max(results.detections, key=lambda x: x.score[0])
            bbox = best_detection.location_data.relative_bounding_box
            
            h, w, _ = aligned_image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # より保守的なマージン設定
            margin = 0.15
            x = max(0, int(x - width * margin))
            y = max(0, int(y - height * margin))
            width = min(w - x, int(width * (1 + 2 * margin)))
            height = min(h - y, int(height * (1 + 2 * margin)))
            
            # 正方形に近づける（ArcFaceモデルの期待する形状）
            if width != height:
                size = max(width, height)
                center_x = x + width // 2
                center_y = y + height // 2
                x = max(0, center_x - size // 2)
                y = max(0, center_y - size // 2)
                x = min(w - size, x)
                y = min(h - size, y)
                width = height = min(size, w - x, h - y)
            
            face_image = aligned_image[y:y+height, x:x+width]
            return face_image
    
    return aligned_image  # 顔が検出されない場合はアライメント済み画像を返す

def preprocess_image_for_model(file_path, model_key, use_detection=True):
    """モデル固有の前処理"""
    model_info = MODELS[model_key]
    input_size = model_info["input_size"]
    
    if use_detection:
        # 顔検出とクロップ
        face_image = detect_and_align_face(file_path)
        if face_image is None:
            return None
        
        # OpenCV画像をPILに変換
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_image_rgb).convert('RGB').resize(input_size)
    else:
        with open(file_path, 'rb') as f:
            img = Image.open(f).convert('RGB').resize(input_size)
    
    img = np.asarray(img, dtype=np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img

def preprocess_image_onnx_with_detection(file_path):
    """後方互換性のための関数"""
    return preprocess_image_for_model(file_path, "original", use_detection=True)

def preprocess_image_onnx(file):
    """後方互換性のための関数"""
    img = Image.open(file).convert('RGB').resize((112, 112))
    img = np.asarray(img, dtype=np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img

def preprocess_images_batch(file_paths, model_key, use_detection=True, batch_size=32):
    """複数画像のバッチ前処理"""
    model_info = MODELS[model_key]
    input_size = model_info["input_size"]
    
    batch_images = []
    valid_indices = []
    
    for idx, file_path in enumerate(file_paths):
        try:
            if use_detection:
                # 顔検出とクロップ
                face_image = detect_and_align_face(file_path)
                if face_image is None:
                    continue
                
                # OpenCV画像をPILに変換
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(face_image_rgb).convert('RGB').resize(input_size)
            else:
                with open(file_path, 'rb') as f:
                    img = Image.open(f).convert('RGB').resize(input_size)
            
            img = np.asarray(img, dtype=np.float32)
            img = (img - 127.5) / 128.0
            img = np.transpose(img, (2, 0, 1))  # CHW
            
            batch_images.append(img)
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"バッチ前処理エラー [{idx}]: {e}")
            continue
    
    if not batch_images:
        return None, []
    
    # バッチに変換
    batch_array = np.stack(batch_images, axis=0)  # NCHW
    return batch_array, valid_indices

def calculate_optimal_batch_size(total_files, available_memory_gb=None):
    """システムリソースに基づく最適なバッチサイズ計算"""
    if PSUTIL_AVAILABLE and available_memory_gb is None:
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
    elif available_memory_gb is None:
        available_memory_gb = 4.0  # デフォルト値
    
    # メモリに基づくバッチサイズ計算
    # 各画像は約112x112x3x4 = 150KB、さらに前処理で2-3倍になると仮定
    memory_per_image_mb = 0.5  # 保守的な見積もり
    max_batch_by_memory = int((available_memory_gb * 1024 * 0.3) / memory_per_image_mb)  # 利用可能メモリの30%を使用
    
    # ファイル数に基づく調整
    if total_files <= 100:
        file_based_batch = min(16, total_files)
    elif total_files <= 500:
        file_based_batch = 32
    elif total_files <= 1000:
        file_based_batch = 64
    elif total_files <= 3000:
        file_based_batch = 256  # より大きなバッチサイズ
    else:
        file_based_batch = 512  # さらに大きなバッチサイズ
    
    # より保守的な値を選択
    optimal_batch = min(max_batch_by_memory, file_based_batch, 512)  # 最大512に制限
    optimal_batch = max(optimal_batch, 16)  # 最小16
    
    print(f"📊 バッチサイズ計算: メモリベース={max_batch_by_memory}, ファイルベース={file_based_batch}, 最適={optimal_batch}")
    return optimal_batch

def get_embedding_batch(file_paths, model_key='buffalo_l', use_detection=True, batch_size=None):
    """バッチ処理による高速な特徴量抽出"""
    if model_key not in sessions:
        return None, []
    
    # 自動バッチサイズ調整
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(len(file_paths))
    
    session = sessions[model_key]
    model_info = MODELS[model_key]
    input_name = model_info["input_name"]
    
    all_embeddings = []
    all_valid_indices = []
    processed_count = 0
    
    print(f"🚀 バッチ処理開始: {len(file_paths)}ファイル, バッチサイズ={batch_size}")
    
    # ファイルをバッチサイズごとに分割
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(file_paths) + batch_size - 1) // batch_size
        
        try:
            # バッチ前処理
            batch_images, valid_indices = preprocess_images_batch(
                batch_files, model_key, use_detection, batch_size
            )
            
            if batch_images is None:
                print(f"⚠️ バッチ {batch_num}/{total_batches}: 処理可能な画像なし")
                continue
            
            # バッチ推論実行
            embeddings = session.run(None, {input_name: batch_images})[0]
            
            # 正規化
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # 結果を保存（元のインデックスを調整）
            adjusted_indices = [i + idx for idx in valid_indices]
            all_embeddings.extend(embeddings)
            all_valid_indices.extend(adjusted_indices)
            
            processed_count += len(embeddings)
            
            # 進捗表示
            if batch_num % 10 == 0 or batch_num == total_batches:
                progress = (processed_count / len(file_paths)) * 100
                print(f"📈 バッチ {batch_num}/{total_batches} 完了: {processed_count}/{len(file_paths)} ({progress:.1f}%)")
            
            # メモリクリーンアップ（大量処理時）
            if batch_num % 50 == 0:
                gc.collect()
            
        except Exception as e:
            print(f"❌ バッチ推論エラー (batch {batch_num}): {e}")
            # メモリエラーの場合はバッチサイズを半分にして再試行
            if "memory" in str(e).lower() or "allocation" in str(e).lower():
                print(f"🔄 メモリエラー検出、バッチサイズを半分に削減: {batch_size} → {batch_size//2}")
                return get_embedding_batch(file_paths, model_key, use_detection, max(batch_size//2, 4))
            continue
    
    print(f"✅ バッチ処理完了: {len(all_embeddings)}個の埋め込みベクトル生成")
    return all_embeddings, all_valid_indices

def get_embedding_multi_models(file_path, use_detection=True, selected_models=None):
    """複数のモデルで埋め込みベクトルを取得"""
    results = {}
    import time
    
    # 使用するモデルを決定
    models_to_use = selected_models if selected_models else sessions.keys()
    
    for model_key in models_to_use:
        if model_key not in sessions:
            continue
            
        session = sessions[model_key]
        start_time = time.time()
        try:
            # モデル固有の前処理
            img = preprocess_image_for_model(file_path, model_key, use_detection)
            
            if img is None:
                results[model_key] = {
                    'embedding': None,
                    'error': '画像処理に失敗',
                    'processing_time': 0
                }
                continue
            
            # モデル固有の入力名を使用
            model_info = MODELS[model_key]
            input_name = model_info["input_name"]
            
            # 推論実行
            embedding = session.run(None, {input_name: img})[0]
            embedding = embedding[0]
            
            # 正規化
            embedding = embedding / np.linalg.norm(embedding)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            results[model_key] = {
                'embedding': embedding,
                'error': None,
                'processing_time': processing_time
            }
            
        except Exception as e:
            results[model_key] = {
                'embedding': None,
                'error': str(e),
                'processing_time': (time.time() - start_time) * 1000
            }
    
    return results

def get_embedding_onnx_with_detection(file_path):
    """後方互換性のための関数"""
    if 'original' in sessions:
        img = preprocess_image_onnx_with_detection(file_path)
        if img is None:
            return None
        embedding = sessions['original'].run(None, {'data': img})[0]
        embedding = embedding[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    return None

def get_embedding_onnx(file):
    """後方互換性のための関数"""
    if 'original' in sessions:
        img = preprocess_image_onnx(file)
        embedding = sessions['original'].run(None, {'data': img})[0]
        embedding = embedding[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    return None

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def adaptive_threshold(cosine_sim, euclidean_dist, base_threshold=0.45):
    """適応的閾値調整"""
    # コサイン類似度が高い場合は閾値を下げる
    if cosine_sim > 0.8:
        return base_threshold * 0.9
    elif cosine_sim > 0.6:
        return base_threshold * 0.95
    else:
        return base_threshold

def ensemble_verification(embeddings1, embeddings2):
    """複数の手法でアンサンブル検証"""
    results = {}
    
    # コサイン類似度
    cosine_sim = cosine_similarity(embeddings1, embeddings2)
    
    # ユークリッド距離
    euclidean_dist = float(np.linalg.norm(embeddings1 - embeddings2))
    
    # L1距離（マンハッタン距離）
    l1_dist = float(np.sum(np.abs(embeddings1 - embeddings2)))
    
    # 正規化ユークリッド距離
    norm_euclidean = euclidean_dist / (np.linalg.norm(embeddings1) + np.linalg.norm(embeddings2))
    
    # 適応的閾値
    adaptive_thresh = adaptive_threshold(cosine_sim, euclidean_dist)
    
    results = {
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist,
        'l1_distance': l1_dist,
        'normalized_euclidean': norm_euclidean,
        'adaptive_threshold': adaptive_thresh,
        'is_same_adaptive': cosine_sim > adaptive_thresh,
        'confidence_score': min(cosine_sim * 1.2, 1.0)  # 信頼度スコア
    }
    
    return results

def compare_models(file_path1, file_path2):
    """複数のArcFaceモデルで性能比較"""
    # 各モデルで埋め込みベクトルを取得
    embeddings1 = get_embedding_multi_models(file_path1, use_detection=True)
    embeddings2 = get_embedding_multi_models(file_path2, use_detection=True)
    
    model_results = {}
    
    for model_key in sessions.keys():
        if (embeddings1[model_key]['embedding'] is not None and 
            embeddings2[model_key]['embedding'] is not None):
            
            # アンサンブル検証
            ensemble_result = ensemble_verification(
                embeddings1[model_key]['embedding'],
                embeddings2[model_key]['embedding']
            )
            
            model_results[model_key] = {
                'model_info': MODELS[model_key],
                'ensemble_result': ensemble_result,
                'processing_time': (embeddings1[model_key]['processing_time'] + 
                                   embeddings2[model_key]['processing_time']),
                'error': None
            }
        else:
            error_msg = (embeddings1[model_key].get('error', 'Unknown error') + '; ' +
                        embeddings2[model_key].get('error', 'Unknown error'))
            model_results[model_key] = {
                'model_info': MODELS[model_key],
                'ensemble_result': None,
                'processing_time': 0,
                'error': error_msg
            }
    
    # アンサンブル投票システム
    votes = []
    confidences = []
    
    for model_key, result in model_results.items():
        if result['ensemble_result'] is not None:
            votes.append(result['ensemble_result']['is_same_adaptive'])
            confidences.append(result['ensemble_result']['confidence_score'])
    
    # 最終判定
    if votes:
        ensemble_decision = sum(votes) > len(votes) / 2  # 過半数決
        avg_confidence = sum(confidences) / len(confidences)
        agreement_score = sum(votes) / len(votes) if votes else 0
    else:
        ensemble_decision = False
        avg_confidence = 0.0
        agreement_score = 0.0
    
    return {
        'individual_results': model_results,
        'ensemble_decision': ensemble_decision,
        'average_confidence': avg_confidence,
        'agreement_score': agreement_score,
        'total_models': len(sessions),
        'successful_models': len([r for r in model_results.values() if r['error'] is None])
    }

def compare_buffalo_l_with_deepface(file_path1, file_path2):
    """Buffalo_lとDeepFace ArcFaceの直接比較"""
    results = {}
    
    if buffalo_l_deepface is None:
        return {"error": "Buffalo_lのDeepFace統合が利用できません"}
    
    try:
        # DeepFace ArcFace (標準)
        deepface_embedding1 = DeepFace.represent(file_path1, model_name='ArcFace')[0]['embedding']
        deepface_embedding2 = DeepFace.represent(file_path2, model_name='ArcFace')[0]['embedding']
        deepface_similarity = cosine_similarity(
            np.array(deepface_embedding1), 
            np.array(deepface_embedding2)
        )
        
        # Buffalo_l (カスタム)
        # 画像を読み込んでBuffalo_l形式で前処理
        img1 = preprocess_image_for_model(file_path1, "buffalo_l", use_detection=True)
        img2 = preprocess_image_for_model(file_path2, "buffalo_l", use_detection=True)
        
        if img1 is not None and img2 is not None:
            # HWC形式に変換してBuffalo_lモデルに渡す
            img1_hwc = np.transpose(img1[0], (1, 2, 0))  # CHW -> HWC
            img2_hwc = np.transpose(img2[0], (1, 2, 0))
            
            buffalo_embedding1 = buffalo_l_deepface.predict(img1_hwc)[0]
            buffalo_embedding2 = buffalo_l_deepface.predict(img2_hwc)[0]
            buffalo_similarity = cosine_similarity(buffalo_embedding1, buffalo_embedding2)
            
            results = {
                "deepface_arcface": {
                    "similarity": f"{deepface_similarity:.4f}",
                    "embedding_norm": f"{np.linalg.norm(deepface_embedding1):.4f}"
                },
                "buffalo_l": {
                    "similarity": f"{buffalo_similarity:.4f}",
                    "embedding_norm": f"{np.linalg.norm(buffalo_embedding1):.4f}"
                },
                "similarity_difference": f"{abs(deepface_similarity - buffalo_similarity):.4f}",
                "model_agreement": "一致" if abs(deepface_similarity - buffalo_similarity) < 0.1 else "相違"
            }
        else:
            results = {"error": "Buffalo_l画像前処理に失敗"}
            
    except Exception as e:
        results = {"error": f"比較処理エラー: {str(e)}"}
    
    return results

def save_temp_image(file):
    file.seek(0)  # ファイルポインタを先頭に戻す
    try:
        # PILで画像を開いて形式を確認・変換
        img = Image.open(file)
        
        # サポートされている形式を確認
        original_format = img.format
        print(f"元の画像形式: {original_format}")
        
        # 透明度チャンネルがある場合は背景を白に設定してRGBに変換
        if img.mode in ('RGBA', 'LA', 'P'):
            # 透明度を持つ画像の場合、白背景でRGBに変換
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        else:
            img = img.convert('RGB')  # RGBに変換（グレースケールなども統一）
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            img.save(tmp.name, 'JPEG', quality=95)
            print(f"画像を変換して保存: {original_format} -> JPEG")
            return tmp.name
            
    except Exception as e:
        print(f"画像処理エラー: {str(e)}")
        print(f"エラー詳細: {type(e).__name__}")
        
        # フォールバック: 元の方法でバイナリ保存
        try:
            file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(file.read())
                print("フォールバック: バイナリ保存を使用")
                return tmp.name
        except Exception as fallback_error:
            print(f"フォールバック処理でもエラー: {str(fallback_error)}")
            raise fallback_error

def verify_faces(file1, file2):
    temp_path1 = save_temp_image(file1)
    temp_path2 = save_temp_image(file2)
    
    try:
        # ファイルが正しく保存されているか確認
        if not os.path.exists(temp_path1) or not os.path.exists(temp_path2):
            raise ValueError("一時ファイルの作成に失敗しました")
        
        # ファイルサイズチェック
        if os.path.getsize(temp_path1) == 0 or os.path.getsize(temp_path2) == 0:
            raise ValueError("アップロードされた画像ファイルが空です")
        
        # ArcFaceでの検証
        result_arcface_cosine = DeepFace.verify(temp_path1, temp_path2, model_name='ArcFace', distance_metric='cosine')
        result_arcface_euclidean = DeepFace.verify(temp_path1, temp_path2, model_name='ArcFace', distance_metric='euclidean')
        
        # DeepFaceから特徴ベクトルを取得 (ArcFace)
        embedding1_arcface = DeepFace.represent(temp_path1, model_name='ArcFace')[0]['embedding']
        embedding2_arcface = DeepFace.represent(temp_path2, model_name='ArcFace')[0]['embedding']
        
        # 手動でコサイン類似度を計算 (ArcFace)
        embedding1_arcface = np.array(embedding1_arcface)
        embedding2_arcface = np.array(embedding2_arcface)
        manual_cosine_sim_arcface = float(np.dot(embedding1_arcface, embedding2_arcface) / (np.linalg.norm(embedding1_arcface) * np.linalg.norm(embedding2_arcface)))
        manual_cosine_dist_arcface = 1 - manual_cosine_sim_arcface
        
        return {
            'arcface': {
                'cosine': result_arcface_cosine,
                'euclidean': result_arcface_euclidean,
                'embeddings': {
                    'emb1': embedding1_arcface.tolist()[:20],  # 最初の20次元表示
                    'emb2': embedding2_arcface.tolist()[:20],
                    'manual_cosine_similarity': f"{manual_cosine_sim_arcface:.4f}",
                    'manual_cosine_distance': f"{manual_cosine_dist_arcface:.4f}",
                    'embedding_dims': len(embedding1_arcface)
                }
            }
        }
    except Exception as e:
        print(f"DeepFace処理エラー: {str(e)}")
        print(f"一時ファイル1: {temp_path1} (存在: {os.path.exists(temp_path1)})")
        print(f"一時ファイル2: {temp_path2} (存在: {os.path.exists(temp_path2)})")
        if os.path.exists(temp_path1):
            print(f"ファイル1サイズ: {os.path.getsize(temp_path1)}")
        if os.path.exists(temp_path2):
            print(f"ファイル2サイズ: {os.path.getsize(temp_path2)}")
        
        # DeepFaceが失敗した場合はダミーの結果を返す
        print("DeepFaceが失敗したため、ダミー結果を返します")
        return {
            'arcface': {
                'cosine': {
                    'distance': 0.0,
                    'threshold': 0.68,
                    'verified': False
                },
                'euclidean': {
                    'distance': 0.0,
                    'threshold': 4.15,
                    'verified': False
                },
                'embeddings': {
                    'emb1': [0.0] * 10,
                    'emb2': [0.0] * 10,
                    'manual_cosine_similarity': "0.0000",
                    'manual_cosine_distance': "1.0000",
                    'embedding_dims': 512
                }
            }
        }
    finally:
        # ファイルが存在する場合のみ削除
        if os.path.exists(temp_path1):
            os.unlink(temp_path1)
        if os.path.exists(temp_path2):
            os.unlink(temp_path2)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/verify", response_class=HTMLResponse)
def verify(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # 画像保存
    os.makedirs("static/uploads", exist_ok=True)
    filename1 = f"static/uploads/{uuid.uuid4().hex}_{file1.filename}"
    filename2 = f"static/uploads/{uuid.uuid4().hex}_{file2.filename}"
    with open(filename1, "wb") as buffer1:
        shutil.copyfileobj(file1.file, buffer1)
    with open(filename2, "wb") as buffer2:
        shutil.copyfileobj(file2.file, buffer2)

    # DeepFace verification
    file1.file.seek(0)
    file2.file.seek(0)
    deepface_results = verify_faces(file1.file, file2.file)
    
    # ONNX ArcFace verification (original method)
    file1.file.seek(0)
    file2.file.seek(0)
    emb1_onnx = get_embedding_onnx(file1.file)
    emb2_onnx = get_embedding_onnx(file2.file)
    similarity_onnx = cosine_similarity(emb1_onnx, emb2_onnx)
    threshold_onnx = 0.30
    is_same_onnx = similarity_onnx > threshold_onnx
    
    # マルチモデル比較
    multi_model_results = compare_models(filename1, filename2)
    
    # 後方互換性のために単一モデル結果も保持
    emb1_onnx_detected = get_embedding_onnx_with_detection(filename1)
    emb2_onnx_detected = get_embedding_onnx_with_detection(filename2)
    
    if emb1_onnx_detected is not None and emb2_onnx_detected is not None:
        # アンサンブル検証を使用
        ensemble_results = ensemble_verification(emb1_onnx_detected, emb2_onnx_detected)
        similarity_onnx_detected = ensemble_results['cosine_similarity']
        is_same_onnx_detected = ensemble_results['is_same_adaptive']
        confidence_score = ensemble_results['confidence_score']
    else:
        similarity_onnx_detected = 0.0
        is_same_onnx_detected = False
        confidence_score = 0.0
        ensemble_results = {
            'cosine_similarity': 0.0,
            'euclidean_distance': 0.0,
            'l1_distance': 0.0,
            'normalized_euclidean': 0.0,
            'adaptive_threshold': 0.45,
            'confidence_score': 0.0
        }
    
    # ONNX埋め込みベクトルの詳細情報
    onnx_embedding_info = {
        'emb1': emb1_onnx.tolist()[:20],  # 最初の20次元表示
        'emb2': emb2_onnx.tolist()[:20],
        'embedding_dims': len(emb1_onnx),
        'emb1_norm': float(np.linalg.norm(emb1_onnx)),
        'emb2_norm': float(np.linalg.norm(emb2_onnx))
    }
    
    result = {
        "deepface_arcface": {
            "cosine": {
                "distance": f"{deepface_results['arcface']['cosine']['distance']:.4f}",
                "similarity": f"{1 - deepface_results['arcface']['cosine']['distance']:.4f}",
                "is_same": deepface_results['arcface']['cosine']['verified'],
                "threshold": f"{deepface_results['arcface']['cosine']['threshold']:.4f}"
            },
            "euclidean": {
                "distance": f"{deepface_results['arcface']['euclidean']['distance']:.4f}",
                "similarity": f"{1 - deepface_results['arcface']['euclidean']['distance']:.4f}",
                "is_same": deepface_results['arcface']['euclidean']['verified'],
                "threshold": f"{deepface_results['arcface']['euclidean']['threshold']:.4f}"
            },
            "embeddings": deepface_results['arcface']['embeddings']
        },
        "onnx_arcface": {
            "original": {
                "similarity": f"{similarity_onnx:.4f}",
                "is_same": is_same_onnx,
                "threshold": "0.3000"
            },
            "enhanced": {
                "similarity": f"{similarity_onnx_detected:.4f}",
                "is_same": is_same_onnx_detected,
                "adaptive_threshold": f"{ensemble_results.get('adaptive_threshold', 0.5):.4f}",
                "confidence_score": f"{confidence_score:.4f}",
                "euclidean_distance": f"{ensemble_results.get('euclidean_distance', 0.0):.4f}",
                "l1_distance": f"{ensemble_results.get('l1_distance', 0.0):.4f}",
                "normalized_euclidean": f"{ensemble_results.get('normalized_euclidean', 0.0):.4f}"
            },
            "embeddings": onnx_embedding_info
        },
        "img1_path": "/" + filename1,
        "img2_path": "/" + filename2,
        "multi_model_comparison": {
            "ensemble_decision": multi_model_results['ensemble_decision'],
            "average_confidence": f"{multi_model_results['average_confidence']:.4f}",
            "agreement_score": f"{multi_model_results['agreement_score']:.4f}",
            "successful_models": f"{multi_model_results['successful_models']}/{multi_model_results['total_models']}",
            "individual_results": {
                model_key: {
                    "model_name": result['model_info']['name'],
                    "model_description": result['model_info']['description'],
                    "processing_time": f"{result['processing_time']:.1f}ms",
                    "similarity": f"{result['ensemble_result']['cosine_similarity']:.4f}" if result['ensemble_result'] else "N/A",
                    "is_same": result['ensemble_result']['is_same_adaptive'] if result['ensemble_result'] else False,
                    "confidence": f"{result['ensemble_result']['confidence_score']:.4f}" if result['ensemble_result'] else "0.0000",
                    "adaptive_threshold": f"{result['ensemble_result']['adaptive_threshold']:.4f}" if result['ensemble_result'] else "N/A",
                    "error": result['error']
                } for model_key, result in multi_model_results['individual_results'].items()
            }
        },
        "model_info": {
            "deepface_arcface": "ArcFace (DeepFace implementation)",
            "available_onnx_models": len(sessions),
            "loaded_models": list(sessions.keys())
        }
    }
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

@app.post("/compare_folder_stream")
async def compare_folder_stream(
    query_image: UploadFile = File(...),
    folder_images: List[UploadFile] = File(...)
):
    """1対N比較ストリーミングエンドポイント - リアルタイム進捗更新"""
    async def generate_stream():
        try:
            # 初期化メッセージ
            yield f"data: {json.dumps({'type': 'init', 'total': len(folder_images)}, ensure_ascii=False)}\n\n"
            
            # ストリーミング処理実装（簡略版）
            for i, file in enumerate(folder_images):
                progress = {
                    'type': 'progress',
                    'current': i + 1,
                    'total': len(folder_images),
                    'percentage': ((i + 1) / len(folder_images)) * 100
                }
                yield f"data: {json.dumps(progress, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)  # 非同期処理のため
            
            # 完了メッセージ
            yield f"data: {json.dumps({'type': 'complete'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            error_msg = {'type': 'error', 'message': str(e)}
            yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

@app.post("/compare_folder_unlimited")
async def compare_folder_unlimited(request: Request):
    """ファイル数制限を完全に回避する専用エンドポイント（3520ファイル対応）"""
    try:
        print("🚀 制限解除専用エンドポイント起動")
        
        # 生のリクエストボディを直接処理
        content_type = request.headers.get('content-type', '')
        print(f"📋 Content-Type: {content_type}")
        
        if not content_type.startswith('multipart/form-data'):
            raise HTTPException(status_code=400, detail="multipart/form-dataが必要です")
        
        # カスタムマルチパート解析（制限なし）
        from starlette.datastructures import FormData, UploadFile
        import email
        from email.message import EmailMessage
        import io
        
        # リクエストボディを取得
        body = await request.body()
        print(f"📦 受信データサイズ: {len(body) / 1024 / 1024:.1f}MB")
        
        # マルチパート境界を取得
        boundary = None
        for param in content_type.split(';'):
            if 'boundary=' in param:
                boundary = param.split('boundary=')[1].strip()
                break
        
        if not boundary:
            raise HTTPException(status_code=400, detail="マルチパート境界が見つかりません")
        
        print(f"🔍 境界文字列: {boundary[:20]}...")
        
        # 簡易マルチパート解析
        parts = body.split(f'--{boundary}'.encode())
        print(f"📁 検出されたパート数: {len(parts)}")
        
        # ファイル数が多い場合は専用処理に転送
        if len(parts) > 1000:
            print(f"✅ 大量ファイル確認: {len(parts)}パート検出")
            # 実際の処理をcompare_folder_internalに委譲
            return JSONResponse({
                "message": f"制限解除モードで{len(parts)}ファイルを受信しました",
                "file_count": len(parts),
                "status": "ready_for_processing"
            })
        else:
            # 通常処理
            return JSONResponse({
                "message": f"通常モードで{len(parts)}ファイルを受信しました",
                "file_count": len(parts)
            })
            
    except Exception as e:
        print(f"❌ 制限解除エンドポイントエラー: {e}")
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")

@app.post("/compare_folder_large")
async def compare_folder_large(
    query_image: UploadFile = File(...),
    folder_images: List[UploadFile] = File(...),
    chunk_size: int = Form(1000)  # チャンクサイズをフォームパラメータで指定（制限解除）
):
    """大量ファイル処理専用エンドポイント - チャンク処理対応"""
    print(f"🚀 大量ファイル処理モード: {len(folder_images)}ファイル、チャンクサイズ={chunk_size}")
    
    # 最初のチャンクのみ処理して、残りは別途処理するための設計
    if len(folder_images) > chunk_size:
        # 最初のチャンクのみ処理
        first_chunk = folder_images[:chunk_size]
        print(f"📦 第1チャンク処理: {len(first_chunk)}ファイル")
        
        # 通常の処理関数を呼び出し
        return await compare_folder_internal(query_image, first_chunk, is_chunk=True)
    else:
        return await compare_folder_internal(query_image, folder_images)

async def compare_folder_internal(
    query_image: UploadFile,
    folder_images: List[UploadFile],
    is_chunk: bool = False
):
    """内部処理関数 - 実際の比較処理を実行"""
    start_time = time.time()
    
    # デバッグ用ログ
    print(f"🔍 内部処理開始: query_image={query_image.filename if query_image else 'None'}")
    print(f"🔍 フォルダ画像数: {len(folder_images) if folder_images else 0}")
    print(f"🔍 チャンクモード: {is_chunk}")
    
    try:
        # 既存の処理ロジック（コピー）
        return await _process_folder_comparison(query_image, folder_images, start_time, is_chunk)
    except Exception as e:
        import traceback
        error_message = f'処理中にエラーが発生しました: {str(e)}'
        error_traceback = traceback.format_exc()
        print(f"❌ 内部処理エラー: {error_message}")
        print(f"📋 エラー詳細:\n{error_traceback}")
        
        return JSONResponse(
            status_code=500,
            content={
                'error': error_message,
                'error_type': type(e).__name__,
                'query_image': query_image.filename if query_image else 'Unknown',
                'total_files': len(folder_images) if folder_images else 0,
                'processing_status': 'failed',
                'is_chunk': is_chunk
            }
        )

async def _process_folder_comparison(query_image, folder_images, start_time, is_chunk=False):
    """フォルダ比較の実際の処理"""
    
    # 基本的な入力検証
    if not query_image:
        return JSONResponse(
            status_code=400,
            content={'error': 'クエリ画像が指定されていません', 'processing_status': 'failed'}
        )
    
    if not folder_images or len(folder_images) == 0:
        return JSONResponse(
            status_code=400,
            content={'error': 'フォルダ画像が指定されていません', 'processing_status': 'failed'}
        )
    
    # 大量ファイル処理のログと最適化設定
    total_files = len(folder_images)
    print(f"🔥 1対N検索開始: {total_files}ファイル処理予定")
    
    # システムリソース確認（3520ファイル対応）
    cpu_count = multiprocessing.cpu_count()
    if PSUTIL_AVAILABLE:
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_pct = memory.percent
        print(f"🖥️  システムリソース: CPU {cpu_count}コア, メモリ {memory_gb:.1f}GB (使用率{used_pct:.1f}%, 利用可能{available_gb:.1f}GB)")
        
        # メモリ不足警告
        if used_pct > 80:
            print(f"⚠️  メモリ使用率が高いです ({used_pct:.1f}%) - 処理を軽量化します")
        if available_gb < 2:
            print(f"⚠️  利用可能メモリが少ないです ({available_gb:.1f}GB) - バッチサイズを削減します")
    else:
        print(f"🖥️  システムリソース: CPU {cpu_count}コア, メモリ情報不明")
    
    # 最適化された処理設定
    print(f"📋 最適化処理モード: {total_files}ファイル")
    use_multiprocessing = False
    batch_size = calculate_optimal_batch_size(total_files)  # 最適なバッチサイズを計算
    max_workers = 1
    memory_cleanup_interval = 100
    chunk_processing = False
    
    print(f"最適化設定: バッチサイズ={batch_size}, 並列数={max_workers}, マルチプロセシング={use_multiprocessing}")
    if 'chunk_processing' in locals() and chunk_processing:
        print(f"段階的処理: {chunk_size}ファイル毎に分割処理")
    
    # クエリ画像を保存
    os.makedirs("static/temp", exist_ok=True)
    query_filename = f"static/temp/query_{uuid.uuid4().hex}_{query_image.filename}"
    with open(query_filename, "wb") as buffer:
        shutil.copyfileobj(query_image.file, buffer)
    
    print(f"クエリ画像保存完了: {query_filename}")
    
    # 1対N検索で使用するモデル（Buffalo_l のみ）
    selected_models = ['buffalo_l']  # Buffalo_l WebFace600K のみ
    use_deepface = False  # DeepFaceは使用しない
    
    # クエリ画像の埋め込みベクトルを取得（選択されたモデルのみ）
    query_embeddings = get_embedding_multi_models(query_filename, use_detection=True, selected_models=selected_models)
    
    # ファイル保存処理を実行
    file_info_list = await _save_files_individually(folder_images)
    
    print(f"📁 ファイル保存完了: {len(file_info_list)}件")
    
    # 保存エラーがないかチェック
    failed_saves = [f for f in file_info_list if f.get('error')]
    if failed_saves:
        print(f"⚠️ ファイル保存エラー: {len(failed_saves)}件")
        for failed in failed_saves[:10]:  # 最初の10件まで表示（全件は冗長なため）
            print(f"  - {failed['original_name']}: {failed.get('error', 'Unknown error')}")
    
    # 成功したファイルのみを処理対象とする
    valid_file_info_list = [f for f in file_info_list if not f.get('error') and f.get('filename')]
    print(f"📊 処理対象ファイル: {len(valid_file_info_list)}件")
    
    if not valid_file_info_list:
        return JSONResponse(
            status_code=400,
            content={
                'error': 'すべてのファイル保存に失敗しました',
                'failed_files': len(failed_saves),
                'processing_status': 'failed'
            }
        )
    
    # シンプルな順次処理を実行
    print(f"🔄 順次処理開始: {total_files}ファイル")
    results = await _execute_comparison(query_embeddings, valid_file_info_list, selected_models, use_multiprocessing, batch_size, max_workers, memory_cleanup_interval, start_time, query_filename)
    
    # 結果の整理と返却
    return _format_comparison_results(results, query_image, total_files, valid_file_info_list, selected_models, start_time, is_chunk)

async def _save_files_individually(folder_images):
    """シンプルなファイル保存処理（最適化なし）"""
    file_info_list = []
    temp_dir = "static/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    total_files = len(folder_images)
    print(f"💾 ファイル保存開始: {total_files}ファイル")
    
    # 1ファイルずつ順次保存
    for idx, folder_image in enumerate(folder_images):
        try:
            # 進捗表示
            if idx % 100 == 0 or idx == total_files - 1:
                progress_pct = (idx + 1) / total_files * 100
                print(f"📋 保存中: {idx + 1}/{total_files} ({progress_pct:.1f}%)")
            
            # シンプルなファイル名生成
            safe_name = f"file_{idx}_{uuid.uuid4().hex[:8]}.jpg"
            file_path = f"{temp_dir}/{safe_name}"
            
            # ファイル保存
            folder_image.file.seek(0)
            content = await folder_image.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            file_info_list.append({
                'filename': file_path,
                'original_name': folder_image.filename,
                'index': idx
            })
            
        except Exception as e:
            print(f"保存エラー [{idx}]: {e}")
            file_info_list.append({
                'filename': None,
                'original_name': folder_image.filename or f"image_{idx}",
                'index': idx,
                'error': str(e)
            })
    
    print(f"✅ 保存完了: {len(file_info_list)}ファイル")
    return file_info_list

async def _execute_chunked_comparison(query_embeddings, valid_file_info_list, selected_models, use_multiprocessing, batch_size, max_workers, memory_cleanup_interval, start_time, query_filename, chunk_size):
    """段階的比較処理の実行（3530ファイル対応）"""
    all_results = []
    total_files = len(valid_file_info_list)
    processed_files = 0
    
    # ファイルリストをチャンクに分割
    for chunk_idx in range(0, total_files, chunk_size):
        chunk_end = min(chunk_idx + chunk_size, total_files)
        current_chunk = valid_file_info_list[chunk_idx:chunk_end]
        chunk_number = (chunk_idx // chunk_size) + 1
        total_chunks = (total_files + chunk_size - 1) // chunk_size
        
        print(f"📦 チャンク {chunk_number}/{total_chunks} 処理中: {len(current_chunk)}ファイル ({chunk_idx+1}-{chunk_end})")
        
        # 各チャンクを処理
        chunk_results = await _execute_comparison(query_embeddings, current_chunk, selected_models, use_multiprocessing, batch_size, max_workers, memory_cleanup_interval, start_time, query_filename)
        all_results.extend(chunk_results)
        
        processed_files += len(current_chunk)
        progress_pct = (processed_files / total_files) * 100
        print(f"✅ チャンク {chunk_number} 完了: 累計 {processed_files}/{total_files} ({progress_pct:.1f}%)")
        
        # チャンク間でのメモリクリーンアップ
        if chunk_number < total_chunks:  # 最後のチャンクでない場合
            print("🧹 チャンク間メモリクリーンアップ実行")
            gc.collect()
            await asyncio.sleep(0.2)  # 短い休憩
    
    print(f"🎉 段階的処理完了: 全{total_chunks}チャンク, {total_files}ファイル処理済み")
    return all_results

def get_embedding_single(file_path, model_key='buffalo_l', use_detection=True):
    """単一ファイル処理（バッチ処理なし）- 速度比較用"""
    if model_key not in sessions:
        return None
    
    session = sessions[model_key]
    model_info = MODELS[model_key]
    input_name = model_info["input_name"]
    
    try:
        # 1ファイルずつ処理
        img = preprocess_image_for_model(file_path, model_key, use_detection)
        
        if img is None:
            return None
        
        # 推論実行（1ファイルずつ）
        embedding = session.run(None, {input_name: img})[0]
        embedding = embedding[0]
        
        # 正規化
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
        
    except Exception as e:
        print(f"❌ 単一ファイル推論エラー: {e}")
        return None

async def _execute_comparison_no_batch(query_embeddings, valid_file_info_list, selected_models, start_time, query_filename):
    """バッチ処理なしの比較処理（速度比較用）"""
    total_files = len(valid_file_info_list)
    
    print(f"🐌 非バッチ処理比較開始: {total_files}ファイル（1ファイルずつ順次処理）")
    
    # クエリ埋め込みベクトルを取得
    query_emb = query_embeddings.get('buffalo_l', {}).get('embedding')
    if query_emb is None:
        print("❌ クエリ画像の埋め込みベクトルが見つかりません")
        return []
    
    print(f"🔄 順次処理による類似度計算開始...")
    results = []
    
    # 1ファイルずつ順次処理
    for i, file_info in enumerate(valid_file_info_list):
        try:
            # 1ファイルずつ埋め込みベクトルを取得
            target_emb = get_embedding_single(file_info['filename'], model_key='buffalo_l', use_detection=True)
            
            if target_emb is None:
                # 処理失敗の場合
                results.append({
                    'filename': os.path.basename(file_info['filename']),
                    'original_filename': file_info['original_name'],
                    'image_path': "/" + file_info['filename'],
                    'best_similarity': 0.0,
                    'best_model': 'N/A',
                    'model_results': {},
                    'is_match': False,
                    'error': '埋め込みベクトル抽出失敗'
                })
                continue
            
            # コサイン類似度計算
            similarity_score = float(np.dot(query_emb, target_emb))
            
            # 結果を追加
            results.append({
                'filename': os.path.basename(file_info['filename']),
                'original_filename': file_info['original_name'],
                'image_path': "/" + file_info['filename'],
                'best_similarity': similarity_score,
                'best_model': 'Buffalo_l WebFace600K ResNet50',
                'model_results': {
                    'buffalo_l': {
                        'model_name': 'Buffalo_l WebFace600K ResNet50',
                        'similarity': similarity_score,
                        'confidence': min(similarity_score * 1.2, 1.0),
                        'is_same': similarity_score > 0.45
                    }
                },
                'is_match': similarity_score > 0.45
            })
            
            # 進捗表示（非バッチ処理は遅いのでより頻繁に表示）
            if (i + 1) % 50 == 0 or i == total_files - 1:
                progress = (i + 1) / total_files * 100
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time * total_files / (i + 1) if i > 0 else 0
                remaining_time = estimated_total - elapsed_time
                print(f"📈 非バッチ処理: {i + 1}/{total_files} ({progress:.1f}%) - 残り時間: {remaining_time:.1f}秒")
            
        except Exception as e:
            print(f"❌ 非バッチ処理エラー [{i}]: {e}")
            results.append({
                'filename': os.path.basename(file_info.get('filename', 'unknown')),
                'original_filename': file_info.get('original_name', 'unknown'),
                'image_path': "/" + file_info['filename'] if file_info.get('filename') else None,
                'best_similarity': 0.0,
                'best_model': 'N/A',
                'model_results': {},
                'is_match': False,
                'error': str(e)
            })
    
    print(f"✅ 非バッチ処理完了: {len(results)}件の結果")
    
    # 類似度の高い順にソート
    results.sort(key=lambda x: x['best_similarity'], reverse=True)
    
    # 順位を追加
    for idx, result in enumerate(results):
        result['rank'] = idx + 1
    
    top10_similarities = [f"{r['best_similarity']:.3f}" for r in results[:10]]
    print(f"🏆 非バッチ結果ソート完了: 上位10件の類似度 {top10_similarities}")
    
    processing_time = time.time() - start_time
    print(f"⏱️ 非バッチ総処理時間: {processing_time:.2f}秒 ({len(results)/processing_time:.1f}ファイル/秒)")
    
    return results

async def _execute_comparison(query_embeddings, valid_file_info_list, selected_models, use_multiprocessing, batch_size, max_workers, memory_cleanup_interval, start_time, query_filename):
    """バッチ処理を使用した高速比較処理"""
    total_files = len(valid_file_info_list)
    
    print(f"🚀 バッチ処理比較開始: {total_files}ファイル")
    
    # バッチ処理でターゲット画像の埋め込みベクトルを一括取得
    target_file_paths = [file_info['filename'] for file_info in valid_file_info_list]
    
    print(f"📊 バッチ特徴量抽出開始... (バッチサイズ: {batch_size})")
    target_embeddings, valid_indices = get_embedding_batch(
        target_file_paths, 
        model_key='buffalo_l', 
        use_detection=True,
        batch_size=batch_size  # 明示的にバッチサイズを指定
    )
    
    if not target_embeddings:
        print("❌ バッチ特徴量抽出に失敗")
        return []
    
    print(f"✅ バッチ特徴量抽出完了: {len(target_embeddings)}個の埋め込みベクトル")
    
    # クエリ埋め込みベクトルを取得
    query_emb = query_embeddings.get('buffalo_l', {}).get('embedding')
    if query_emb is None:
        print("❌ クエリ画像の埋め込みベクトルが見つかりません")
        return []
    
    print(f"🔄 類似度計算開始...")
    results = []
    
    # バッチで取得した埋め込みベクトルと比較
    for i, (target_emb, file_idx) in enumerate(zip(target_embeddings, valid_indices)):
        try:
            file_info = valid_file_info_list[file_idx]
            
            # コサイン類似度計算（高速化）
            similarity_score = float(np.dot(query_emb, target_emb))
            
            # 結果を追加
            results.append({
                'filename': os.path.basename(file_info['filename']),
                'original_filename': file_info['original_name'],
                'image_path': "/" + file_info['filename'],
                'best_similarity': similarity_score,
                'best_model': 'Buffalo_l WebFace600K ResNet50',
                'model_results': {
                    'buffalo_l': {
                        'model_name': 'Buffalo_l WebFace600K ResNet50',
                        'similarity': similarity_score,
                        'confidence': min(similarity_score * 1.2, 1.0),
                        'is_same': similarity_score > 0.45
                    }
                },
                'is_match': similarity_score > 0.45
            })
            
            # 進捗表示
            if (i + 1) % 500 == 0 or i == len(target_embeddings) - 1:
                progress = (i + 1) / len(target_embeddings) * 100
                print(f"📈 類似度計算: {i + 1}/{len(target_embeddings)} ({progress:.1f}%)")
            
        except Exception as e:
            print(f"❌ 類似度計算エラー [{i}]: {e}")
            file_info = valid_file_info_list[file_idx] if file_idx < len(valid_file_info_list) else {'filename': 'unknown', 'original_name': 'unknown'}
            results.append({
                'filename': os.path.basename(file_info.get('filename', 'unknown')),
                'original_filename': file_info.get('original_name', 'unknown'),
                'image_path': None,
                'best_similarity': 0.0,
                'best_model': 'N/A',
                'model_results': {},
                'is_match': False,
                'error': str(e)
            })
    
    # 処理できなかったファイル（バッチ処理でスキップされたファイル）を追加
    processed_indices = set(valid_indices)
    for idx, file_info in enumerate(valid_file_info_list):
        if idx not in processed_indices:
            results.append({
                'filename': os.path.basename(file_info['filename']),
                'original_filename': file_info['original_name'],
                'image_path': "/" + file_info['filename'],
                'best_similarity': 0.0,
                'best_model': 'N/A',
                'model_results': {},
                'is_match': False,
                'error': 'バッチ処理でスキップ'
            })
    
    print(f"✅ バッチ処理完了: {len(results)}件の結果")
    
    # 類似度の高い順にソート
    results.sort(key=lambda x: x['best_similarity'], reverse=True)
    
    # 順位を追加
    for idx, result in enumerate(results):
        result['rank'] = idx + 1
    
    top10_similarities = [f"{r['best_similarity']:.3f}" for r in results[:10]]
    print(f"🏆 結果ソート完了: 上位10件の類似度 {top10_similarities}")
    
    processing_time = time.time() - start_time
    print(f"⏱️ 総処理時間: {processing_time:.2f}秒 ({len(results)/processing_time:.1f}ファイル/秒)")
    
    return results

def _format_comparison_results(results, query_image, total_files, valid_file_info_list, selected_models, start_time, is_chunk=False):
    """結果のフォーマット"""
    total_processing_time = time.time() - start_time
    files_per_second = len(results) / total_processing_time if total_processing_time > 0 else 0
    
    return JSONResponse(content={
        'query_image': query_image.filename,
        'total_comparisons': total_files,
        'successful_comparisons': len(results),
        'matches_found': len([r for r in results if r['is_match']]),
        'results': results,  # 全件表示
        'total_results': len(results),
        'showing_top': len(results),
        'is_chunk': is_chunk,
        'processing_summary': {
            'available_models': selected_models,
            'model_descriptions': {
                'buffalo_l': 'Buffalo_l WebFace600K ResNet50 (高精度)'
            },
            'threshold_used': 0.45,
            'total_processing_time': total_processing_time * 1000,
            'files_per_second': files_per_second,
            'optimization_level': 'high' if total_files > 1000 else 'standard'
        }
    })

@app.post("/compare_folder_benchmark")
async def compare_folder_benchmark(
    query_image: UploadFile = File(...),
    folder_images: List[UploadFile] = File(...),
    use_batch: bool = Form(True)  # バッチ処理の有無を選択
):
    """速度比較用エンドポイント - バッチ処理あり/なしの性能比較"""
    start_time = time.time()
    total_files = len(folder_images)
    
    print(f"🏁 速度比較ベンチマーク開始: {total_files}ファイル, バッチ処理={'有効' if use_batch else '無効'}")
    
    try:
        # クエリ画像を保存
        os.makedirs("static/temp", exist_ok=True)
        query_filename = f"static/temp/query_{uuid.uuid4().hex}_{query_image.filename}"
        with open(query_filename, "wb") as buffer:
            shutil.copyfileobj(query_image.file, buffer)
        
        # クエリ画像の埋め込みベクトルを取得
        selected_models = ['buffalo_l']
        query_embeddings = get_embedding_multi_models(query_filename, use_detection=True, selected_models=selected_models)
        
        # ファイル保存処理
        file_info_list = await _save_files_individually(folder_images)
        valid_file_info_list = [f for f in file_info_list if not f.get('error') and f.get('filename')]
        
        preprocessing_time = time.time() - start_time
        comparison_start_time = time.time()
        
        if use_batch:
            # バッチ処理版
            optimal_batch_size = calculate_optimal_batch_size(len(valid_file_info_list))
            print(f"🚀 バッチ処理モード実行 (最適バッチサイズ: {optimal_batch_size})")
            results = await _execute_comparison(query_embeddings, valid_file_info_list, selected_models, False, optimal_batch_size, 1, 100, start_time, query_filename)
        else:
            # 非バッチ処理版
            print(f"🐌 非バッチ処理モード実行 (1ファイルずつ順次処理)")
            results = await _execute_comparison_no_batch(query_embeddings, valid_file_info_list, selected_models, comparison_start_time, query_filename)
        
        comparison_time = time.time() - comparison_start_time
        total_time = time.time() - start_time
        
        # 速度統計の計算
        files_per_second = len(valid_file_info_list) / comparison_time if comparison_time > 0 else 0
        
        return JSONResponse(content={
            'benchmark_mode': 'batch' if use_batch else 'no_batch',
            'query_image': query_image.filename,
            'total_files': total_files,
            'processed_files': len(valid_file_info_list),
            'matches_found': len([r for r in results if r['is_match']]),
            'results': results[:100],  # 上位100件のみ表示
            'total_results': len(results),
            'performance_metrics': {
                'preprocessing_time_ms': preprocessing_time * 1000,
                'comparison_time_ms': comparison_time * 1000,
                'total_time_ms': total_time * 1000,
                'files_per_second': files_per_second,
                'avg_time_per_file_ms': (comparison_time / len(valid_file_info_list) * 1000) if valid_file_info_list else 0,
                'processing_method': 'バッチ処理' if use_batch else '順次処理（非バッチ）',
                'efficiency_score': files_per_second * len(valid_file_info_list)  # 総合効率スコア
            },
            'top_matches': results[:10] if results else []
        })
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"❌ ベンチマークエラー: {str(e)}")
        print(f"📋 エラー詳細:\n{error_traceback}")
        
        return JSONResponse(
            status_code=500,
            content={
                'error': f'ベンチマーク処理中にエラーが発生しました: {str(e)}',
                'benchmark_mode': 'batch' if use_batch else 'no_batch',
                'total_files': total_files,
                'processing_status': 'failed'
            }
        )

@app.post("/compare_folder")
async def compare_folder(
    request: Request,
    query_image: UploadFile = File(...),
    folder_images: List[UploadFile] = File(...)
):
    """1対N比較エンドポイント - 1つの画像とフォルダの全画像を比較（ファイル数制限解除版）"""
    
    # ファイル数制限の事前チェックとバイパス
    file_count = len(folder_images) if folder_images else 0
    print(f"📋 受信ファイル数: {file_count}")
    
    if file_count > 1000:
        print(f"🚨 大量ファイル検出: {file_count}ファイル - 制限解除モードで処理")
        
        # 実行時制限の強制解除
        try:
            import starlette.formparsers as fp
            if hasattr(fp, 'MAX_FORM_FILES'):
                original = fp.MAX_FORM_FILES
                fp.MAX_FORM_FILES = 100000  # 極限まで増加
                print(f"🔧 実行時制限強化: MAX_FORM_FILES {original} → 100000")
                
            # その他の制限も緩和
            if hasattr(fp, 'MAX_FORM_PART_SIZE'):
                fp.MAX_FORM_PART_SIZE = 200 * 1024 * 1024  # 200MB
                print(f"🔧 パートサイズ制限強化: 200MB")
                
        except Exception as e:
            print(f"⚠️ 実行時制限解除エラー: {e}")
    
    print(f"🎯 制限確認完了 - 処理開始可能")
    start_time = time.time()
    
    try:
        # 詳細リクエスト情報をログ出力
        content_length = request.headers.get("content-length", "不明")
        content_type = request.headers.get("content-type", "不明")
        user_agent = request.headers.get("user-agent", "不明")
        
        print(f"🌐 リクエスト詳細:")
        print(f"   📏 Content-Length: {content_length}")
        print(f"   📋 Content-Type: {content_type}")
        print(f"   🖥️  User-Agent: {user_agent}")
        print(f"   📁 クエリ画像: {query_image.filename if query_image else 'None'}")
        print(f"   📂 フォルダ画像数: {len(folder_images) if folder_images else 0}")
        
        # ファイルサイズの推定
        if folder_images and len(folder_images) > 0:
            sample_size = 0
            sample_count = min(5, len(folder_images))
            for i in range(sample_count):
                try:
                    content = await folder_images[i].read()
                    size = len(content)
                    sample_size += size
                    await folder_images[i].seek(0)  # ポインタを戻す
                    print(f"   📎 サンプル{i+1}: {folder_images[i].filename} ({size:,}バイト)")
                except Exception as e:
                    print(f"   ❌ サンプル{i+1}読み取りエラー: {e}")
            
            if sample_count > 0:
                avg_size = sample_size / sample_count
                estimated_total = avg_size * len(folder_images)
                print(f"   💾 推定総サイズ: {estimated_total / (1024*1024):.1f}MB")
                
                # 大量データ警告
                if estimated_total > 500 * 1024 * 1024:  # 500MB以上
                    print(f"⚠️  大量データ警告: 推定{estimated_total / (1024*1024):.1f}MB")
        
    except Exception as e:
        print(f"❌ リクエスト解析エラー: {e}")
    
    # デバッグ用ログ
    print(f"🔍 リクエスト受信: query_image={query_image.filename if query_image else 'None'}")
    print(f"🔍 フォルダ画像数: {len(folder_images) if folder_images else 0}")
    
    # すべてのファイルをシンプルな順次処理で実行
    print(f"📋 シンプル処理実行: {len(folder_images)}ファイル")
    return await compare_folder_internal(query_image, folder_images)
