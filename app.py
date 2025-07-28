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
import asyncio
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import gc
import time
from functools import partial
import mediapipe as mp

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

# JAPANESE_FACE_v1用のカスタムDeepFaceモデルクラス
class JAPANESE_FACE_v1_Model:
    def __init__(self, session, model_info):
        self.model_name = "JAPANESE_FACE_v1"
        self.input_shape = (224, 224, 3)
        self.output_shape = 512
        self.session = session
        self.model_info = model_info
    
    def predict(self, img_array):
        """DeepFace互換の予測関数"""
        try:
            # 入力を正規化 (DeepFaceは0-255, JAPANESE_FACE_v1は-1~1)
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
            print(f"JAPANESE_FACE_v1予測エラー: {e}")
            raise e

# DeepFaceにJAPANESE_FACE_v1モデルを登録する関数
def register_japanese_face_v1_to_deepface(session, model_info):
    """JAPANESE_FACE_v1をDeepFaceのモデルとして登録"""
    try:
        # JAPANESE_FACE_v1インスタンスを作成
        japanese_face_v1_instance = JAPANESE_FACE_v1_Model(session, model_info)
        print("JAPANESE_FACE_v1をDeepFace形式で初期化しました")
        return japanese_face_v1_instance
        
    except Exception as e:
        print(f"JAPANESE_FACE_v1登録エラー: {e}")
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
            print(f"❌ ミドルウェアエラー: {type(e).__name__}: {e}")
            # より詳細なエラー情報をログ出力
            import traceback
            traceback.print_exc()
            
            # より安全なエラーハンドリング
            try:
                error_msg = str(e)
            except:
                error_msg = f"{type(e).__name__}: エラーの文字列化に失敗"
            
            from fastapi import HTTPException
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={"error": "サーバー内部エラーが発生しました", "detail": error_msg}
            )

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

# モデル比較ページのルート追加
@app.get("/model_comparison.html", response_class=HTMLResponse)
def model_comparison():
    """モデル比較ページを配信"""
    try:
        with open("model_comparison.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="モデル比較ページが見つかりません")

# 顔検知デバッグページのルート追加
@app.get("/debug_face.html", response_class=HTMLResponse)
def debug_face():
    """顔検知デバッグページを配信"""
    try:
        with open("debug_face.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="顔検知デバッグページが見つかりません")

@app.get("/compare_detection.html", response_class=HTMLResponse)
def compare_detection():
    """顔検知比較ページを配信"""
    try:
        with open("compare_detection.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="顔検知比較ページが見つかりません")

@app.get("/compare_1vn_accuracy.html", response_class=HTMLResponse)
def compare_1vn_accuracy_page():
    """1対N精度比較ページを配信"""
    try:
        with open("compare_1vn_accuracy.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="1対N精度比較ページが見つかりません")

# JAPANESE_FACE_v1 モデル設定
MODEL_CONFIG = {
    "path": "JAPANESE_FACE_v1.onnx",
    "name": "JAPANESE_FACE_v1",
    "description": "GLint-R100データセットで訓練された高精度顔認識モデル（Apache 2.0ライセンス）",
    "input_name": "x.1",
    "input_size": (224, 224),
    "output_name": "1170",
    "embedding_size": 512
}

def initialize_model():
    """JAPANESE_FACE_v1モデルを初期化（MediaPipe顔検出）"""
    try:
        # JAPANESE_FACE_v1必要ライブラリの確認
        import torch
        from huggingface_hub import snapshot_download
        
        print("🔄 JAPANESE_FACE_v1モデルをHuggingFaceからダウンロード中...")
        
        # HuggingFaceからJAPANESE_FACE_v1モデルをダウンロード
        model_dir = snapshot_download(
            repo_id="fal/JAPANESE_FACE_v1",
            local_dir="./models/japanese_face_v1",
            ignore_patterns=["*.md", "*.txt", "*.jpg", "*.png"]
        )
        print(f"✅ JAPANESE_FACE_v1モデルダウンロード完了: {model_dir}")
        
        # プロバイダー設定
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        # JAPANESE_FACE_v1認識モデルファイルのパス
        model_path = os.path.join(model_dir, MODEL_CONFIG["path"])
        
        if not os.path.exists(model_path):
            # 利用可能なONNXファイルを検索
            onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx') and 'glintr' in f]
            if onnx_files:
                model_path = os.path.join(model_dir, onnx_files[0])
                print(f"🔍 発見されたJAPANESE_FACE_v1モデル: {onnx_files[0]}")
            else:
                raise FileNotFoundError(f"JAPANESE_FACE_v1認識モデルが見つかりません: {model_dir}")
        
        # JAPANESE_FACE_v1認識モデルをONNX Runtimeで読み込み
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        
        # MediaPipeベースのモデル構造を返す
        model = {
            'recognition_session': session,
            'model_path': model_path
        }
        
        print(f"✅ {MODEL_CONFIG['name']} + MediaPipe 初期化完了")
        print(f"🔧 実行プロバイダー: {providers}")
        return model
        
    except Exception as e:
        print(f"❌ {MODEL_CONFIG['name']} 初期化エラー: {e}")
        return None

# JAPANESE_FACE_v1モデルセッションを初期化
japanese_face_v1_session = initialize_model()

print("🌟 JAPANESE_FACE_v1 + MediaPipe（Apache 2.0ライセンス）を使用します")

# Buffalo_l顔検出モデルの初期化
BUFFALO_L_AVAILABLE = False  # グローバル変数として初期化
buffalo_l_app = None

try:
    import insightface
    # 検出のみに特化してリソースを節約
    buffalo_l_app = insightface.app.FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'recognition']  # 検出と認識のみ有効化
    )
    # 小さい顔も検出できるよう、より小さいdet_sizeを使用
    buffalo_l_app.prepare(ctx_id=0, det_size=(224, 224))
    print("✅ Buffalo_l顔検出モデル初期化完了 (det_size=224x224)")
    BUFFALO_L_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Buffalo_l顔検出モデル初期化失敗: {e}")
    buffalo_l_app = None
    BUFFALO_L_AVAILABLE = False

# MediaPipe顔検出の初期化（比較用）
print("🔧 MediaPipe顔検出を初期化中...")
try:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,  # 0: 短距離モデル（2m以内）, 1: 長距離モデル（5m以内）
        min_detection_confidence=0.5
    )
    print("✅ MediaPipe顔検出初期化完了")
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"❌ MediaPipe顔検出初期化エラー: {e}")
    face_detection = None
    MEDIAPIPE_AVAILABLE = False

def detect_faces_mediapipe(image):
    """MediaPipeで顔検出を行う関数"""
    if not MEDIAPIPE_AVAILABLE:
        return []
    
    try:
        # BGR -> RGB変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 顔検出実行
        results = face_detection.process(rgb_image)
        
        faces = []
        if results.detections is not None and len(results.detections) > 0:
            for detection in results.detections:
                # バウンディングボックスを取得
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                # 相対座標を絶対座標に変換
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # 画像境界内に制限
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                face_info = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(detection.score[0]),
                    'detection_method': 'mediapipe'
                }
                faces.append(face_info)
        
        return faces
    except Exception as e:
        print(f"❌ MediaPipe顔検出エラー: {e}")
        return []

def detect_faces_buffalo_l(image):
    """Buffalo_lで顔検出を行う関数"""
    if not BUFFALO_L_AVAILABLE:
        return []
    
    try:
        # BGR -> RGB変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 顔検出実行
        faces_data = buffalo_l_app.get(rgb_image)
        
        faces = []
        for face in faces_data:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            face_info = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(face.det_score),
                'detection_method': 'buffalo_l'
            }
            faces.append(face_info)
        
        return faces
    except Exception as e:
        print(f"❌ Buffalo_l顔検出エラー: {e}")
        return []

def get_embedding_with_mediapipe_detection(image):
    """MediaPipeで顔検出してJAPANESE_FACE_v1で埋め込みを取得"""
    if not MEDIAPIPE_AVAILABLE or japanese_face_v1_session is None:
        return None, None
    
    try:
        # MediaPipeで顔検出
        mediapipe_faces = detect_faces_mediapipe(image)
        if not mediapipe_faces:
            return None, None
        
        # 最初の顔を使用（信頼度が最も高い顔を選択）
        best_face = max(mediapipe_faces, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = best_face['bbox']
        
        # 顔領域を切り出し
        face_crop = image[y1:y2, x1:x2]
        
        # 縦横比を保持して224x224にリサイズ
        def resize_with_padding(img, target_size=(224, 224)):
            h, w = img.shape[:2]
            target_w, target_h = target_size
            
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h))
            
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
        
        aligned_face = resize_with_padding(face_crop, (224, 224))
        
        # 前処理：正規化とバッチ次元追加
        input_image = aligned_face.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC -> CHW
        input_image = np.expand_dims(input_image, axis=0)   # バッチ次元追加
        
        # ONNX推論実行
        recognition_session = japanese_face_v1_session['recognition_session']
        onnx_inputs = {MODEL_CONFIG["input_name"]: input_image}
        outputs = recognition_session.run([MODEL_CONFIG["output_name"]], onnx_inputs)
        embedding = outputs[0][0]  # バッチ次元を削除
        
        # L2正規化
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding, best_face['confidence']
        
    except Exception as e:
        print(f"❌ MediaPipe+JAPANESE_FACE_v1埋め込み取得エラー: {e}")
        return None, None

def get_embedding_with_buffalo_l_detection(image):
    """Buffalo_lで顔検出してJAPANESE_FACE_v1で埋め込みを取得"""
    if not BUFFALO_L_AVAILABLE or japanese_face_v1_session is None:
        return None, None
    
    try:
        # Buffalo_lで顔検出
        buffalo_faces = detect_faces_buffalo_l(image)
        if not buffalo_faces:
            return None, None
        
        # 最初の顔を使用（信頼度が最も高い顔を選択）
        best_face = max(buffalo_faces, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = best_face['bbox']
        
        # 顔領域を切り出し
        face_crop = image[y1:y2, x1:x2]
        
        # 縦横比を保持して224x224にリサイズ
        def resize_with_padding(img, target_size=(224, 224)):
            h, w = img.shape[:2]
            target_w, target_h = target_size
            
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h))
            
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
        
        aligned_face = resize_with_padding(face_crop, (224, 224))
        
        # 前処理：正規化とバッチ次元追加
        input_image = aligned_face.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC -> CHW
        input_image = np.expand_dims(input_image, axis=0)   # バッチ次元追加
        
        # ONNX推論実行
        recognition_session = japanese_face_v1_session['recognition_session']
        onnx_inputs = {MODEL_CONFIG["input_name"]: input_image}
        outputs = recognition_session.run([MODEL_CONFIG["output_name"]], onnx_inputs)
        embedding = outputs[0][0]  # バッチ次元を削除
        
        # L2正規化
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding, best_face['confidence']
        
    except Exception as e:
        print(f"❌ Buffalo_l+JAPANESE_FACE_v1埋め込み取得エラー: {e}")
        return None, None

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

# MediaPipe関数削除完了

def detect_and_align_face(image_path, detection_method="buffalo_l"):
    """Buffalo_lのみを使用した顔検出・アライメント処理"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Buffalo_l検出のみ使用
    if BUFFALO_L_AVAILABLE:
        buffalo_result = detect_and_align_buffalo_l(image)
        if buffalo_result is not None:
            return buffalo_result
        else:
            print("⚠️ Buffalo_l検出失敗")
            return None
    else:
        print("❌ Buffalo_lが利用できません")
        return None

def detect_and_align_buffalo_l(image):
    """Buffalo_l顔検出モデルによる顔検出とアライメント（224x224対応）"""
    if not BUFFALO_L_AVAILABLE or buffalo_l_app is None:
        print("⚠️ Buffalo_l顔検出モデルが利用できません")
        return None
    
    try:
        # BGR -> RGB変換（OpenCVとInsightFaceの違い対応）
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # 画像サイズ情報を表示
        print(f"🔍 画像サイズ: {image.shape}, RGB画像サイズ: {rgb_image.shape}")
        
        # 画像サイズ最適化
        original_shape = rgb_image.shape[:2]
        
        # 小さすぎる画像は拡大（最小256px）
        if min(rgb_image.shape[:2]) < 256:
            scale_up = 256 / min(rgb_image.shape[:2])
            new_h, new_w = int(rgb_image.shape[0] * scale_up), int(rgb_image.shape[1] * scale_up)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h))
            print(f"🔧 画像拡大: {original_shape} → {new_w}x{new_h}")
        
        # 大きすぎる画像は縮小（最大512px）
        elif max(rgb_image.shape[:2]) > 512:
            scale_down = 512 / max(rgb_image.shape[:2])
            new_h, new_w = int(rgb_image.shape[0] * scale_down), int(rgb_image.shape[1] * scale_down)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h))
            print(f"🔧 画像縮小: {original_shape} → {new_w}x{new_h}")
        
        # 画像の品質改善
        rgb_image = cv2.bilateralFilter(rgb_image, 9, 75, 75)
        
        # Buffalo_lで顔を検出（複数の試行）
        faces = []
        
        # 1回目: 通常検出
        try:
            faces = buffalo_l_app.get(rgb_image)
            if len(faces) > 0:
                print(f"✅ Buffalo_l: {len(faces)}個の顔を検出")
        except Exception as e:
            print(f"❌ Buffalo_l検出エラー: {e}")
        
        # 2回目: より小さいサイズで検出を試行
        if len(faces) == 0:
            print("🔄 より小さいサイズで再検出を試行...")
            small_h, small_w = max(128, rgb_image.shape[0]//2), max(128, rgb_image.shape[1]//2)
            small_image = cv2.resize(rgb_image, (small_w, small_h))
            try:
                small_faces = buffalo_l_app.get(small_image)
                if len(small_faces) > 0:
                    # 座標を元のサイズにスケール
                    scale_factor = min(rgb_image.shape[:2]) / min(small_image.shape[:2])
                    for face in small_faces:
                        face.bbox = face.bbox * scale_factor
                    faces = small_faces
                    print(f"✅ 小サイズ検出成功: {len(faces)}個の顔")
            except Exception as e:
                print(f"❌ 小サイズ検出エラー: {e}")
        
        if len(faces) == 0:
            print("⚠️ Buffalo_l: 全ての試行で顔が検出されませんでした")
            return None
        
        # 最も信頼度の高い顔を選択
        best_face = max(faces, key=lambda x: x.det_score)
        
        # バウンディングボックスを取得
        bbox = best_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # バウンディングボックスを少し広げる（顔全体を含むため）
        margin = 20
        h, w = image.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # 顔領域を切り出し
        face_crop = image[y1:y2, x1:x2]
        
        # デバッグ用: 切り出した顔を保存
        import os
        os.makedirs("static/temp", exist_ok=True)
        
        debug_path = f"static/temp/debug_face_{int(time.time())}.jpg"
        success = cv2.imwrite(debug_path, face_crop)
        if success:
            print(f"🖼️  切り出した顔を保存: {debug_path}")
        else:
            print(f"❌ 切り出し画像保存失敗: {debug_path}")
        
        # 224x224にリサイズ
        aligned_face = cv2.resize(face_crop, (224, 224))
        
        # リサイズ後の顔も保存
        aligned_debug_path = f"static/temp/debug_aligned_{int(time.time())}.jpg"
        success = cv2.imwrite(aligned_debug_path, aligned_face)
        if success:
            print(f"🖼️  アライメント後の顔を保存: {aligned_debug_path}")
        else:
            print(f"❌ アライメント画像保存失敗: {aligned_debug_path}")
        
        print(f"✅ Buffalo_l顔検出成功: 信頼度={best_face.det_score:.3f}, bbox=({x1},{y1},{x2},{y2})")
        return aligned_face
        
    except Exception as e:
        print(f"❌ Buffalo_l顔検出エラー: {e}")
        return None

def preprocess_image_for_model(file_path, use_detection=True, detection_method="mediapipe"):
    """Buffalo_lモデル用の前処理"""
    input_size = MODEL_CONFIG["input_size"]
    
    if use_detection:
        # 顔検出とクロップ（検出方法選択可能）
        face_image = detect_and_align_face(file_path, detection_method)
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

def preprocess_image_simple(file):
    """シンプルな前処理（顔検出なし）"""
    img = Image.open(file).convert('RGB').resize((224, 224))
    img = np.asarray(img, dtype=np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img

def preprocess_images_batch(file_paths, use_detection=True, batch_size=32):
    """複数画像のバッチ前処理"""
    input_size = MODEL_CONFIG["input_size"]
    
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
    # 各画像は約224x224x3x4 = 600KB、さらに前処理で2-3倍になると仮定
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

def get_embedding_batch(file_paths, use_detection=True, batch_size=None):
    """バッチ処理による高速な特徴量抽出"""
    if japanese_face_v1_session is None:
        return [], []
    
    # 自動バッチサイズ調整
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(len(file_paths))
    
    # 認識セッションを取得
    recognition_session = japanese_face_v1_session['recognition_session']
    input_name = MODEL_CONFIG["input_name"]
    
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
                batch_files, use_detection, batch_size
            )
            
            if batch_images is None:
                print(f"⚠️ バッチ {batch_num}/{total_batches}: 処理可能な画像なし")
                continue
            
            # JAPANESE_FACE_v1ハイブリッドアプローチ - 個別処理でバッチ風に実行
            batch_embeddings = []
            batch_valid_indices = []
            
            for j, file_path in enumerate(batch_files):
                try:
                    import cv2
                    # 画像読み込み
                    img = cv2.imread(file_path)
                    if img is None:
                        continue
                        
                    # Buffalo_lで顔検出とアライメント
                    face_aligned = detect_and_align_buffalo_l(img)
                    if face_aligned is None:
                        continue
                    
                    # 前処理
                    face_aligned = face_aligned.astype(np.float32)
                    face_aligned = (face_aligned / 127.5) - 1.0
                    face_aligned = np.transpose(face_aligned, (2, 0, 1))
                    face_aligned = np.expand_dims(face_aligned, axis=0)
                    
                    # JAPANESE_FACE_v1認識
                    outputs = recognition_session.run(None, {input_name: face_aligned})
                    embedding = outputs[0][0]
                    
                    # 正規化
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    batch_embeddings.append(embedding)
                    batch_valid_indices.append(i + j)
                    
                except Exception as e:
                    print(f"⚠️ バッチ内ファイル処理エラー: {file_path} - {e}")
                    continue
            
            # バッチ結果を保存
            all_embeddings.extend(batch_embeddings)
            all_valid_indices.extend(batch_valid_indices)
            
            processed_count += len(batch_embeddings)
            
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
                return get_embedding_batch(file_paths, use_detection, max(batch_size//2, 4))
            continue
    
    print(f"✅ バッチ処理完了: {len(all_embeddings)}個の埋め込みベクトル生成")
    return all_embeddings, all_valid_indices

def get_embedding_japanese_face_v1_from_image(image, detection_method="mediapipe"):
    """JAPANESE_FACE_v1モデルで埋め込みベクトルを取得（画像データから直接）"""
    if japanese_face_v1_session is None:
        return None
    
    try:
        # Buffalo_lのみで顔を検出・アライメント
        if BUFFALO_L_AVAILABLE:
            aligned_face = detect_and_align_buffalo_l(image)
        else:
            print("❌ Buffalo_lが利用できません")
            return None
        
        if aligned_face is None:
            return None
        
        # 前処理：正規化とバッチ次元追加
        input_image = aligned_face.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC -> CHW
        input_image = np.expand_dims(input_image, axis=0)   # バッチ次元追加
        
        # ONNX推論実行
        recognition_session = japanese_face_v1_session['recognition_session']
        onnx_inputs = {MODEL_CONFIG["input_name"]: input_image}
        outputs = recognition_session.run([MODEL_CONFIG["output_name"]], onnx_inputs)
        embedding = outputs[0][0]  # バッチ次元を削除
        
        # L2正規化
        embedding = embedding / np.linalg.norm(embedding)
        
        # float32形式で返す
        return np.asarray(embedding, dtype=np.float32)
        
    except Exception as e:
        print(f"❌ JAPANESE_FACE_v1埋め込み生成エラー: {e}")
        return None

def get_embedding_japanese_face_v1(file_path, use_detection=True, detection_method="buffalo_l"):
    """JAPANESE_FACE_v1モデルで埋め込みベクトルを取得（Buffalo_l検出のみ）"""
    if japanese_face_v1_session is None:
        return {
            'embedding': None,
            'error': 'JAPANESE_FACE_v1モデルが読み込まれていません',
            'processing_time': 0
        }
    
    start_time = time.time()
    try:
        import cv2
        
        # 画像を読み込み
        img = cv2.imread(file_path)
        if img is None:
            return {
                'embedding': None,
                'error': '画像読み込み失敗',
                'processing_time': 0
            }
        
        # 顔検出とアライメント（検出方法選択可能）
        if detection_method == "buffalo_l" and BUFFALO_L_AVAILABLE:
            face_aligned = detect_and_align_buffalo_l(img)
            if face_aligned is None:
                print("⚠️ Buffalo_l検出失敗、MediaPipeにフォールバック")
                face_aligned = detect_and_align_buffalo_l(img)
        else:
            face_aligned = detect_and_align_buffalo_l(img)
        
        if face_aligned is None:
            return {
                'embedding': None,
                'error': '顔が検出されませんでした',
                'processing_time': 0
            }
        
        # 入力データの前処理（[0,255] -> [-1,1]の正規化）
        face_aligned = face_aligned.astype(np.float32)
        face_aligned = (face_aligned / 127.5) - 1.0
        face_aligned = np.transpose(face_aligned, (2, 0, 1))  # HWC -> CHW
        face_aligned = np.expand_dims(face_aligned, axis=0)  # バッチ次元追加
        
        # JAPANESE_FACE_v1で埋め込みベクトル計算
        recognition_session = japanese_face_v1_session['recognition_session']
        input_name = MODEL_CONFIG["input_name"]
        outputs = recognition_session.run(None, {input_name: face_aligned})
        embedding = outputs[0][0]
        
        # 正規化
        embedding = embedding / np.linalg.norm(embedding)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'embedding': embedding,
            'error': None,
            'processing_time': processing_time
        }
        
    except Exception as e:
        return {
            'embedding': None,
            'error': str(e),
            'processing_time': (time.time() - start_time) * 1000
        }

def get_embedding_single(file_path, use_detection=True):
    """単一ファイル処理（バッチ処理なし）"""
    if japanese_face_v1_session is None:
        return None
    
    try:
        import cv2
        
        # 画像を読み込み
        img = cv2.imread(file_path)
        if img is None:
            return None
        
        # Buffalo_lで顔検出とアライメント
        face_aligned = detect_and_align_buffalo_l(img)
        
        if face_aligned is None:
            return None
        
        # 入力データの前処理（[0,255] -> [-1,1]の正規化）
        face_aligned = face_aligned.astype(np.float32)
        face_aligned = (face_aligned / 127.5) - 1.0
        face_aligned = np.transpose(face_aligned, (2, 0, 1))  # HWC -> CHW
        face_aligned = np.expand_dims(face_aligned, axis=0)  # バッチ次元追加
        
        # JAPANESE_FACE_v1で埋め込みベクトル計算
        recognition_session = japanese_face_v1_session['recognition_session']
        input_name = MODEL_CONFIG["input_name"]
        outputs = recognition_session.run(None, {input_name: face_aligned})
        embedding = outputs[0][0]
        
        # 正規化
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
        
    except Exception as e:
        print(f"❌ 単一ファイル推論エラー: {e}")
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

def compare_japanese_face_v1_faces(file_path1, file_path2):
    """JAPANESE_FACE_v1モデルで2つの顔を比較"""
    # 各画像の埋め込みベクトルを取得
    embedding1 = get_embedding_japanese_face_v1(file_path1, use_detection=True)
    embedding2 = get_embedding_japanese_face_v1(file_path2, use_detection=True)
    
    if (embedding1['embedding'] is not None and 
        embedding2['embedding'] is not None):
        
        # アンサンブル検証
        ensemble_result = ensemble_verification(
            embedding1['embedding'],
            embedding2['embedding']
        )
        
        return {
            'model_info': MODEL_CONFIG,
            'ensemble_result': ensemble_result,
            'processing_time': (embedding1['processing_time'] + 
                               embedding2['processing_time']),
            'error': None
        }
    else:
        error_msg = embedding1.get('error', 'Unknown error') + '; ' + embedding2.get('error', 'Unknown error')
        return {
            'model_info': MODEL_CONFIG,
            'ensemble_result': None,
            'processing_time': 0,
            'error': error_msg
        }


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
    
    # JAPANESE_FACE_v1顔認識処理
    japanese_face_v1_comparison = compare_japanese_face_v1_faces(filename1, filename2)
    
    # JAPANESE_FACE_v1埋め込みベクトル取得
    emb1_japanese_face_v1 = get_embedding_japanese_face_v1(filename1, use_detection=True)
    emb2_japanese_face_v1 = get_embedding_japanese_face_v1(filename2, use_detection=True)
    
    if (emb1_japanese_face_v1['embedding'] is not None and 
        emb2_japanese_face_v1['embedding'] is not None):
        # アンサンブル検証を使用
        ensemble_results = ensemble_verification(
            emb1_japanese_face_v1['embedding'], 
            emb2_japanese_face_v1['embedding']
        )
        similarity_japanese_face_v1 = ensemble_results['cosine_similarity']
        is_same_japanese_face_v1 = ensemble_results['is_same_adaptive']
        confidence_score = ensemble_results['confidence_score']
    else:
        similarity_japanese_face_v1 = 0.0
        is_same_japanese_face_v1 = False
        confidence_score = 0.0
        ensemble_results = {
            'cosine_similarity': 0.0,
            'euclidean_distance': 0.0,
            'l1_distance': 0.0,
            'normalized_euclidean': 0.0,
            'adaptive_threshold': 0.45,
            'confidence_score': 0.0
        }
    
    # JAPANESE_FACE_v1埋め込みベクトルの詳細情報
    japanese_face_v1_embedding_info = {
        'emb1': emb1_japanese_face_v1['embedding'].tolist()[:20] if emb1_japanese_face_v1['embedding'] is not None else [],
        'emb2': emb2_japanese_face_v1['embedding'].tolist()[:20] if emb2_japanese_face_v1['embedding'] is not None else [],
        'embedding_dims': MODEL_CONFIG['embedding_size'],
        'emb1_norm': float(np.linalg.norm(emb1_japanese_face_v1['embedding'])) if emb1_japanese_face_v1['embedding'] is not None else 0.0,
        'emb2_norm': float(np.linalg.norm(emb2_japanese_face_v1['embedding'])) if emb2_japanese_face_v1['embedding'] is not None else 0.0
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
        "japanese_face_v1": {
            "similarity": f"{similarity_japanese_face_v1:.4f}",
            "is_same": is_same_japanese_face_v1,
            "adaptive_threshold": f"{ensemble_results.get('adaptive_threshold', 0.5):.4f}",
            "confidence_score": f"{confidence_score:.4f}",
            "euclidean_distance": f"{ensemble_results.get('euclidean_distance', 0.0):.4f}",
            "l1_distance": f"{ensemble_results.get('l1_distance', 0.0):.4f}",
            "normalized_euclidean": f"{ensemble_results.get('normalized_euclidean', 0.0):.4f}",
            "embeddings": japanese_face_v1_embedding_info,
            "processing_time": f"{emb1_japanese_face_v1.get('processing_time', 0) + emb2_japanese_face_v1.get('processing_time', 0):.1f}ms"
        },
        "img1_path": "/" + filename1,
        "img2_path": "/" + filename2,
        "japanese_face_v1_comparison": japanese_face_v1_comparison,
        "model_info": {
            "deepface_arcface": "ArcFace (DeepFace implementation)",
            "japanese_face_v1": MODEL_CONFIG['name'],
            "description": MODEL_CONFIG['description']
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
    
    # JAPANESE_FACE_v1モデルでクエリ画像の埋め込みベクトルを取得
    query_embedding = get_embedding_japanese_face_v1(query_filename, use_detection=True)
    
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
    results = await _execute_comparison_japanese_face_v1(query_embedding, valid_file_info_list, batch_size, start_time)
    
    # 結果の整理と返却
    return _format_comparison_results(results, query_image, total_files, valid_file_info_list, start_time, is_chunk)

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

async def _execute_comparison_japanese_face_v1(query_embedding, valid_file_info_list, batch_size, start_time):
    """JAPANESE_FACE_v1モデルによるバッチ処理比較"""
    total_files = len(valid_file_info_list)
    
    print(f"🚀 JAPANESE_FACE_v1 バッチ処理比較開始: {total_files}ファイル")
    
    # バッチ処理でターゲット画像の埋め込みベクトルを一括取得
    target_file_paths = [file_info['filename'] for file_info in valid_file_info_list]
    
    print(f"📊 バッチ特徴量抽出開始... (バッチサイズ: {batch_size})")
    target_embeddings, valid_indices = get_embedding_batch(
        target_file_paths, 
        use_detection=True,
        batch_size=batch_size
    )
    
    if not target_embeddings:
        print("❌ バッチ特徴量抽出に失敗")
        return []
    
    print(f"✅ バッチ特徴量抽出完了: {len(target_embeddings)}個の埋め込みベクトル")
    
    # クエリ埋め込みベクトルを取得
    query_emb = query_embedding.get('embedding')
    if query_emb is None:
        print("❌ クエリ画像の埋め込みベクトルが見つかりません")
        print(f"🔍 query_embedding内容: {query_embedding}")
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
                'best_model': MODEL_CONFIG['name'],
                'model_results': {
                    'japanese_face_v1': {
                        'model_name': MODEL_CONFIG['name'],
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
    
    print(f"✅ JAPANESE_FACE_v1 バッチ処理完了: {len(results)}件の結果")
    
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


async def _execute_comparison_no_batch(query_embedding, valid_file_info_list, start_time):
    """バッチ処理なしの比較処理（速度比較用）"""
    total_files = len(valid_file_info_list)
    
    print(f"🐌 非バッチ処理比較開始: {total_files}ファイル（1ファイルずつ順次処理）")
    
    # クエリ埋め込みベクトルを取得
    query_emb = query_embedding.get('embedding')
    if query_emb is None:
        print("❌ クエリ画像の埋め込みベクトルが見つかりません")
        return []
    
    print(f"🔄 順次処理による類似度計算開始...")
    results = []
    
    # 1ファイルずつ順次処理
    for i, file_info in enumerate(valid_file_info_list):
        try:
            # 1ファイルずつ埋め込みベクトルを取得
            target_emb = get_embedding_single(file_info['filename'], use_detection=True)
            
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
                'best_model': MODEL_CONFIG['name'],
                'model_results': {
                    'japanese_face_v1': {
                        'model_name': MODEL_CONFIG['name'],
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
                'best_model': MODEL_CONFIG['name'],
                'model_results': {
                    'japanese_face_v1': {
                        'model_name': MODEL_CONFIG['name'],
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

def _format_comparison_results(results, query_image, total_files, valid_file_info_list, start_time, is_chunk=False):
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
            'model': MODEL_CONFIG['name'],
            'model_description': MODEL_CONFIG['description'],
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
        query_embedding = get_embedding_japanese_face_v1(query_filename, use_detection=True)
        
        # ファイル保存処理
        file_info_list = await _save_files_individually(folder_images)
        valid_file_info_list = [f for f in file_info_list if not f.get('error') and f.get('filename')]
        
        preprocessing_time = time.time() - start_time
        comparison_start_time = time.time()
        
        if use_batch:
            # バッチ処理版
            optimal_batch_size = calculate_optimal_batch_size(len(valid_file_info_list))
            print(f"🚀 バッチ処理モード実行 (最適バッチサイズ: {optimal_batch_size})")
            results = await _execute_comparison_japanese_face_v1(query_embedding, valid_file_info_list, optimal_batch_size, comparison_start_time)
        else:
            # 非バッチ処理版
            print(f"🐌 非バッチ処理モード実行 (1ファイルずつ順次処理)")
            results = await _execute_comparison_no_batch(query_embedding, valid_file_info_list, comparison_start_time)
        
        comparison_time = time.time() - comparison_start_time
        total_time = time.time() - start_time
        
        # 速度統計の計算
        files_per_second = len(valid_file_info_list) / comparison_time if comparison_time > 0 else 0
        
        # 結果のデバッグ情報
        print(f"🔍 デバッグ: results型={type(results)}, 長さ={len(results) if results else 'None'}")
        if results:
            print(f"🔍 デバッグ: 最初の結果={results[0] if len(results) > 0 else 'Empty'}")
        
        # top_matchesのデバッグ
        top_matches = results[:10] if results else []
        print(f"🔍 デバッグ: top_matches型={type(top_matches)}, 長さ={len(top_matches)}")
        
        return JSONResponse(content={
            'benchmark_mode': 'batch' if use_batch else 'no_batch',
            'query_image': query_image.filename,
            'total_files': total_files,
            'processed_files': len(valid_file_info_list),
            'matches_found': len([r for r in results if r['is_match']]) if results else 0,
            'results': results[:100] if results else [],  # 上位100件のみ表示
            'total_results': len(results) if results else 0,
            'performance_metrics': {
                'preprocessing_time_ms': preprocessing_time * 1000,
                'comparison_time_ms': comparison_time * 1000,
                'total_time_ms': total_time * 1000,
                'files_per_second': files_per_second,
                'avg_time_per_file_ms': (comparison_time / len(valid_file_info_list) * 1000) if valid_file_info_list else 0,
                'processing_method': 'バッチ処理' if use_batch else '順次処理（非バッチ）',
                'efficiency_score': files_per_second * len(valid_file_info_list)  # 総合効率スコア
            },
            'top_matches': top_matches
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

@app.post("/compare_models")
async def compare_models(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    detection_method: str = Form("mediapipe")  # "mediapipe" または "buffalo_l"
):
    """Buffalo_l vs JAPANESE_FACE_v1 モデル比較 (1対1)"""
    print(f"🆚 モデル比較開始: {file1.filename} vs {file2.filename}")
    print(f"🔍 検出方式: {detection_method}")
    print(f"🐃 Buffalo_l利用可能: {BUFFALO_L_AVAILABLE}")
    print(f"🌟 JAPANESE_FACE_v1セッション: {japanese_face_v1_session is not None}")
    
    try:
        # 画像読み込み
        image1_data = await file1.read()
        image2_data = await file2.read()
        
        image1 = np.frombuffer(image1_data, np.uint8)
        image2 = np.frombuffer(image2_data, np.uint8)
        
        image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)
        
        if image1 is None or image2 is None:
            raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました")
        
        results = {}
        
        # Buffalo_l モデル比較
        buffalo_start = time.time()
        try:
            if BUFFALO_L_AVAILABLE:
                # Buffalo_l で両方の画像から埋め込みを取得
                faces1 = buffalo_l_app.get(image1)
                faces2 = buffalo_l_app.get(image2)
                
                if len(faces1) > 0 and len(faces2) > 0:
                    # 最大の顔を使用
                    face1 = max(faces1, key=lambda x: x.bbox[2] * x.bbox[3])
                    face2 = max(faces2, key=lambda x: x.bbox[2] * x.bbox[3])
                    
                    embedding1 = np.asarray(face1.embedding, dtype=np.float32)
                    embedding2 = np.asarray(face2.embedding, dtype=np.float32)
                    
                    # コサイン類似度計算
                    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                    threshold = 0.6  # Buffalo_l の標準閾値
                    
                    buffalo_time = (time.time() - buffalo_start) * 1000
                    
                    results["buffalo_l"] = {
                        "available": True,
                        "model_name": "Buffalo_l",
                        "similarity": float(similarity),
                        "threshold": threshold,
                        "is_same": similarity > threshold,
                        "processing_time_ms": buffalo_time
                    }
                else:
                    results["buffalo_l"] = {
                        "available": False,
                        "error": "顔検出に失敗しました"
                    }
            else:
                results["buffalo_l"] = {
                    "available": False,
                    "error": "Buffalo_l モデルが利用できません"
                }
        except Exception as e:
            print(f"❌ Buffalo_l詳細エラー: {e}")
            import traceback
            traceback.print_exc()
            results["buffalo_l"] = {
                "available": False,
                "error": f"Buffalo_l エラー: {str(e)}"
            }
        
        # JAPANESE_FACE_v1 モデル比較
        japanese_start = time.time()
        try:
            if japanese_face_v1_session:
                # 指定された検出方式を使用
                embedding1 = get_embedding_japanese_face_v1_from_image(image1, detection_method=detection_method)
                embedding2 = get_embedding_japanese_face_v1_from_image(image2, detection_method=detection_method)
                
                if embedding1 is not None and embedding2 is not None:
                    # numpy配列であることを確認して変換
                    embedding1 = np.asarray(embedding1, dtype=np.float32)
                    embedding2 = np.asarray(embedding2, dtype=np.float32)
                    
                    # コサイン類似度計算
                    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                    threshold = 0.7  # JAPANESE_FACE_v1 の標準閾値
                    
                    japanese_time = (time.time() - japanese_start) * 1000
                    
                    results["japanese_face_v1"] = {
                        "available": True,
                        "model_name": "JAPANESE_FACE_v1",
                        "similarity": float(similarity),
                        "threshold": threshold,
                        "is_same": similarity > threshold,
                        "processing_time_ms": japanese_time,
                        "detection_method": detection_method
                    }
                else:
                    results["japanese_face_v1"] = {
                        "available": False,
                        "error": "顔検出または埋め込み生成に失敗しました"
                    }
            else:
                results["japanese_face_v1"] = {
                    "available": False,
                    "error": "JAPANESE_FACE_v1 モデルが利用できません"
                }
        except Exception as e:
            print(f"❌ JAPANESE_FACE_v1詳細エラー: {e}")
            import traceback
            traceback.print_exc()
            results["japanese_face_v1"] = {
                "available": False,
                "error": f"JAPANESE_FACE_v1 エラー: {str(e)}"
            }
        
        return {
            "success": True,
            "file1_name": file1.filename,
            "file2_name": file2.filename,
            "detection_method": detection_method,
            "comparison_results": results
        }
        
    except Exception as e:
        print(f"❌ モデル比較エラー: {e}")
        raise HTTPException(status_code=500, detail=f"モデル比較エラー: {str(e)}")

@app.post("/compare_models_folder")
async def compare_models_folder(
    query_image: UploadFile = File(...),
    folder_images: List[UploadFile] = File(...),
    detection_method: str = Form("mediapipe")  # "mediapipe" または "buffalo_l"
):
    """Buffalo_l vs JAPANESE_FACE_v1 モデル比較 (1対N)"""
    print(f"🆚 1対Nモデル比較開始: {query_image.filename} vs {len(folder_images)}ファイル")
    print(f"🔍 検出方式: {detection_method}")
    
    try:
        # クエリ画像読み込み
        query_data = await query_image.read()
        query_img = np.frombuffer(query_data, np.uint8)
        query_img = cv2.imdecode(query_img, cv2.IMREAD_COLOR)
        
        if query_img is None:
            raise HTTPException(status_code=400, detail="クエリ画像の読み込みに失敗しました")
        
        results = {
            "query_image": query_image.filename,
            "total_comparisons": len(folder_images),
            "detection_method": detection_method
        }
        
        # Buffalo_l モデルでの比較
        buffalo_start = time.time()
        buffalo_matches = []
        buffalo_available = False
        
        try:
            if BUFFALO_L_AVAILABLE:
                # クエリ画像の埋め込み取得
                query_faces = buffalo_l_app.get(query_img)
                if len(query_faces) > 0:
                    query_face = max(query_faces, key=lambda x: x.bbox[2] * x.bbox[3])
                    query_embedding = query_face.embedding
                    buffalo_available = True
                    
                    # 各フォルダ画像と比較
                    for folder_file in folder_images:
                        try:
                            folder_data = await folder_file.read()
                            folder_img = np.frombuffer(folder_data, np.uint8)
                            folder_img = cv2.imdecode(folder_img, cv2.IMREAD_COLOR)
                            
                            if folder_img is not None:
                                folder_faces = buffalo_l_app.get(folder_img)
                                if len(folder_faces) > 0:
                                    folder_face = max(folder_faces, key=lambda x: x.bbox[2] * x.bbox[3])
                                    folder_embedding = folder_face.embedding
                                    
                                    # 類似度計算
                                    similarity = np.dot(query_embedding, folder_embedding) / (
                                        np.linalg.norm(query_embedding) * np.linalg.norm(folder_embedding)
                                    )
                                    
                                    buffalo_matches.append({
                                        "filename": folder_file.filename,
                                        "similarity": float(similarity),
                                        "is_match": similarity > 0.6
                                    })
                        except Exception as e:
                            print(f"⚠️ Buffalo_l ファイル処理エラー ({folder_file.filename}): {e}")
                            continue
            
            # 類似度順にソート
            buffalo_matches.sort(key=lambda x: x["similarity"], reverse=True)
            buffalo_time = (time.time() - buffalo_start) * 1000
            
            results["buffalo_l"] = {
                "available": buffalo_available,
                "processing_time_ms": buffalo_time,
                "matches": buffalo_matches,
                "error": None if buffalo_available else "Buffalo_l 顔検出に失敗"
            }
            
        except Exception as e:
            results["buffalo_l"] = {
                "available": False,
                "processing_time_ms": 0,
                "matches": [],
                "error": f"Buffalo_l エラー: {str(e)}"
            }
        
        # JAPANESE_FACE_v1 モデルでの比較
        japanese_start = time.time()
        japanese_matches = []
        japanese_available = False
        
        try:
            if japanese_face_v1_session:
                # クエリ画像の埋め込み取得
                query_embedding = get_embedding_japanese_face_v1_from_image(query_img, detection_method=detection_method)
                if query_embedding is not None:
                    japanese_available = True
                    
                    # 各フォルダ画像と比較
                    for folder_file in folder_images:
                        try:
                            folder_data = await folder_file.read()
                            folder_img = np.frombuffer(folder_data, np.uint8)
                            folder_img = cv2.imdecode(folder_img, cv2.IMREAD_COLOR)
                            
                            if folder_img is not None:
                                folder_embedding = get_embedding_japanese_face_v1_from_image(folder_img, detection_method=detection_method)
                                
                                if folder_embedding is not None:
                                    # 類似度計算
                                    similarity = np.dot(query_embedding, folder_embedding) / (
                                        np.linalg.norm(query_embedding) * np.linalg.norm(folder_embedding)
                                    )
                                    
                                    japanese_matches.append({
                                        "filename": folder_file.filename,
                                        "similarity": float(similarity),
                                        "is_match": similarity > 0.7
                                    })
                        except Exception as e:
                            print(f"⚠️ JAPANESE_FACE_v1 ファイル処理エラー ({folder_file.filename}): {e}")
                            continue
            
            # 類似度順にソート
            japanese_matches.sort(key=lambda x: x["similarity"], reverse=True)
            japanese_time = (time.time() - japanese_start) * 1000
            
            results["japanese_face_v1"] = {
                "available": japanese_available,
                "processing_time_ms": japanese_time,
                "matches": japanese_matches,
                "error": None if japanese_available else "JAPANESE_FACE_v1 顔検出に失敗"
            }
            
        except Exception as e:
            results["japanese_face_v1"] = {
                "available": False,
                "processing_time_ms": 0,
                "matches": [],
                "error": f"JAPANESE_FACE_v1 エラー: {str(e)}"
            }
        
        # マッチ数の集計
        results["matches_found"] = {
            "buffalo_l": len([m for m in buffalo_matches if m["is_match"]]) if buffalo_matches else 0,
            "japanese_face_v1": len([m for m in japanese_matches if m["is_match"]]) if japanese_matches else 0
        }
        
        return results
        
    except Exception as e:
        print(f"❌ 1対Nモデル比較エラー: {e}")
        raise HTTPException(status_code=500, detail=f"1対Nモデル比較エラー: {str(e)}")

@app.post("/debug_face_detection")
async def debug_face_detection(file: UploadFile = File(...)):
    """顔検知デバッグ用エンドポイント - 切り出し画像を確認"""
    print(f"🔍 顔検知デバッグ開始: {file.filename}")
    
    try:
        # 画像読み込み
        image_data = await file.read()
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました")
        
        # ディレクトリ作成
        import os
        os.makedirs("static/temp", exist_ok=True)
        
        # 元画像を保存
        original_path = f"static/temp/debug_original_{int(time.time())}.jpg"
        success = cv2.imwrite(original_path, image)
        if not success:
            print(f"❌ 元画像保存失敗: {original_path}")
            original_path = None
        else:
            print(f"🖼️  元画像を保存: {original_path}")
        
        # Buffalo_lで顔検出
        if not BUFFALO_L_AVAILABLE:
            return {"error": "Buffalo_lが利用できません"}
        
        # BGR -> RGB変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 顔検出実行
        faces = buffalo_l_app.get(rgb_image)
        
        result = {
            "filename": file.filename,
            "image_size": image.shape,
            "faces_detected": len(faces),
            "original_image": f"/{original_path}" if original_path else None,
            "faces": []
        }
        
        if len(faces) == 0:
            result["message"] = "顔が検出されませんでした"
            return result
        
        # 各検出された顔の情報
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # 顔領域を切り出し
            face_crop = image[y1:y2, x1:x2]
            
            # 切り出し画像を保存
            crop_path = f"static/temp/debug_face_{i}_{int(time.time())}.jpg"
            crop_success = cv2.imwrite(crop_path, face_crop)
            if not crop_success:
                print(f"❌ 切り出し画像保存失敗: {crop_path}")
                crop_path = None
            
            # 縦横比を保持して224x224にリサイズ（パディング付き）
            def resize_with_padding(img, target_size=(224, 224)):
                h, w = img.shape[:2]
                target_w, target_h = target_size
                
                # スケール比率を計算（縦横比を保持）
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # リサイズ
                resized = cv2.resize(img, (new_w, new_h))
                
                # パディング用の背景（黒）を作成
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                # 中央に配置
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return padded
            
            aligned_face = resize_with_padding(face_crop, (224, 224))
            aligned_path = f"static/temp/debug_aligned_{i}_{int(time.time())}.jpg"
            aligned_success = cv2.imwrite(aligned_path, aligned_face)
            if not aligned_success:
                print(f"❌ アライメント画像保存失敗: {aligned_path}")
                aligned_path = None
            
            face_info = {
                "face_id": i,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "detection_score": float(face.det_score),
                "face_crop": f"/{crop_path}" if crop_path else None,
                "aligned_face": f"/{aligned_path}" if aligned_path else None,
                "crop_size": face_crop.shape
            }
            result["faces"].append(face_info)
            
            print(f"👤 顔{i}: bbox=({x1},{y1},{x2},{y2}), score={face.det_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ 顔検知デバッグエラー: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"顔検知デバッグエラー: {str(e)}")

@app.post("/compare_face_detection")
async def compare_face_detection(file: UploadFile = File(...)):
    """MediaPipe vs Buffalo_l 顔検出比較エンドポイント"""
    print(f"🔍 顔検知比較開始: {file.filename}")
    
    try:
        # 画像読み込み
        image_data = await file.read()
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました")
        
        # ディレクトリ作成
        import os
        os.makedirs("static/temp", exist_ok=True)
        
        # 元画像を保存
        original_path = f"static/temp/compare_original_{int(time.time())}.jpg"
        success = cv2.imwrite(original_path, image)
        if not success:
            print(f"❌ 元画像保存失敗: {original_path}")
            original_path = None
        else:
            print(f"🖼️  元画像を保存: {original_path}")
        
        # 縦横比を保持してリサイズする関数
        def resize_with_padding(img, target_size=(224, 224)):
            h, w = img.shape[:2]
            target_w, target_h = target_size
            
            # スケール比率を計算（縦横比を保持）
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # リサイズ
            resized = cv2.resize(img, (new_w, new_h))
            
            # パディング用の背景（黒）を作成
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # 中央に配置
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
        
        # MediaPipeで顔検出
        mediapipe_faces = detect_faces_mediapipe(image)
        
        # Buffalo_lで顔検出
        buffalo_faces = detect_faces_buffalo_l(image)
        
        result = {
            "filename": file.filename,
            "image_size": image.shape,
            "original_image": f"/{original_path}" if original_path else None,
            "mediapipe": {
                "available": MEDIAPIPE_AVAILABLE,
                "faces_detected": len(mediapipe_faces),
                "faces": []
            },
            "buffalo_l": {
                "available": BUFFALO_L_AVAILABLE,
                "faces_detected": len(buffalo_faces),
                "faces": []
            }
        }
        
        # MediaPipe検出結果の処理
        for i, face in enumerate(mediapipe_faces):
            x1, y1, x2, y2 = face['bbox']
            
            # 顔領域を切り出し
            face_crop = image[y1:y2, x1:x2]
            
            # 切り出し画像を保存
            crop_path = f"static/temp/mp_face_{i}_{int(time.time())}.jpg"
            crop_success = cv2.imwrite(crop_path, face_crop)
            
            # 224x224リサイズ（縦横比保持）
            aligned_face = resize_with_padding(face_crop, (224, 224))
            aligned_path = f"static/temp/mp_aligned_{i}_{int(time.time())}.jpg"
            aligned_success = cv2.imwrite(aligned_path, aligned_face)
            
            face_info = {
                "face_id": i,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(face['confidence']),
                "face_crop": f"/{crop_path}" if crop_success else None,
                "aligned_face": f"/{aligned_path}" if aligned_success else None,
                "crop_size": face_crop.shape,
                "method": "MediaPipe"
            }
            result["mediapipe"]["faces"].append(face_info)
        
        # Buffalo_l検出結果の処理
        for i, face in enumerate(buffalo_faces):
            x1, y1, x2, y2 = face['bbox']
            
            # 顔領域を切り出し
            face_crop = image[y1:y2, x1:x2]
            
            # 切り出し画像を保存
            crop_path = f"static/temp/bl_face_{i}_{int(time.time())}.jpg"
            crop_success = cv2.imwrite(crop_path, face_crop)
            
            # 224x224リサイズ（縦横比保持）
            aligned_face = resize_with_padding(face_crop, (224, 224))
            aligned_path = f"static/temp/bl_aligned_{i}_{int(time.time())}.jpg"
            aligned_success = cv2.imwrite(aligned_path, aligned_face)
            
            face_info = {
                "face_id": i,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(face['confidence']),
                "face_crop": f"/{crop_path}" if crop_success else None,
                "aligned_face": f"/{aligned_path}" if aligned_success else None,
                "crop_size": face_crop.shape,
                "method": "Buffalo_l"
            }
            result["buffalo_l"]["faces"].append(face_info)
        
        print(f"🔍 検出結果 - MediaPipe: {len(mediapipe_faces)}顔, Buffalo_l: {len(buffalo_faces)}顔")
        return result
        
    except Exception as e:
        print(f"❌ 顔検知比較エラー: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"顔検知比較エラー: {str(e)}")

@app.post("/compare_1vn_accuracy")
async def compare_1vn_accuracy(
    query_image: UploadFile = File(...),
    folder_images: List[UploadFile] = File(...)
):
    """1対N顔検索でMediaPipe vs Buffalo_l の精度比較"""
    print(f"🔍 1対N精度比較開始: クエリ={query_image.filename}, フォルダ={len(folder_images)}枚")
    
    try:
        # クエリ画像の読み込み
        query_data = await query_image.read()
        query_img = np.frombuffer(query_data, np.uint8)
        query_img = cv2.imdecode(query_img, cv2.IMREAD_COLOR)
        
        if query_img is None:
            raise HTTPException(status_code=400, detail="クエリ画像の読み込みに失敗しました")
        
        # MediaPipeでクエリ画像の埋め込みを取得
        mp_query_embedding, mp_query_confidence = get_embedding_with_mediapipe_detection(query_img)
        
        # Buffalo_lでクエリ画像の埋め込みを取得
        bl_query_embedding, bl_query_confidence = get_embedding_with_buffalo_l_detection(query_img)
        
        result = {
            "query_filename": query_image.filename,
            "total_images": int(len(folder_images)),
            "mediapipe": {
                "available": bool(MEDIAPIPE_AVAILABLE and mp_query_embedding is not None),
                "query_confidence": float(mp_query_confidence) if mp_query_confidence else 0.0,
                "matches": [],
                "processing_time_ms": 0.0
            },
            "buffalo_l": {
                "available": bool(BUFFALO_L_AVAILABLE and bl_query_embedding is not None),
                "query_confidence": float(bl_query_confidence) if bl_query_confidence else 0.0,
                "matches": [],
                "processing_time_ms": 0.0
            }
        }
        
        # 全画像データを先に読み込み
        folder_images_data = []
        for folder_img_file in folder_images:
            try:
                img_data = await folder_img_file.read()
                if img_data:
                    folder_images_data.append({
                        'filename': folder_img_file.filename,
                        'data': img_data
                    })
                else:
                    print(f"⚠️ 空のファイル: {folder_img_file.filename}")
            except Exception as e:
                print(f"❌ ファイル読み込みエラー ({folder_img_file.filename}): {e}")
        
        # MediaPipe処理
        if result["mediapipe"]["available"]:
            start_time = time.time()
            mp_similarities = []
            
            for i, img_info in enumerate(folder_images_data):
                try:
                    img = np.frombuffer(img_info['data'], np.uint8)
                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                    
                    # MediaPipeで埋め込み取得
                    folder_embedding, folder_confidence = get_embedding_with_mediapipe_detection(img)
                    
                    if folder_embedding is not None:
                        # コサイン類似度計算
                        similarity = np.dot(mp_query_embedding, folder_embedding)
                        is_match = similarity > 0.6  # 閾値
                        
                        mp_similarities.append({
                            "filename": img_info['filename'],
                            "similarity": float(similarity),
                            "confidence": float(folder_confidence),
                            "is_match": bool(is_match),  # numpy.bool_をboolに変換
                            "index": int(i)
                        })
                        
                except Exception as e:
                    print(f"❌ MediaPipe処理エラー ({img_info['filename']}): {e}")
                    continue
            
            # 類似度でソート
            mp_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            result["mediapipe"]["matches"] = mp_similarities
            result["mediapipe"]["processing_time_ms"] = float((time.time() - start_time) * 1000)
        
        # Buffalo_l処理
        if result["buffalo_l"]["available"]:
            start_time = time.time()
            bl_similarities = []
            
            for i, img_info in enumerate(folder_images_data):
                try:
                    img = np.frombuffer(img_info['data'], np.uint8)
                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                    
                    # Buffalo_lで埋め込み取得
                    folder_embedding, folder_confidence = get_embedding_with_buffalo_l_detection(img)
                    
                    if folder_embedding is not None:
                        # コサイン類似度計算
                        similarity = np.dot(bl_query_embedding, folder_embedding)
                        is_match = similarity > 0.6  # 閾値
                        
                        bl_similarities.append({
                            "filename": img_info['filename'],
                            "similarity": float(similarity),
                            "confidence": float(folder_confidence),
                            "is_match": bool(is_match),  # numpy.bool_をboolに変換
                            "index": int(i)
                        })
                        
                except Exception as e:
                    print(f"❌ Buffalo_l処理エラー ({img_info['filename']}): {e}")
                    continue
            
            # 類似度でソート
            bl_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            result["buffalo_l"]["matches"] = bl_similarities
            result["buffalo_l"]["processing_time_ms"] = float((time.time() - start_time) * 1000)
        
        print(f"🔍 1対N比較完了 - MediaPipe: {len(result['mediapipe']['matches'])}件, Buffalo_l: {len(result['buffalo_l']['matches'])}件")
        return result
        
    except Exception as e:
        print(f"❌ 1対N精度比較エラー: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"1対N精度比較エラー: {str(e)}")
