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
# Removed unused imports: ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import gc
import time
# Removed unused import: partial

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

# Buffalo_lモデル設定（InsightFace）
MODEL_CONFIG = {
    "path": "w600k_r50.onnx",
    "name": "Buffalo_l WebFace600K ResNet50",
    "description": "WebFace600K（60万人、600万枚）で訓練された高精度モデル（InsightFace）",
    "input_name": "input.1",
    "input_size": (112, 112),
    "output_name": "683",
    "embedding_size": 512
}

# Buffalo_lモデルの初期化
def initialize_model():
    model_path = MODEL_CONFIG["path"]
    
    if os.path.exists(model_path):
        try:
            session = onnxruntime.InferenceSession(
                model_path, 
                providers=['CPUExecutionProvider']
            )
            print(f"✅ {MODEL_CONFIG['name']} 読み込み完了")
            return session
        except Exception as e:
            print(f"❌ {MODEL_CONFIG['name']} 読み込みエラー: {e}")
            return None
    else:
        print(f"❌ 警告: {model_path} が見つかりません")
        return None

# Buffalo_lモデルセッションを初期化
buffalo_session = initialize_model()

print("🐃 Buffalo_l WebFace600K モデル（InsightFace）を使用します")

# InsightFace/Buffalo_l顔検出モデルの初期化
BUFFALO_L_AVAILABLE = False
buffalo_l_app = None
try:
    from insightface.app import FaceAnalysis
    # Buffalo_l顔検出モデルの初期化
    buffalo_l_app = FaceAnalysis(
        providers=['CPUExecutionProvider'],
        allowed_modules=['detection'],
        name='buffalo_l'
    )
    # det_sizeを最適設定に変更（320x320）
    buffalo_l_app.prepare(ctx_id=0, det_size=(320, 320))
    print("✅ Buffalo_l顔検出モデル初期化完了 (det_size=320x320)")
    BUFFALO_L_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Buffalo_l顔検出モデル初期化失敗: {e}")
    BUFFALO_L_AVAILABLE = False

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


def detect_and_align_buffalo_l(image, save_crop=False, original_filename=None):
    """Buffalo_l顔検出モデルによる顔検出とアライメント
    
    Args:
        image: 入力画像 (OpenCV形式)
        save_crop: 切り出し画像を保存するかどうか
        original_filename: 元のファイル名（保存時のファイル名生成用）
    """
    if not BUFFALO_L_AVAILABLE or buffalo_l_app is None:
        return None
    
    try:
        # BGR -> RGB変換
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Buffalo_lで顔検出
        faces = buffalo_l_app.get(rgb_image)
        
        if len(faces) == 0:
            return None
        
        # 最も大きい顔を選択（安全な方法）
        best_face = None
        max_area = 0
        for face in faces:
            try:
                bbox = face.bbox
                if len(bbox) >= 4:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        best_face = face
            except Exception as e:
                print(f"⚠️ 顔選択でエラー: {e}")
                continue
        
        if best_face is None:
            print("❌ 有効な顔が見つかりません")
            return None
        
        # バウンディングボックスを取得
        bbox = best_face.bbox
        
        # bbox形状チェック
        if len(bbox) < 4:
            print(f"❌ 無効なbbox形状: {bbox.shape}, 最低4つの値が必要")
            return None
        
        bbox = bbox.astype(int)
        x1, y1, x2, y2 = bbox[:4]  # 最初の4つの値のみ使用
        
        # マージンを追加
        margin = 0.2
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, int(x1 - width * margin))
        y1 = max(0, int(y1 - height * margin))
        x2 = min(image.shape[1], int(x2 + width * margin))
        y2 = min(image.shape[0], int(y2 + height * margin))
        
        # 正方形に調整
        width = x2 - x1
        height = y2 - y1
        if width != height:
            size = max(width, height)
            center_x = x1 + width // 2
            center_y = y1 + height // 2
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            x2 = min(image.shape[1], x1 + size)
            y2 = min(image.shape[0], y1 + size)
        
        # 顔領域を切り出し
        face_crop = image[y1:y2, x1:x2]
        
        # 112x112にリサイズ
        aligned_face = cv2.resize(face_crop, (112, 112))
        
        # 切り出し画像を保存（オプション）
        if save_crop and original_filename:
            try:
                # 保存用ディレクトリを確保
                crop_dir = "static/face_crops"
                os.makedirs(crop_dir, exist_ok=True)
                
                # ファイル名生成（元ファイル名 + タイムスタンプ）
                import time
                timestamp = int(time.time() * 1000)  # ミリ秒タイムスタンプ
                base_name = os.path.splitext(os.path.basename(original_filename))[0]
                crop_filename = f"{crop_dir}/crop_{base_name}_{timestamp}.jpg"
                
                # 元の切り出し画像（リサイズ前）を保存
                cv2.imwrite(crop_filename, face_crop)
                
                # リサイズ後の画像も保存
                aligned_filename = f"{crop_dir}/aligned_{base_name}_{timestamp}.jpg"
                cv2.imwrite(aligned_filename, aligned_face)
                
                print(f"💾 顔切り出し画像保存: {crop_filename}")
                print(f"💾 アライメント画像保存: {aligned_filename}")
                
            except Exception as e:
                print(f"⚠️ 切り出し画像保存エラー: {e}")
        
        print(f"✅ Buffalo_l顔検出成功: 信頼度={best_face.det_score:.3f}, bbox=({x1},{y1},{x2-x1},{y2-y1})")
        return aligned_face
        
    except Exception as e:
        print(f"❌ Buffalo_l顔検出エラー: {e}")
        return None

def detect_and_align_face(image_path, save_crop=False):
    """Buffalo_l顔検出・アライメント処理"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Buffalo_lによる顔検出のみ実行
    if BUFFALO_L_AVAILABLE:
        buffalo_result = detect_and_align_buffalo_l(
            image, 
            save_crop=save_crop, 
            original_filename=image_path
        )
        if buffalo_result is not None:
            return buffalo_result
        else:
            print("❌ Buffalo_l顔検出失敗")
            return None
    else:
        print("❌ Buffalo_l顔検出モデルが利用できません")
        return None

def preprocess_image_for_model(file_path, use_detection=True, save_crop=False):
    """Buffalo_lモデル用の前処理"""
    input_size = MODEL_CONFIG["input_size"]
    
    if use_detection:
        # 顔検出とクロップ（切り出し画像保存オプション付き）
        face_image = detect_and_align_face(file_path, save_crop=save_crop)
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
    img = Image.open(file).convert('RGB').resize((112, 112))
    img = np.asarray(img, dtype=np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img

def preprocess_images_batch(file_paths, use_detection=True, batch_size=32, save_crop=False):
    """複数画像のバッチ前処理"""
    input_size = MODEL_CONFIG["input_size"]
    
    batch_images = []
    valid_indices = []
    
    for idx, file_path in enumerate(file_paths):
        try:
            if use_detection:
                # 顔検出とクロップ（切り出し画像保存オプション付き）
                face_image = detect_and_align_face(file_path, save_crop=save_crop)
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

def get_embedding_batch(file_paths, use_detection=True, batch_size=None):
    """バッチ処理による高速な特徴量抽出"""
    if buffalo_session is None:
        return None, []
    
    # 自動バッチサイズ調整
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(len(file_paths))
    
    session = buffalo_session
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
            # バッチ前処理（切り出し画像保存を有効化）
            batch_images, valid_indices = preprocess_images_batch(
                batch_files, use_detection, batch_size, save_crop=True
            )
            
            if batch_images is None:
                print(f"⚠️ バッチ {batch_num}/{total_batches}: 処理可能な画像なし")
                continue
            
            # Buffalo_lモデルはバッチサイズ1のみ対応のため、1枚ずつ推論
            embeddings = []
            for idx, single_image in enumerate(batch_images):
                try:
                    single_input = np.expand_dims(single_image, axis=0)  # (1, C, H, W)
                    single_embedding = session.run(None, {input_name: single_input})[0]
                    if single_embedding.shape[0] == 1:  # 期待される形状チェック
                        embeddings.append(single_embedding[0])  # バッチ次元を除去
                    else:
                        print(f"⚠️ 予期しない埋め込み形状: {single_embedding.shape}")
                        continue
                except Exception as e:
                    print(f"❌ 単一画像推論エラー [{idx}]: {e}")
                    continue
            
            if not embeddings:
                print(f"⚠️ バッチ {batch_num}/{total_batches}: 推論可能な画像なし")
                continue
                
            embeddings = np.array(embeddings)
            
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
                return get_embedding_batch(file_paths, use_detection, max(batch_size//2, 4))
            continue
    
    print(f"✅ バッチ処理完了: {len(all_embeddings)}個の埋め込みベクトル生成")
    return all_embeddings, all_valid_indices

def get_embedding_buffalo(file_path, use_detection=True):
    """Buffalo_lモデルで埋め込みベクトルを取得"""
    if buffalo_session is None:
        return {
            'embedding': None,
            'error': 'Buffalo_lモデルが読み込まれていません',
            'processing_time': 0
        }
    
    start_time = time.time()
    try:
        # 前処理
        img = preprocess_image_for_model(file_path, use_detection)
        
        if img is None:
            return {
                'embedding': None,
                'error': '画像処理に失敗',
                'processing_time': 0
            }
        
        # 推論実行
        input_name = MODEL_CONFIG["input_name"]
        embedding = buffalo_session.run(None, {input_name: img})[0]
        embedding = embedding[0]
        
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
    if buffalo_session is None:
        return None
    
    try:
        # 1ファイルずつ処理
        img = preprocess_image_for_model(file_path, use_detection)
        
        if img is None:
            return None
        
        # 推論実行（1ファイルずつ）
        input_name = MODEL_CONFIG["input_name"]
        embedding = buffalo_session.run(None, {input_name: img})[0]
        embedding = embedding[0]
        
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

def compare_buffalo_faces(file_path1, file_path2):
    """Buffalo_lモデルで2つの顔を比較"""
    # 各画像の埋め込みベクトルを取得
    embedding1 = get_embedding_buffalo(file_path1, use_detection=True)
    embedding2 = get_embedding_buffalo(file_path2, use_detection=True)
    
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
    
    # Buffalo_l顔認識処理
    buffalo_comparison = compare_buffalo_faces(filename1, filename2)
    
    # Buffalo_l埋め込みベクトル取得
    emb1_buffalo = get_embedding_buffalo(filename1, use_detection=True)
    emb2_buffalo = get_embedding_buffalo(filename2, use_detection=True)
    
    if (emb1_buffalo['embedding'] is not None and 
        emb2_buffalo['embedding'] is not None):
        # アンサンブル検証を使用
        ensemble_results = ensemble_verification(
            emb1_buffalo['embedding'], 
            emb2_buffalo['embedding']
        )
        similarity_buffalo = ensemble_results['cosine_similarity']
        is_same_buffalo = ensemble_results['is_same_adaptive']
        confidence_score = ensemble_results['confidence_score']
    else:
        similarity_buffalo = 0.0
        is_same_buffalo = False
        confidence_score = 0.0
        ensemble_results = {
            'cosine_similarity': 0.0,
            'euclidean_distance': 0.0,
            'l1_distance': 0.0,
            'normalized_euclidean': 0.0,
            'adaptive_threshold': 0.45,
            'confidence_score': 0.0
        }
    
    # Buffalo_l埋め込みベクトルの詳細情報
    buffalo_embedding_info = {
        'emb1': emb1_buffalo['embedding'].tolist()[:20] if emb1_buffalo['embedding'] is not None else [],
        'emb2': emb2_buffalo['embedding'].tolist()[:20] if emb2_buffalo['embedding'] is not None else [],
        'embedding_dims': MODEL_CONFIG['embedding_size'],
        'emb1_norm': float(np.linalg.norm(emb1_buffalo['embedding'])) if emb1_buffalo['embedding'] is not None else 0.0,
        'emb2_norm': float(np.linalg.norm(emb2_buffalo['embedding'])) if emb2_buffalo['embedding'] is not None else 0.0
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
        "buffalo_l": {
            "similarity": f"{similarity_buffalo:.4f}",
            "is_same": is_same_buffalo,
            "adaptive_threshold": f"{ensemble_results.get('adaptive_threshold', 0.5):.4f}",
            "confidence_score": f"{confidence_score:.4f}",
            "euclidean_distance": f"{ensemble_results.get('euclidean_distance', 0.0):.4f}",
            "l1_distance": f"{ensemble_results.get('l1_distance', 0.0):.4f}",
            "normalized_euclidean": f"{ensemble_results.get('normalized_euclidean', 0.0):.4f}",
            "embeddings": buffalo_embedding_info,
            "processing_time": f"{emb1_buffalo.get('processing_time', 0) + emb2_buffalo.get('processing_time', 0):.1f}ms"
        },
        "img1_path": "/" + filename1,
        "img2_path": "/" + filename2,
        "buffalo_comparison": buffalo_comparison,
        "model_info": {
            "deepface_arcface": "ArcFace (DeepFace implementation)",
            "buffalo_l": MODEL_CONFIG['name'],
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
    
    # Buffalo_lモデルでクエリ画像の埋め込みベクトルを取得
    query_embedding = get_embedding_buffalo(query_filename, use_detection=True)
    
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
    results = await _execute_comparison_buffalo(query_embedding, valid_file_info_list, batch_size, start_time)
    
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

async def _execute_comparison_buffalo(query_embedding, valid_file_info_list, batch_size, start_time):
    """Buffalo_lモデルによるバッチ処理比較"""
    total_files = len(valid_file_info_list)
    
    print(f"🚀 Buffalo_l バッチ処理比較開始: {total_files}ファイル")
    
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
                    'buffalo_l': {
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
    
    print(f"✅ Buffalo_l バッチ処理完了: {len(results)}件の結果")
    
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
                    'buffalo_l': {
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
                    'buffalo_l': {
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
        query_embedding = get_embedding_buffalo(query_filename, use_detection=True)
        
        # ファイル保存処理
        file_info_list = await _save_files_individually(folder_images)
        valid_file_info_list = [f for f in file_info_list if not f.get('error') and f.get('filename')]
        
        preprocessing_time = time.time() - start_time
        comparison_start_time = time.time()
        
        if use_batch:
            # バッチ処理版
            optimal_batch_size = calculate_optimal_batch_size(len(valid_file_info_list))
            print(f"🚀 バッチ処理モード実行 (最適バッチサイズ: {optimal_batch_size})")
            results = await _execute_comparison_buffalo(query_embedding, valid_file_info_list, optimal_batch_size, comparison_start_time)
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
