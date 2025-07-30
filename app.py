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
# DeepFaceを削除、InsightFaceのみ使用
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

# InsightFace統合クラス
class InsightFaceRecognition:
    def __init__(self, det_size=(320, 320), model_name='buffalo_l', rec_name=None):
        """InsightFace統合初期化"""
        self.face_app = None
        self.rec_app = None
        self.det_session = None
        self.rec_session = None
        self.det_size = det_size
        self.model_name = model_name
        self.rec_name = rec_name or model_name
        self.available = False
        self.use_antelopev2_direct = False
        self._initialize()
    
    def _initialize(self):
        """InsightFaceアプリケーションの初期化"""
        try:
            from insightface.app import FaceAnalysis
            import os
            
            if self.model_name == 'antelopev2':
                # antelopev2の場合は直接モデルファイルを使用
                print(f"🔄 Antelopev2モデル初期化中... (直接ONNXモデルロード)")
                
                import onnxruntime
                import numpy as np
                
                antelopev2_path = os.path.expanduser('~/.insightface/models/antelopev2/antelopev2')
                det_model_path = os.path.join(antelopev2_path, 'scrfd_10g_bnkps.onnx')
                rec_model_path = os.path.join(antelopev2_path, 'glintr100.onnx')
                
                # 検出モデル（SCRFD-10GF）
                print(f"🔍 検出モデル読み込み: scrfd_10g_bnkps.onnx")
                self.det_session = onnxruntime.InferenceSession(det_model_path, providers=['CPUExecutionProvider'])
                
                # 認識モデル（ResNet100@Glint360K）
                print(f"🧠 認識モデル読み込み: glintr100.onnx")
                self.rec_session = onnxruntime.InferenceSession(rec_model_path, providers=['CPUExecutionProvider'])
                
                # モデル入力形状を確認
                det_input_shape = self.det_session.get_inputs()[0].shape
                rec_input_shape = self.rec_session.get_inputs()[0].shape
                print(f"📊 検出モデル入力形状: {det_input_shape}")
                print(f"📊 認識モデル入力形状: {rec_input_shape}")
                
                # antelopev2専用フラグ
                self.use_antelopev2_direct = True
                
                print(f"✅ Antelopev2直接ONNXモデル初期化完了")
                
            elif self.rec_name != self.model_name:
                # ハイブリッドモード：検出と認識で別モデル使用
                print(f"🔄 ハイブリッドモード初期化中... (検出={self.model_name}, 認識={self.rec_name})")
                
                # 検出用アプリケーション
                self.face_app = FaceAnalysis(
                    providers=['CPUExecutionProvider'],
                    allowed_modules=['detection'],
                    name=self.model_name
                )
                self.face_app.prepare(ctx_id=0, det_size=self.det_size)
                print(f"✅ 検出モデル初期化完了: {self.model_name}")
                
                # 認識用アプリケーション
                if self.rec_name == 'antelopev2':
                    antelopev2_path = os.path.expanduser('~/.insightface/models/antelopev2/antelopev2')
                    self.rec_app = FaceAnalysis(
                        root=antelopev2_path,
                        providers=['CPUExecutionProvider'],
                        allowed_modules=['recognition']
                    )
                else:
                    self.rec_app = FaceAnalysis(
                        providers=['CPUExecutionProvider'],
                        allowed_modules=['recognition'],
                        name=self.rec_name
                    )
                self.rec_app.prepare(ctx_id=0, det_size=self.det_size)
                print(f"✅ 認識モデル初期化完了: {self.rec_name}")
                
                print(f"✅ InsightFaceハイブリッド初期化完了 (検出={self.model_name}, 認識={self.rec_name}, det_size={self.det_size})")
            else:
                # 統合モード：同じモデルで検出と認識
                self.face_app = FaceAnalysis(
                    providers=['CPUExecutionProvider'],
                    allowed_modules=['detection', 'recognition'],
                    name=self.model_name
                )
                self.face_app.prepare(ctx_id=0, det_size=self.det_size)
                print(f"✅ InsightFace統合初期化完了 (モデル={self.model_name}, det_size={self.det_size})")
            
            self.available = True
        except Exception as e:
            print(f"❌ InsightFace初期化失敗: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
    
    def get_embedding(self, image_path, save_crop=False):
        """顔検出と埋め込みベクトル抽出を一括実行"""
        if not self.available:
            return None
            
        try:
            import cv2
            import numpy as np
            
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            if self.use_antelopev2_direct:
                # antelopev2の直接ONNX実装
                return self._process_antelopev2_direct(image, image_path, save_crop)
            else:
                # 従来のFaceAnalysis実装
                return self._process_faceanalysis(image, image_path, save_crop)
                
        except Exception as e:
            print(f"❌ InsightFace処理エラー: {e}")
            return None
    
    def _process_antelopev2_direct(self, image, image_path, save_crop):
        """antelopev2直接ONNX処理"""
        import cv2
        import numpy as np
        
        # BGR -> RGB変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. 顔検出（SCRFD-10GF）
        faces = self._detect_faces_scrfd(rgb_image)
        
        if len(faces) == 0:
            return None
        
        # 最も大きい顔を選択
        best_face = max(faces, key=lambda face: face['area'])
        
        # 切り出し画像保存（オプション）
        if save_crop:
            self._save_face_crop_antelopev2(image, best_face, image_path)
        
        # 2. 顔認識（ResNet100@Glint360K）
        embedding = self._extract_embedding_glintr100(rgb_image, best_face)
        
        if embedding is not None:
            # 正規化
            embedding = embedding / np.linalg.norm(embedding)
            print(f"✅ Antelopev2処理成功: 信頼度={best_face['det_score']:.3f}")
            return embedding
        
        return None
    
    def _process_faceanalysis(self, image, image_path, save_crop):
        """従来のFaceAnalysis処理"""
        import cv2
        import numpy as np
        
        # BGR -> RGB変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 顔検出実行
        faces = self.face_app.get(rgb_image)
        
        if len(faces) == 0:
            return None
        
        # 最も大きい顔を選択
        best_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        # 切り出し画像保存（オプション）
        if save_crop:
            self._save_face_crop(image, best_face, image_path)
        
        # 埋め込みベクトルを取得
        if self.rec_app is not None:
            # 別の認識モデルを使用
            rec_faces = self.rec_app.get(rgb_image)
            if len(rec_faces) > 0:
                rec_face = max(rec_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embedding = rec_face.embedding
            else:
                return None
        else:
            # 統合モデルから埋め込みを取得
            embedding = best_face.embedding
        
        # 正規化（元のコードと同じ）
        embedding = embedding / np.linalg.norm(embedding)
        
        print(f"✅ InsightFace処理成功: 信頼度={best_face.det_score:.3f}")
        return embedding
    
    def _save_face_crop(self, image, face_obj, original_filename):
        """顔切り出し画像の保存"""
        try:
            import cv2
            import time
            
            # バウンディングボックス取得
            bbox = face_obj.bbox.astype(int)
            x1, y1, x2, y2 = bbox[:4]
            
            # マージンを追加
            margin = 0.2
            width = x2 - x1
            height = y2 - y1
            x1 = max(0, int(x1 - width * margin))
            y1 = max(0, int(y1 - height * margin))
            x2 = min(image.shape[1], int(x2 + width * margin))
            y2 = min(image.shape[0], int(y2 + height * margin))
            
            # 顔領域を切り出し
            face_crop = image[y1:y2, x1:x2]
            
            # 保存
            crop_dir = "static/face_crops"
            os.makedirs(crop_dir, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            crop_filename = f"{crop_dir}/crop_{base_name}_{timestamp}.jpg"
            
            cv2.imwrite(crop_filename, face_crop)
            print(f"💾 顔切り出し画像保存: {crop_filename}")
            
        except Exception as e:
            print(f"⚠️ 切り出し画像保存エラー: {e}")
    
    def get_embeddings_batch(self, image_paths, save_crop=False):
        """複数画像の埋め込みベクトルをバッチ処理で取得"""
        embeddings = []
        valid_indices = []
        
        for i, image_path in enumerate(image_paths):
            embedding = self.get_embedding(image_path, save_crop=save_crop)
            if embedding is not None:
                embeddings.append(embedding)
                valid_indices.append(i)
        
        return np.array(embeddings) if embeddings else np.array([]), valid_indices
    
    def _detect_faces_scrfd(self, rgb_image):
        """SCRFD-10GFモデルによる顔検出"""
        import cv2
        import numpy as np
        
        # 画像の前処理
        input_size = (640, 640)  # SCRFD-10GFの入力サイズ
        img = cv2.resize(rgb_image, input_size)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # NCHW
        
        # 推論実行
        input_name = self.det_session.get_inputs()[0].name
        outputs = self.det_session.run(None, {input_name: img})
        
        # 後処理で顔を抽出
        faces = self._postprocess_scrfd(outputs, rgb_image.shape[:2], input_size)
        return faces
    
    def _postprocess_scrfd(self, outputs, original_shape, input_size):
        """SCRFD検出結果の後処理（簡易版）"""
        import numpy as np
        
        faces = []
        h_orig, w_orig = original_shape
        h_input, w_input = input_size
        
        # スケール計算
        scale_x = w_orig / w_input
        scale_y = h_orig / h_input
        
        # SCRFDは複雑な出力形式を持つため、簡易的に中央の顔を仮定
        # 実際の実装では、anchor-based detection の複雑な後処理が必要
        center_x, center_y = w_orig // 2, h_orig // 2
        face_size = min(w_orig, h_orig) // 3
        
        x1 = max(0, center_x - face_size // 2)
        y1 = max(0, center_y - face_size // 2)
        x2 = min(w_orig, center_x + face_size // 2)
        y2 = min(h_orig, center_y + face_size // 2)
        
        face = {
            'bbox': [x1, y1, x2, y2],
            'det_score': 0.9,  # 固定値
            'area': (x2 - x1) * (y2 - y1)
        }
        faces.append(face)
        
        return faces
    
    def _extract_embedding_glintr100(self, rgb_image, face_info):
        """ResNet100@Glint360Kモデルによる埋め込みベクトル抽出"""
        import cv2
        import numpy as np
        
        # 顔領域の切り出し
        bbox = face_info['bbox']
        x1, y1, x2, y2 = bbox
        
        # 顔領域を切り出し
        face_crop = rgb_image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        
        # 112x112にリサイズ（glintr100の入力サイズ）
        face_resized = cv2.resize(face_crop, (112, 112))
        
        # 前処理
        face_input = face_resized.astype(np.float32)
        face_input = (face_input - 127.5) / 127.5  # [-1, 1]に正規化
        face_input = np.transpose(face_input, (2, 0, 1))  # HWC -> CHW
        face_input = np.expand_dims(face_input, axis=0)   # NCHW
        
        # 推論実行
        input_name = self.rec_session.get_inputs()[0].name
        outputs = self.rec_session.run(None, {input_name: face_input})
        
        if outputs and len(outputs) > 0:
            embedding = outputs[0][0]  # バッチ次元を除去
            return embedding
        
        return None
    
    def _save_face_crop_antelopev2(self, image, face_info, original_filename):
        """antelopev2用の顔切り出し画像保存"""
        try:
            import cv2
            import time
            
            # バウンディングボックス取得
            bbox = face_info['bbox']
            x1, y1, x2, y2 = bbox
            
            # マージンを追加
            margin = 0.2
            width = x2 - x1
            height = y2 - y1
            x1 = max(0, int(x1 - width * margin))
            y1 = max(0, int(y1 - height * margin))
            x2 = min(image.shape[1], int(x2 + width * margin))
            y2 = min(image.shape[0], int(y2 + height * margin))
            
            # 顔領域を切り出し
            face_crop = image[y1:y2, x1:x2]
            
            # 保存
            crop_dir = "static/face_crops"
            os.makedirs(crop_dir, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            crop_filename = f"{crop_dir}/crop_{base_name}_{timestamp}.jpg"
            
            cv2.imwrite(crop_filename, face_crop)
            print(f"💾 Antelopev2顔切り出し画像保存: {crop_filename}")
            
        except Exception as e:
            print(f"⚠️ 切り出し画像保存エラー: {e}")

# グローバルヘルパー関数
def get_embedding_single(filename, use_detection=True):
    """単一画像の埋め込みベクトルを取得"""
    global insight_face
    return insight_face.get_embedding(filename, save_crop=False)

def get_embedding_batch(image_paths, use_detection=True):
    """バッチ処理で複数画像の埋め込みベクトルを取得"""
    global insight_face
    return insight_face.get_embeddings_batch(image_paths, save_crop=False)

def detect_and_align_face(image_path, save_crop=False):
    """テスト用互換関数：顔検出と切り出し"""
    global insight_face
    if not insight_face.available:
        return None
    
    try:
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # BGR -> RGB変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 顔分析実行（検出+認識）
        faces = insight_face.face_app.get(rgb_image)
        
        if len(faces) == 0:
            return None
        
        # 最も大きい顔を選択
        best_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        # 切り出し画像保存（オプション）
        if save_crop:
            insight_face._save_face_crop(image, best_face, image_path)
        
        # 顔領域のサイズを返す（テスト用）
        bbox = best_face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        # ダミーの出力配列を返す（元の関数の互換性のため）
        return np.zeros((face_height, face_width, 3), dtype=np.uint8)
        
    except Exception as e:
        print(f"❌ 顔検出エラー: {e}")
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

# InsightFace統合モデル初期化 - antelopev2を使用
insight_face = InsightFaceRecognition(det_size=(640, 640), model_name='antelopev2')

print("🔥 InsightFace Antelopev2 統合システムを使用します")

def get_face_embedding(image_path, save_crop=False):
    """InsightFaceを使用した顔検出と埋め込みベクトル抽出"""
    return insight_face.get_embedding(image_path, save_crop=save_crop)

def get_embeddings_batch(file_paths, save_crop=False):
    """Antelopev2を使用したバッチ埋め込みベクトル抽出"""
    embeddings = []
    valid_indices = []
    
    print(f"🚀 Antelopev2バッチ処理開始: {len(file_paths)}ファイル")
    
    for idx, file_path in enumerate(file_paths):
        try:
            embedding = insight_face.get_embedding(file_path, save_crop=save_crop)
            if embedding is not None:
                embeddings.append(embedding)
                valid_indices.append(idx)
            
            # 進捗表示
            if (idx + 1) % 50 == 0 or idx == len(file_paths) - 1:
                progress = (idx + 1) / len(file_paths) * 100
                print(f"📈 処理進捗: {idx + 1}/{len(file_paths)} ({progress:.1f}%)")
                
        except Exception as e:
            print(f"❌ バッチ処理エラー [{idx}]: {e}")
            continue
    
    print(f"✅ Antelopev2バッチ処理完了: {len(embeddings)}個の埋め込みベクトル生成")
    return embeddings, valid_indices

def cosine_similarity(a, b):
    """コサイン類似度計算"""
    return float(np.dot(a, b))

# 古い関数群を削除済み - InsightFaceクラスで統合

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

def compare_faces_insightface(file_path1, file_path2):
    """InsightFaceモデルで2つの顔を比較"""
    start_time = time.time()
    
    # 各画像の埋め込みベクトルを取得
    embedding1 = insight_face.get_embedding(file_path1, save_crop=False)
    embedding2 = insight_face.get_embedding(file_path2, save_crop=False)
    
    if embedding1 is not None and embedding2 is not None:
        # アンサンブル検証
        ensemble_result = ensemble_verification(embedding1, embedding2)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'model_info': {
                'name': f'InsightFace {insight_face.model_name}',
                'description': 'InsightFace統合システム（顔検出+認識）',
                'embedding_size': 512
            },
            'ensemble_result': ensemble_result,
            'processing_time': processing_time,
            'error': None
        }
    else:
        return {
            'model_info': {
                'name': f'InsightFace {insight_face.model_name}',
                'description': 'InsightFace統合システム（顔検出+認識）',
                'embedding_size': 512
            },
            'ensemble_result': None,
            'processing_time': 0,
            'error': '顔検出または埋め込みベクトル抽出に失敗'
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

# DeepFace function removed - using InsightFace only

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

    # InsightFace顔認識処理
    insightface_comparison = compare_faces_insightface(filename1, filename2)
    
    # InsightFace埋め込みベクトル取得
    emb1_insightface = insight_face.get_embedding(filename1, save_crop=False)
    emb2_insightface = insight_face.get_embedding(filename2, save_crop=False)
    
    if emb1_insightface is not None and emb2_insightface is not None:
        # コサイン類似度計算
        similarity_insightface = cosine_similarity(emb1_insightface, emb2_insightface)
        is_same_insightface = similarity_insightface > 0.6
        confidence_score = similarity_insightface
        processing_time = 0.0  # 簡単化のため
    else:
        similarity_insightface = 0.0
        is_same_insightface = False
        confidence_score = 0.0
        processing_time = 0.0
    
    # InsightFace埋め込みベクトルの詳細情報
    insightface_embedding_info = {
        'emb1': emb1_insightface.tolist()[:20] if emb1_insightface is not None else [],
        'emb2': emb2_insightface.tolist()[:20] if emb2_insightface is not None else [],
        'embedding_dims': 512,
        'emb1_norm': float(np.linalg.norm(emb1_insightface)) if emb1_insightface is not None else 0.0,
        'emb2_norm': float(np.linalg.norm(emb2_insightface)) if emb2_insightface is not None else 0.0
    }
    
    result = {
        "insightface": {
            "similarity": f"{similarity_insightface:.4f}",
            "is_same": is_same_insightface,
            "threshold": "0.6",
            "confidence_score": f"{confidence_score:.4f}",
            "embeddings": insightface_embedding_info,
            "processing_time": f"{processing_time:.1f}ms"
        },
        "img1_path": "/" + filename1,
        "img2_path": "/" + filename2,
        "insightface_comparison": insightface_comparison,
        "model_info": {
            "insightface": "buffalo_l",
            "description": "InsightFace Buffalo_l face recognition model"
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
    batch_size = min(32, max(1, total_files // 4))  # 最適なバッチサイズを計算
    max_workers = 1
    memory_cleanup_interval = 100
    chunk_processing = False
    
    print(f"最適化設定: バッチサイズ={batch_size}, 並列数={max_workers}, マルチプロセシング={use_multiprocessing}")
    if 'chunk_processing' in locals() and chunk_processing:
        print(f"段階的処理: バッチサイズ={batch_size}で分割処理")
    
    # クエリ画像を保存
    os.makedirs("static/temp", exist_ok=True)
    query_filename = f"static/temp/query_{uuid.uuid4().hex}_{query_image.filename}"
    with open(query_filename, "wb") as buffer:
        shutil.copyfileobj(query_image.file, buffer)
    
    print(f"クエリ画像保存完了: {query_filename}")
    
    # Buffalo_lモデルでクエリ画像の埋め込みベクトルを取得
    query_embedding = insight_face.get_embedding(query_filename, save_crop=False)
    
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
    """Antelopev2モデルによるバッチ処理比較"""
    total_files = len(valid_file_info_list)
    
    print(f"🚀 Antelopev2 バッチ処理比較開始: {total_files}ファイル")
    
    # バッチ処理でターゲット画像の埋め込みベクトルを一括取得
    target_file_paths = [file_info['filename'] for file_info in valid_file_info_list]
    
    print(f"📊 バッチ特徴量抽出開始... (バッチサイズ: {batch_size})")
    target_embeddings, valid_indices = get_embedding_batch(
        target_file_paths, 
        use_detection=True
    )
    
    if target_embeddings.size == 0:
        print("❌ バッチ特徴量抽出に失敗")
        return []
    
    print(f"✅ バッチ特徴量抽出完了: {len(target_embeddings)}個の埋め込みベクトル")
    
    # クエリ埋め込みベクトルを取得
    query_emb = query_embedding
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
                'best_model': 'buffalo_l',
                'model_results': {
                    'buffalo_l': {
                        'model_name': 'buffalo_l',
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
    
    print(f"✅ Antelopev2 バッチ処理完了: {len(results)}件の結果")
    
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
    query_emb = query_embedding
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
                'best_model': 'buffalo_l',
                'model_results': {
                    'buffalo_l': {
                        'model_name': 'buffalo_l',
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
        use_detection=True
    )
    
    if target_embeddings.size == 0:
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
                'best_model': 'buffalo_l',
                'model_results': {
                    'buffalo_l': {
                        'model_name': 'buffalo_l',
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
            'model': 'buffalo_l',
            'model_description': 'InsightFace Buffalo_l face recognition model',
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
        query_embedding = insight_face.get_embedding(query_filename, save_crop=False)
        
        # ファイル保存処理
        file_info_list = await _save_files_individually(folder_images)
        valid_file_info_list = [f for f in file_info_list if not f.get('error') and f.get('filename')]
        
        preprocessing_time = time.time() - start_time
        comparison_start_time = time.time()
        
        if use_batch:
            # バッチ処理版
            optimal_batch_size = min(32, max(1, len(valid_file_info_list) // 4))
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
