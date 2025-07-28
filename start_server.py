#!/usr/bin/env python3
"""
大量ファイル処理対応のサーバー起動スクリプト
3520ファイルなどの大量データセット処理に最適化
ファイル数制限を10000に設定
"""

import os
import uvicorn

# 環境変数でファイル数制限を設定
os.environ['MAX_FORM_FILES'] = '10000'
os.environ['STARLETTE_MAX_FORM_FILES'] = '10000'

# Python起動時の制限も緩和
import sys
sys.setrecursionlimit(20000)  # 再帰制限も増加

print("🚀 大量ファイル処理サーバー起動準備")
print(f"📁 MAX_FORM_FILES環境変数: {os.environ.get('MAX_FORM_FILES', '未設定')}")
print(f"🔧 再帰制限: {sys.getrecursionlimit()}")

if __name__ == "__main__":
    # 3520ファイル対応の極限設定（ファイル数制限解除）
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        # 極限まで制限緩和
        limit_max_requests=10000000,  # 最大リクエスト数を極限まで増加
        limit_concurrency=10000,      # 同時接続数を極限まで増加
        timeout_keep_alive=3600,      # キープアライブタイムアウト（60分）
        timeout_graceful_shutdown=600,  # グレースフルシャットダウン（10分）
        # メモリとファイルサイズ制限を極限まで緩和
        backlog=16384,               # バックログ極限増加
        workers=1,                  # シングルワーカーで安定性重視
        # ログ設定
        log_level="info",
        access_log=False,           # アクセスログ無効化でパフォーマンス向上
        # HTTP設定
        http="h11",                 # HTTPサーバー実装
        loop="asyncio",             # イベントループ
        # 超大容量アップロード対応（ファイル数制限も含む）
        h11_max_incomplete_event_size=2048 * 1024 * 1024,  # 2GB
    )