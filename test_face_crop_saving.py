#!/usr/bin/env python3
"""
顔切り出し画像保存機能のテストスクリプト
"""

import cv2
import os
import app

def test_face_crop_saving():
    """顔切り出し画像保存機能のテスト"""
    print("🔍 顔切り出し画像保存テスト開始")
    
    # テスト画像パス
    test_image_path = "static/uploads/e3761115e16747ba83cc406b2df617c9_2576.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"❌ テスト画像が見つかりません: {test_image_path}")
        return
    
    print(f"📷 テスト画像: {test_image_path}")
    
    # 切り出し画像保存ディレクトリを確認
    crop_dir = "static/face_crops"
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir, exist_ok=True)
        print(f"📁 切り出し画像ディレクトリ作成: {crop_dir}")
    
    # 保存前のファイル数を確認
    files_before = len([f for f in os.listdir(crop_dir) if f.endswith('.jpg')])
    print(f"📊 保存前のファイル数: {files_before}")
    
    try:
        # 顔検出と切り出し画像保存を実行
        result = app.detect_and_align_face(test_image_path, save_crop=True)
        
        if result is not None:
            print(f"✅ 顔検出成功: 出力サイズ={result.shape}")
            
            # 保存後のファイル数を確認
            files_after = len([f for f in os.listdir(crop_dir) if f.endswith('.jpg')])
            print(f"📊 保存後のファイル数: {files_after}")
            
            if files_after > files_before:
                print(f"✅ 切り出し画像が正常に保存されました: {files_after - files_before}枚追加")
                
                # 保存されたファイルを一覧表示
                print("📋 保存されたファイル:")
                crop_files = sorted([f for f in os.listdir(crop_dir) if f.endswith('.jpg')])
                for file in crop_files[-2:]:  # 最新の2ファイルを表示
                    file_path = os.path.join(crop_dir, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  - {file} ({file_size:,}バイト)")
            else:
                print("⚠️ 切り出し画像が保存されませんでした")
        else:
            print("❌ 顔検出に失敗しました")
            
    except Exception as e:
        print(f"❌ テストでエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_crop_saving()