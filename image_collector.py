#!/usr/bin/env python3
"""
画像ファイル集約スクリプト

指定されたディレクトリ内を再帰的に探索し、すべての画像ファイルを
1つのディレクトリに集約します。

使用方法:
    python image_collector.py <元ディレクトリ> <出力ディレクトリ>

特徴:
- 再帰的なディレクトリ探索
- 複数の画像形式をサポート
- ファイル重複の自動リネーム
- 詳細なログ出力
- 安全なファイル操作
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Set
import hashlib

# 対応する画像形式
SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
    '.webp', '.avif', '.svg', '.ico', '.raw', '.heic', '.heif'
}

class ImageCollector:
    """画像ファイル集約クラス"""
    
    def __init__(self, source_dir: str, output_dir: str, copy_mode: bool = True):
        """
        初期化
        
        Args:
            source_dir: 元ディレクトリパス
            output_dir: 出力ディレクトリパス
            copy_mode: True=コピー, False=移動
        """
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.copy_mode = copy_mode
        self.processed_files = 0
        self.skipped_files = 0
        self.error_files = 0
        self.duplicate_names: Set[str] = set()
        
        # ログ設定
        self._setup_logging()
        
    def _setup_logging(self):
        """ログ設定を初期化"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('image_collector.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def is_image_file(self, file_path: Path) -> bool:
        """
        ファイルが画像ファイルかどうかを判定
        
        Args:
            file_path: チェックするファイルパス
            
        Returns:
            bool: 画像ファイルの場合True
        """
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    
    def find_image_files(self) -> List[Path]:
        """
        ディレクトリ内の画像ファイルを再帰的に探索
        
        Returns:
            List[Path]: 見つかった画像ファイルのリスト
        """
        image_files = []
        
        self.logger.info(f"ディレクトリ探索開始: {self.source_dir}")
        
        try:
            for root, dirs, files in os.walk(self.source_dir):
                current_path = Path(root)
                
                for file in files:
                    file_path = current_path / file
                    
                    if self.is_image_file(file_path):
                        image_files.append(file_path)
                        
        except Exception as e:
            self.logger.error(f"ディレクトリ探索エラー: {e}")
            
        self.logger.info(f"画像ファイル発見数: {len(image_files)}")
        return image_files
    
    def generate_unique_filename(self, original_path: Path) -> str:
        """
        重複を避けるユニークなファイル名を生成
        
        Args:
            original_path: 元のファイルパス
            
        Returns:
            str: ユニークなファイル名
        """
        base_name = original_path.stem
        extension = original_path.suffix
        counter = 1
        
        # 元のファイル名を試す
        new_name = f"{base_name}{extension}"
        if new_name not in self.duplicate_names:
            self.duplicate_names.add(new_name)
            return new_name
        
        # 重複している場合、連番を追加
        while True:
            new_name = f"{base_name}_{counter:03d}{extension}"
            if new_name not in self.duplicate_names:
                self.duplicate_names.add(new_name)
                return new_name
            counter += 1
    
    def get_file_hash(self, file_path: Path) -> str:
        """
        ファイルのMD5ハッシュを計算
        
        Args:
            file_path: ファイルパス
            
        Returns:
            str: MD5ハッシュ値
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception as e:
            self.logger.error(f"ハッシュ計算エラー {file_path}: {e}")
            return ""
        return hash_md5.hexdigest()
    
    def copy_or_move_file(self, source_path: Path, destination_path: Path):
        """
        ファイルをコピーまたは移動
        
        Args:
            source_path: 元ファイルパス
            destination_path: 出力先ファイルパス
        """
        try:
            if self.copy_mode:
                shutil.copy2(source_path, destination_path)
                self.logger.debug(f"コピー完了: {source_path} -> {destination_path}")
            else:
                shutil.move(str(source_path), str(destination_path))
                self.logger.debug(f"移動完了: {source_path} -> {destination_path}")
                
            self.processed_files += 1
            
        except Exception as e:
            self.logger.error(f"ファイル操作エラー {source_path}: {e}")
            self.error_files += 1
    
    def collect_images(self) -> bool:
        """
        画像ファイルを集約
        
        Returns:
            bool: 成功した場合True
        """
        # 出力ディレクトリを作成
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"出力ディレクトリ: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"出力ディレクトリ作成エラー: {e}")
            return False
        
        # 画像ファイルを探索
        image_files = self.find_image_files()
        
        if not image_files:
            self.logger.warning("画像ファイルが見つかりませんでした")
            return True
        
        # ファイル処理
        total_files = len(image_files)
        
        for i, image_path in enumerate(image_files, 1):
            # 進捗表示
            if i % 10 == 0 or i == total_files:
                self.logger.info(f"処理中... {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            try:
                # ユニークなファイル名を生成
                new_filename = self.generate_unique_filename(image_path)
                destination_path = self.output_dir / new_filename
                
                # ファイルをコピー/移動
                self.copy_or_move_file(image_path, destination_path)
                
            except Exception as e:
                self.logger.error(f"ファイル処理エラー {image_path}: {e}")
                self.error_files += 1
        
        return True
    
    def print_summary(self):
        """処理結果のサマリーを表示"""
        self.logger.info("\n" + "="*50)
        self.logger.info("処理完了サマリー")
        self.logger.info("="*50)
        self.logger.info(f"元ディレクトリ: {self.source_dir}")
        self.logger.info(f"出力ディレクトリ: {self.output_dir}")
        self.logger.info(f"処理モード: {'コピー' if self.copy_mode else '移動'}")
        self.logger.info(f"処理成功: {self.processed_files} ファイル")
        self.logger.info(f"スキップ: {self.skipped_files} ファイル")
        self.logger.info(f"エラー: {self.error_files} ファイル")
        self.logger.info("="*50)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='ディレクトリ内の画像ファイルを1つのディレクトリに集約'
    )
    parser.add_argument(
        'source_dir',
        help='元ディレクトリパス'
    )
    parser.add_argument(
        'output_dir',
        help='出力ディレクトリパス'
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='ファイルを移動（デフォルトはコピー）'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細ログを出力'
    )
    
    args = parser.parse_args()
    
    # ログレベル設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # ディレクトリの存在確認
    source_path = Path(args.source_dir)
    if not source_path.exists():
        print(f"エラー: 元ディレクトリが存在しません: {source_path}")
        return 1
    
    if not source_path.is_dir():
        print(f"エラー: 指定されたパスはディレクトリではありません: {source_path}")
        return 1
    
    # 画像収集実行
    collector = ImageCollector(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        copy_mode=not args.move
    )
    
    try:
        success = collector.collect_images()
        collector.print_summary()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        collector.logger.info("\n処理が中断されました")
        return 1
    except Exception as e:
        collector.logger.error(f"予期しないエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())