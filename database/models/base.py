"""
データベースモデルの基底クラス
"""
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import os
from dotenv import load_dotenv

load_dotenv()

# データベース設定
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/horseracing")

# エンジンの作成
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # 接続プーリングを無効化（Celeryワーカー用）
    echo=False
)

# セッションファクトリ
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ベースクラス
Base = declarative_base()
metadata = MetaData()

def get_db():
    """データベースセッションを取得"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """データベースの初期化"""
    Base.metadata.create_all(bind=engine)