# HorseRacingAI - 高精度競馬予測システム

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com)

## 概要

HorseRacingAIは、機械学習を用いた高精度な競馬予測システムです。ネット競馬からのデータ収集、高度な特徴量エンジニアリング、アンサンブル機械学習モデル、そして資金管理戦略を統合した包括的なソリューションを提供します。

### 主な特徴

- 🏇 **自動データ収集**: ネット競馬からの包括的なデータスクレイピング
- 🤖 **アンサンブルML**: LightGBM、XGBoost、ニューラルネットワークの統合
- 📊 **高度な特徴量**: 100以上の特徴量による精密な分析
- 💰 **資金管理**: ケリー基準に基づく最適な賭け金計算
- 🚀 **高性能API**: FastAPIによるRESTful API
- 📈 **リアルタイム予測**: レース直前までの最新データ反映
- 🔍 **パフォーマンス監視**: Prometheus/Grafanaによる詳細な監視

## プロジェクト構造

```
keiba-analysis-tool/
├── services/              # マイクロサービス
│   ├── data_collector/    # データ収集サービス
│   ├── feature_engineering/   # 特徴量生成サービス
│   ├── prediction_engine/     # 予測エンジン
│   ├── money_management/      # 資金管理サービス
│   └── api_gateway/          # APIゲートウェイ
├── database/             # データベース関連
│   ├── models/          # SQLAlchemyモデル
│   └── migrations/      # Alembicマイグレーション
├── deployment/          # デプロイメント設定
│   ├── docker/         # Dockerファイル
│   └── kubernetes/     # Kubernetes設定
├── tests/              # テスト
├── docs/               # ドキュメント
└── docker-compose.yml  # Docker Compose設定
```

## 技術スタック

### バックエンド
- **言語**: Python 3.9+
- **Webフレームワーク**: FastAPI
- **タスクキュー**: Celery + Redis
- **データベース**: PostgreSQL
- **オブジェクトストレージ**: MinIO

### 機械学習
- **勾配ブースティング**: LightGBM, XGBoost
- **深層学習**: TensorFlow/Keras
- **データ処理**: Pandas, NumPy
- **特徴量エンジニアリング**: scikit-learn

### インフラストラクチャ
- **コンテナ化**: Docker, Docker Compose
- **監視**: Prometheus, Grafana
- **ログ管理**: Loguru

## クイックスタート

### 前提条件

- Docker 20.10+
- Docker Compose 2.0+
- Git

### インストール

1. リポジトリをクローン
```bash
git clone https://github.com/takashi5144/keiba-2.git
cd keiba-2
```

2. 環境変数を設定
```bash
cp .env.example .env
# .envファイルを編集して適切な値を設定
```

3. Dockerコンテナを起動
```bash
docker-compose up -d
```

### APIの使用

APIドキュメント: http://localhost:8000/docs

#### 主要エンドポイント

1. **レース予測**
```bash
GET /api/v1/predictions/{race_id}
```

2. **賭け推奨**
```bash
GET /api/v1/betting/{race_id}?bankroll=100000&risk_level=moderate
```

## 機械学習モデル

### アンサンブル構成

- **LightGBM** (40%): 高速で高精度な勾配ブースティング
- **XGBoost** (30%): 安定した性能の勾配ブースティング
- **Neural Network** (30%): 非線形パターンの捕捉

### パフォーマンス目標

- **ROI**: 5%以上
- **的中率**: 単勝15%以上
- **シャープレシオ**: 1.0以上
- **最大ドローダウン**: 20%以下

## 開発

### ローカル開発環境

```bash
# Poetry環境のセットアップ
poetry install

# 開発サーバーの起動
poetry run uvicorn services.api_gateway.main:app --reload

# テストの実行
poetry run pytest
```

## 免責事項

このシステムは研究・教育目的で開発されています。実際の賭博行為は自己責任で行ってください。