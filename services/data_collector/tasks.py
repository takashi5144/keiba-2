"""
Celeryタスク定義
"""
from celery import Celery, Task
from celery.schedules import crontab
from datetime import datetime, date, timedelta
from typing import List, Dict
import os
from loguru import logger

from .scraper import NetKeibaScraper
from database.models.base import SessionLocal
from database.models.race import Race, RaceResult
from database.models.horse import Horse, Jockey, Trainer

# Celeryアプリケーションの設定
celery_app = Celery(
    'data_collector',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

# Celery設定
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Tokyo',
    enable_utc=True,
    beat_schedule={
        # 毎日午前4時にデータ収集
        'daily-data-collection': {
            'task': 'services.data_collector.tasks.collect_daily_races',
            'schedule': crontab(hour=4, minute=0),
        },
        # 1時間ごとに当日レースの更新
        'hourly-race-update': {
            'task': 'services.data_collector.tasks.update_today_races',
            'schedule': crontab(minute=0),
        },
    }
)

class DatabaseTask(Task):
    """データベース接続を管理するベースタスク"""
    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db

@celery_app.task(base=DatabaseTask, bind=True)
def collect_race_data(self, race_id: str) -> Dict:
    """単一レースのデータを収集"""
    logger.info(f"Collecting data for race: {race_id}")
    
    scraper = NetKeibaScraper(use_selenium=True)
    db = self.db
    
    try:
        # レース情報を取得
        race_data = scraper.scrape_race_info(race_id)
        if not race_data:
            logger.error(f"Failed to scrape race: {race_id}")
            return {'status': 'failed', 'race_id': race_id}
        
        # レース情報を保存
        race_info = race_data['race_info']
        race = db.query(Race).filter_by(race_id=race_id).first()
        
        if not race:
            race = Race(**race_info)
            db.add(race)
        else:
            for key, value in race_info.items():
                setattr(race, key, value)
        
        # レース結果を保存
        for result_data in race_data['results']:
            # 馬情報を収集（存在しない場合）
            horse = db.query(Horse).filter_by(horse_id=result_data['horse_id']).first()
            if not horse:
                collect_horse_data.delay(result_data['horse_id'])
            
            # 騎手情報を収集（存在しない場合）
            if result_data.get('jockey_id'):
                jockey = db.query(Jockey).filter_by(jockey_id=result_data['jockey_id']).first()
                if not jockey:
                    collect_jockey_data.delay(result_data['jockey_id'])
            
            # レース結果を保存
            existing_result = db.query(RaceResult).filter_by(
                race_id=race_id,
                horse_id=result_data['horse_id']
            ).first()
            
            if not existing_result:
                result = RaceResult(**result_data)
                db.add(result)
        
        db.commit()
        logger.info(f"Successfully collected data for race: {race_id}")
        return {'status': 'success', 'race_id': race_id}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error collecting race data: {e}")
        return {'status': 'error', 'race_id': race_id, 'error': str(e)}
    finally:
        scraper.close()

@celery_app.task(base=DatabaseTask, bind=True)
def collect_horse_data(self, horse_id: str) -> Dict:
    """馬データを収集"""
    logger.info(f"Collecting data for horse: {horse_id}")
    
    scraper = NetKeibaScraper()
    db = self.db
    
    try:
        horse_data = scraper.scrape_horse_info(horse_id)
        if not horse_data:
            logger.error(f"Failed to scrape horse: {horse_id}")
            return {'status': 'failed', 'horse_id': horse_id}
        
        horse = db.query(Horse).filter_by(horse_id=horse_id).first()
        
        if not horse:
            horse = Horse(**horse_data)
            db.add(horse)
        else:
            for key, value in horse_data.items():
                if value is not None:
                    setattr(horse, key, value)
        
        db.commit()
        logger.info(f"Successfully collected data for horse: {horse_id}")
        return {'status': 'success', 'horse_id': horse_id}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error collecting horse data: {e}")
        return {'status': 'error', 'horse_id': horse_id, 'error': str(e)}
    finally:
        scraper.close()

@celery_app.task(base=DatabaseTask, bind=True)
def collect_jockey_data(self, jockey_id: str) -> Dict:
    """騎手データを収集"""
    # 実装は horse_data と同様
    pass

@celery_app.task(base=DatabaseTask, bind=True)
def collect_daily_races(self, target_date: str = None) -> Dict:
    """指定日の全レースを収集"""
    if target_date is None:
        target_date = date.today()
    else:
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    logger.info(f"Collecting races for date: {target_date}")
    
    scraper = NetKeibaScraper()
    
    try:
        race_ids = scraper.scrape_race_list(target_date)
        logger.info(f"Found {len(race_ids)} races for {target_date}")
        
        # 各レースのデータ収集タスクをキューに追加
        for race_id in race_ids:
            collect_race_data.delay(race_id)
        
        return {
            'status': 'success',
            'date': str(target_date),
            'race_count': len(race_ids),
            'race_ids': race_ids
        }
        
    except Exception as e:
        logger.error(f"Error collecting daily races: {e}")
        return {
            'status': 'error',
            'date': str(target_date),
            'error': str(e)
        }
    finally:
        scraper.close()

@celery_app.task(base=DatabaseTask, bind=True)
def update_today_races(self) -> Dict:
    """当日のレース情報を更新"""
    return collect_daily_races.apply_async(args=[None]).get()

@celery_app.task
def cleanup_old_predictions(days: int = 90) -> Dict:
    """古い予測データをクリーンアップ"""
    # 90日以上前の予測データを削除
    pass