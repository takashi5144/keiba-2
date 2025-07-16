"""
レース関連のデータベースモデル
"""
from sqlalchemy import Column, String, Date, DateTime, Integer, ForeignKey, DECIMAL, Text, Time, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class Race(Base):
    """レース情報テーブル"""
    __tablename__ = "races"
    
    race_id = Column(String(20), primary_key=True)
    race_date = Column(Date, nullable=False, index=True)
    race_number = Column(Integer, nullable=False)
    race_name = Column(String(200))
    race_name_eng = Column(String(200))
    track = Column(String(50), nullable=False, index=True)
    course_type = Column(String(10))  # 右、左、直線
    distance = Column(Integer, nullable=False, index=True)
    surface = Column(String(20), nullable=False)  # 芝、ダート
    weather = Column(String(20))
    track_condition = Column(String(20))
    race_class = Column(String(50))
    grade = Column(String(10), index=True)
    age_requirement = Column(String(50))
    weight_requirement = Column(String(50))
    prize_1st = Column(Integer)
    prize_2nd = Column(Integer)
    prize_3rd = Column(Integer)
    prize_4th = Column(Integer)
    prize_5th = Column(Integer)
    start_time = Column(Time)
    entries_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーション
    race_results = relationship("RaceResult", back_populates="race")
    predictions = relationship("Prediction", back_populates="race")
    
    def __repr__(self):
        return f"<Race(id={self.race_id}, name={self.race_name}, date={self.race_date})>"

class RaceResult(Base):
    """レース結果テーブル"""
    __tablename__ = "race_results"
    
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String(20), ForeignKey("races.race_id"), nullable=False)
    horse_id = Column(String(20), ForeignKey("horses.horse_id"), nullable=False)
    jockey_id = Column(String(20), ForeignKey("jockeys.jockey_id"))
    finish_position = Column(Integer)
    post_position = Column(Integer)
    horse_number = Column(Integer)
    horse_weight = Column(Integer)
    weight_change = Column(Integer)
    odds = Column(DECIMAL(6, 2))
    popularity = Column(Integer)
    time_seconds = Column(DECIMAL(6, 2))
    margin = Column(String(10))
    corner_positions = Column(String(50))  # "1-2-3-4"形式
    final_600m_time = Column(DECIMAL(4, 2))
    horse_age = Column(Integer)
    horse_sex = Column(String(10))
    carried_weight = Column(DECIMAL(4, 1))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # リレーション
    race = relationship("Race", back_populates="race_results")
    horse = relationship("Horse", back_populates="race_results")
    jockey = relationship("Jockey", back_populates="race_results")
    
    def __repr__(self):
        return f"<RaceResult(race={self.race_id}, horse={self.horse_id}, position={self.finish_position})>"

class TrainingData(Base):
    """調教データテーブル"""
    __tablename__ = "training_data"
    
    training_id = Column(Integer, primary_key=True, autoincrement=True)
    horse_id = Column(String(20), ForeignKey("horses.horse_id"), nullable=False)
    training_date = Column(Date, nullable=False)
    training_course = Column(String(50))
    training_type = Column(String(50))
    time_seconds = Column(DECIMAL(5, 2))
    evaluation = Column(String(20))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)