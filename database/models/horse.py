"""
馬関連のデータベースモデル
"""
from sqlalchemy import Column, String, Date, DateTime, Integer, ForeignKey, DECIMAL, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class Horse(Base):
    """馬マスタテーブル"""
    __tablename__ = "horses"
    
    horse_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    name_eng = Column(String(100))
    sex = Column(String(10))
    birth_date = Column(Date)
    coat_color = Column(String(20))
    father_id = Column(String(20), ForeignKey("horses.horse_id"))
    mother_id = Column(String(20), ForeignKey("horses.horse_id"))
    trainer_id = Column(String(20), ForeignKey("trainers.trainer_id"))
    owner_name = Column(String(100))
    breeder_name = Column(String(100))
    birth_place = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーション
    father = relationship("Horse", foreign_keys=[father_id], remote_side=[horse_id])
    mother = relationship("Horse", foreign_keys=[mother_id], remote_side=[horse_id])
    trainer = relationship("Trainer", back_populates="horses")
    race_results = relationship("RaceResult", back_populates="horse")
    
    def __repr__(self):
        return f"<Horse(id={self.horse_id}, name={self.name})>"

class Jockey(Base):
    """騎手マスタテーブル"""
    __tablename__ = "jockeys"
    
    jockey_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    name_kana = Column(String(100))
    birth_date = Column(Date)
    license_date = Column(Date)
    belonging = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーション
    race_results = relationship("RaceResult", back_populates="jockey")
    
    def __repr__(self):
        return f"<Jockey(id={self.jockey_id}, name={self.name})>"

class Trainer(Base):
    """調教師マスタテーブル"""
    __tablename__ = "trainers"
    
    trainer_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    name_kana = Column(String(100))
    birth_date = Column(Date)
    license_date = Column(Date)
    belonging = Column(String(50))
    training_center = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # リレーション
    horses = relationship("Horse", back_populates="trainer")
    
    def __repr__(self):
        return f"<Trainer(id={self.trainer_id}, name={self.name})>"