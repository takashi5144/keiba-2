"""
ネット競馬スクレイピングサービス
"""
import time
import random
from typing import Dict, List, Optional
from datetime import datetime, date
from urllib.parse import urljoin
import re

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests
from loguru import logger

from database.models.base import SessionLocal
from database.models.horse import Horse, Jockey, Trainer
from database.models.race import Race, RaceResult

class NetKeibaScraperConfig:
    """スクレイパー設定"""
    BASE_URL = "https://db.netkeiba.com"
    RACE_URL = "https://race.netkeiba.com"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    REQUEST_DELAY = (1, 3)
    MAX_RETRIES = 3
    TIMEOUT = 30
    
    # robots.txt準拠パス
    ALLOWED_PATHS = [
        "/race/list/",
        "/race/",
        "/horse/",
        "/jockey/",
        "/trainer/",
        "/owner/",
    ]

class NetKeibaScraper:
    """ネット競馬スクレイピングクラス"""
    
    def __init__(self, use_selenium: bool = False):
        self.config = NetKeibaScraperConfig()
        self.session = self._create_session()
        self.driver = None
        if use_selenium:
            self.driver = self._create_driver()
    
    def _create_session(self) -> requests.Session:
        """HTTPセッションを作成"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.config.USER_AGENT,
            'Accept-Language': 'ja,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Connection': 'keep-alive',
        })
        return session
    
    def _create_driver(self) -> webdriver.Chrome:
        """Seleniumドライバーを作成"""
        options = Options()
        options.add_argument(f'user-agent={self.config.USER_AGENT}')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # ヘッドレスモード（必要に応じて）
        # options.add_argument('--headless')
        
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    
    def _wait_random_delay(self):
        """ランダムな遅延を追加"""
        delay = random.uniform(*self.config.REQUEST_DELAY)
        time.sleep(delay)
    
    def _get_page(self, url: str, use_selenium: bool = False) -> Optional[BeautifulSoup]:
        """ページを取得してBeautifulSoupオブジェクトを返す"""
        logger.info(f"Fetching page: {url}")
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                self._wait_random_delay()
                
                if use_selenium and self.driver:
                    self.driver.get(url)
                    WebDriverWait(self.driver, self.config.TIMEOUT).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                else:
                    response = self.session.get(url, timeout=self.config.TIMEOUT)
                    response.raise_for_status()
                    response.encoding = response.apparent_encoding
                    soup = BeautifulSoup(response.content, 'html.parser')
                
                return soup
                
            except Exception as e:
                logger.error(f"Error fetching {url} (attempt {attempt + 1}): {e}")
                if attempt == self.config.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def scrape_race_list(self, target_date: date) -> List[str]:
        """指定日のレース一覧を取得"""
        date_str = target_date.strftime("%Y%m%d")
        url = f"{self.config.RACE_URL}/race/list/{date_str}/"
        
        soup = self._get_page(url)
        if not soup:
            return []
        
        race_ids = []
        race_links = soup.find_all('a', href=re.compile(r'/race/\d+/'))
        
        for link in race_links:
            race_id_match = re.search(r'/race/(\d+)/', link['href'])
            if race_id_match:
                race_ids.append(race_id_match.group(1))
        
        logger.info(f"Found {len(race_ids)} races for {date_str}")
        return list(set(race_ids))  # 重複除去
    
    def scrape_race_info(self, race_id: str) -> Optional[Dict]:
        """レース情報を取得"""
        url = f"{self.config.BASE_URL}/race/{race_id}/"
        soup = self._get_page(url, use_selenium=True)
        
        if not soup:
            return None
        
        try:
            race_data = {
                'race_id': race_id,
                'race_name': self._extract_text(soup, '.RaceName'),
                'race_date': self._parse_race_date(soup),
                'track': self._extract_track(soup),
                'distance': self._extract_distance(soup),
                'surface': self._extract_surface(soup),
                'weather': self._extract_text(soup, '.Weather'),
                'track_condition': self._extract_text(soup, '.TrackCondition'),
                'race_number': self._extract_race_number(soup),
                'grade': self._extract_grade(soup),
                'prize_money': self._extract_prize_money(soup),
            }
            
            # レース結果を取得
            results = self._extract_race_results(soup, race_id)
            
            return {
                'race_info': race_data,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error parsing race {race_id}: {e}")
            return None
    
    def _extract_race_results(self, soup: BeautifulSoup, race_id: str) -> List[Dict]:
        """レース結果を抽出"""
        results = []
        result_table = soup.find('table', class_='race_table_01')
        
        if not result_table:
            return results
        
        rows = result_table.find_all('tr')[1:]  # ヘッダーをスキップ
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 10:
                continue
            
            try:
                result = {
                    'race_id': race_id,
                    'finish_position': self._safe_int(cells[0].text),
                    'post_position': self._safe_int(cells[1].text),
                    'horse_number': self._safe_int(cells[2].text),
                    'horse_id': self._extract_horse_id(cells[3]),
                    'horse_name': cells[3].text.strip(),
                    'horse_age': self._extract_horse_age(cells[4].text),
                    'horse_sex': self._extract_horse_sex(cells[4].text),
                    'carried_weight': self._safe_float(cells[5].text),
                    'jockey_id': self._extract_jockey_id(cells[6]),
                    'jockey_name': cells[6].text.strip(),
                    'time_seconds': self._parse_time(cells[7].text),
                    'margin': cells[8].text.strip(),
                    'popularity': self._safe_int(cells[9].text),
                    'odds': self._safe_float(cells[10].text) if len(cells) > 10 else None,
                    'final_600m_time': self._safe_float(cells[11].text) if len(cells) > 11 else None,
                    'corner_positions': cells[12].text.strip() if len(cells) > 12 else None,
                    'horse_weight': self._extract_horse_weight(cells[13].text) if len(cells) > 13 else None,
                    'weight_change': self._extract_weight_change(cells[13].text) if len(cells) > 13 else None,
                }
                results.append(result)
            except Exception as e:
                logger.error(f"Error parsing result row: {e}")
                continue
        
        return results
    
    def scrape_horse_info(self, horse_id: str) -> Optional[Dict]:
        """馬情報を取得"""
        url = f"{self.config.BASE_URL}/horse/{horse_id}/"
        soup = self._get_page(url)
        
        if not soup:
            return None
        
        try:
            profile_table = soup.find('table', class_='db_prof_table')
            if not profile_table:
                return None
            
            horse_data = {
                'horse_id': horse_id,
                'name': self._extract_text(soup, '.horse_title h1'),
                'birth_date': self._extract_birth_date(profile_table),
                'sex': self._extract_profile_value(profile_table, '性別'),
                'coat_color': self._extract_profile_value(profile_table, '毛色'),
                'father_id': self._extract_parent_id(profile_table, '父'),
                'mother_id': self._extract_parent_id(profile_table, '母'),
                'trainer_id': self._extract_trainer_id(profile_table),
                'owner_name': self._extract_profile_value(profile_table, '馬主'),
                'breeder_name': self._extract_profile_value(profile_table, '生産者'),
            }
            
            return horse_data
            
        except Exception as e:
            logger.error(f"Error parsing horse {horse_id}: {e}")
            return None
    
    # ヘルパーメソッド群
    def _extract_text(self, soup: BeautifulSoup, selector: str) -> Optional[str]:
        """セレクタからテキストを抽出"""
        element = soup.select_one(selector)
        return element.text.strip() if element else None
    
    def _safe_int(self, text: str) -> Optional[int]:
        """安全に整数に変換"""
        try:
            return int(re.sub(r'[^\d]', '', text))
        except:
            return None
    
    def _safe_float(self, text: str) -> Optional[float]:
        """安全に浮動小数点数に変換"""
        try:
            return float(re.sub(r'[^\d.]', '', text))
        except:
            return None
    
    def _parse_time(self, time_str: str) -> Optional[float]:
        """タイムを秒に変換"""
        if not time_str or time_str == '---':
            return None
        
        match = re.match(r'(\d+):(\d+)\.(\d+)', time_str.strip())
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            fraction = int(match.group(3))
            return minutes * 60 + seconds + fraction / 10
        return None
    
    def _extract_horse_id(self, cell) -> Optional[str]:
        """馬IDを抽出"""
        link = cell.find('a', href=re.compile(r'/horse/'))
        if link:
            match = re.search(r'/horse/(\w+)/', link['href'])
            return match.group(1) if match else None
        return None
    
    def _extract_jockey_id(self, cell) -> Optional[str]:
        """騎手IDを抽出"""
        link = cell.find('a', href=re.compile(r'/jockey/'))
        if link:
            match = re.search(r'/jockey/(\w+)/', link['href'])
            return match.group(1) if match else None
        return None
    
    def close(self):
        """リソースをクリーンアップ"""
        if self.driver:
            self.driver.quit()
        self.session.close()