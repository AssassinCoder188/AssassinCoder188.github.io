import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
import urllib.robotparser
from urllib.parse import urljoin, urlparse
import time
import json
import hashlib
import zlib
import pickle
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional
import logging
import re
import backoff
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
import dill
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

@dataclass
class CrawlConfig:
    max_pages: int = 10000
    max_depth: int = 6
    politeness_delay: float = 0.5
    timeout: int = 30
    max_retries: int = 3
    user_agent: str = "AdvancedCrawler/2.0"
    max_connections: int = 100
    respect_robots: bool = True
    allowed_domains: List[str] = None
    seed_urls: List[str] = None
    output_file: str = "crawled_data.jsonl"
    stats_file: str = "crawl_stats.json"

@dataclass
class PageData:
    url: str
    title: str
    content: str
    meta_description: str
    headers: Dict
    status_code: int
    load_time: float
    timestamp: str
    depth: int
    links: List[str]
    images: List[str]
    language: str = "en"
    content_type: str = "text/html"
    word_count: int = 0
    text_quality: float = 0.0

class AdvancedWebCrawler:
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.visited_urls: Set[str] = set()
        self.url_queue: List[tuple] = []  # (url, depth)
        self.robots_parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.crawl_stats = {
            'pages_crawled': 0,
            'pages_failed': 0,
            'total_bytes': 0,
            'start_time': None,
            'domains_crawled': set()
        }
        
        # Advanced features
        self.content_quality_threshold = 0.3
        self.domain_delay_tracker = {}
        self.html_graph = nx.DiGraph()
        
        self.setup_logging()
        self.setup_signal_handlers()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler('crawler.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_signal_handlers(self):
        def signal_handler(sig, frame):
            self.logger.info("Received interrupt signal. Saving state...")
            self.save_state()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
    
    async def init_session(self):
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=20,
            enable_cleanup_closed=True,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': self.config.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )
    
    def get_domain(self, url: str) -> str:
        return urlparse(url).netloc
    
    async def get_robots_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        domain = self.get_domain(url)
        
        if domain not in self.robots_parsers:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(f"https://{domain}", "/robots.txt")
            
            try:
                async with self.session.get(robots_url) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        rp.parse(robots_content.splitlines())
                    else:
                        rp.allow_all = True
            except Exception as e:
                self.logger.warning(f"Failed to fetch robots.txt for {domain}: {e}")
                rp.allow_all = True
            
            self.robots_parsers[domain] = rp
        
        return self.robots_parsers[domain]
    
    def can_fetch(self, url: str) -> bool:
        if not self.config.respect_robots:
            return True
        
        try:
            parser = self.robots_parsers.get(self.get_domain(url))
            if parser:
                return parser.can_fetch(self.config.user_agent, url)
            return True
        except:
            return True
    
    def is_allowed_domain(self, url: str) -> bool:
        if not self.config.allowed_domains:
            return True
        
        domain = self.get_domain(url)
        return any(allowed in domain for allowed in self.config.allowed_domains)
    
    def normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        
        # Remove fragments and normalize
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        
        return normalized.rstrip('/')
    
    def should_crawl_url(self, url: str, current_depth: int) -> bool:
        if current_depth >= self.config.max_depth:
            return False
        
        if not url.startswith(('http://', 'https://')):
            return False
        
        if not self.is_allowed_domain(url):
            return False
        
        normalized = self.normalize_url(url)
        
        if normalized in self.visited_urls:
            return False
        
        # Check file extensions to avoid non-HTML content
        excluded_extensions = {
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.zip', '.tar', '.gz', '.exe', '.dmg', '.png', '.jpg', '.jpeg',
            '.gif', '.bmp', '.svg', '.mp4', '.avi', '.mov', '.mp3', '.wav'
        }
        
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False
        
        return self.can_fetch(url)
    
    def calculate_text_quality(self, text: str) -> float:
        """Calculate text quality score based on various metrics"""
        if not text or len(text) < 50:
            return 0.0
        
        words = text.split()
        word_count = len(words)
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / word_count
        
        # Calculate sentence complexity (rough estimate)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = word_count / max(len(sentences), 1)
        
        # Calculate unique word ratio
        unique_ratio = len(set(words)) / word_count
        
        # Penalize very short or very long average word lengths
        word_length_score = 1.0 - abs(avg_word_length - 5.5) / 5.5
        
        # Penalize very short or very long sentences
        sentence_length_score = 1.0 - min(abs(avg_sentence_length - 15) / 15, 1.0)
        
        # Combine scores
        quality_score = (
            word_length_score * 0.3 +
            sentence_length_score * 0.3 +
            unique_ratio * 0.2 +
            min(word_count / 1000, 1.0) * 0.2
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract rich metadata from page"""
        metadata = {
            'authors': [],
            'keywords': [],
            'publish_date': None,
            'categories': [],
            'reading_time': 0
        }
        
        # Extract authors
        author_selectors = [
            '[name="author"]',
            '[rel="author"]',
            '.author',
            '.byline',
            '[property="article:author"]'
        ]
        
        for selector in author_selectors:
            elements = soup.select(selector)
            for elem in elements:
                author = elem.get('content') or elem.get_text(strip=True)
                if author and author not in metadata['authors']:
                    metadata['authors'].append(author)
        
        # Extract keywords
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords and keywords.get('content'):
            metadata['keywords'] = [k.strip() for k in keywords['content'].split(',')]
        
        # Extract publish date
        date_selectors = [
            '[property="article:published_time"]',
            '[name="publish_date"]',
            '.publish-date',
            '.post-date',
            'time[datetime]'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('datetime') or element.get('content') or element.get_text(strip=True)
                if date_str:
                    metadata['publish_date'] = date_str
                    break
        
        # Estimate reading time (assuming 200 words per minute)
        text_content = soup.get_text()
        word_count = len(text_content.split())
        metadata['reading_time'] = max(1, round(word_count / 200))
        
        return metadata
    
    @backoff.on_exception(backoff.expo, 
                         (aiohttp.ClientError, asyncio.TimeoutError),
                         max_tries=3)
    async def fetch_page(self, url: str) -> Dict:
        start_time = time.time()
        
        try:
            async with self.session.get(url) as response:
                content = await response.read()
                load_time = time.time() - start_time
                
                # Detect encoding
                encoding = response.charset or 'utf-8'
                
                try:
                    html = content.decode(encoding, errors='ignore')
                except UnicodeDecodeError:
                    html = content.decode('utf-8', errors='ignore')
                
                return {
                    'url': url,
                    'html': html,
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'load_time': load_time,
                    'content_length': len(content),
                    'success': True
                }
                
        except Exception as e:
            return {
                'url': url,
                'html': '',
                'status_code': 0,
                'headers': {},
                'load_time': time.time() - start_time,
                'content_length': 0,
                'success': False,
                'error': str(e)
            }
    
    def parse_page(self, html: str, url: str) -> PageData:
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract title
        title = soup.title.string if soup.title else ""
        
        # Extract meta description
        meta_description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            meta_description = meta_desc['content']
        
        # Extract main content (try to get article content)
        content_selectors = [
            'article',
            '.content',
            '.main-content',
            '.post-content',
            '[role="main"]',
            'body'
        ]
        
        content_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content_text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                if len(content_text) > 200:  # Found substantial content
                    break
        
        if not content_text:
            content_text = soup.get_text(separator=' ', strip=True)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            normalized_url = self.normalize_url(absolute_url)
            if normalized_url.startswith(('http://', 'https://')):
                links.append(normalized_url)
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            absolute_src = urljoin(url, src)
            images.append(absolute_src)
        
        # Calculate text quality
        text_quality = self.calculate_text_quality(content_text)
        
        return PageData(
            url=url,
            title=title.strip(),
            content=content_text.strip(),
            meta_description=meta_description.strip(),
            headers={},  # Will be filled later
            status_code=200,  # Will be filled later
            load_time=0,  # Will be filled later
            timestamp=datetime.utcnow().isoformat(),
            depth=0,  # Will be filled later
            links=links,
            images=images,
            word_count=len(content_text.split()),
            text_quality=text_quality
        )
    
    async def process_url(self, url: str, depth: int) -> List[str]:
        """Process a single URL and return new URLs to crawl"""
        
        # Respect politeness delay per domain
        domain = self.get_domain(url)
        last_access = self.domain_delay_tracker.get(domain, 0)
        current_time = time.time()
        
        if current_time - last_access < self.config.politeness_delay:
            await asyncio.sleep(self.config.politeness_delay)
        
        self.domain_delay_tracker[domain] = time.time()
        
        # Fetch the page
        fetch_result = await self.fetch_page(url)
        
        if not fetch_result['success']:
            self.crawl_stats['pages_failed'] += 1
            self.logger.warning(f"Failed to fetch {url}: {fetch_result.get('error', 'Unknown error')}")
            return []
        
        if fetch_result['status_code'] != 200:
            self.logger.debug(f"Non-200 status for {url}: {fetch_result['status_code']}")
            return []
        
        # Parse the page
        page_data = self.parse_page(fetch_result['html'], url)
        page_data.status_code = fetch_result['status_code']
        page_data.headers = fetch_result['headers']
        page_data.load_time = fetch_result['load_time']
        page_data.depth = depth
        
        # Only save if content meets quality threshold
        if page_data.text_quality >= self.content_quality_threshold:
            await self.save_page_data(page_data)
            self.crawl_stats['pages_crawled'] += 1
            self.crawl_stats['total_bytes'] += fetch_result['content_length']
            self.crawl_stats['domains_crawled'].add(domain)
            
            self.logger.info(
                f"Crawled: {url} | Depth: {depth} | "
                f"Quality: {page_data.text_quality:.2f} | "
                f"Words: {page_data.word_count} | "
                f"Links: {len(page_data.links)}"
            )
        
        # Return new URLs to crawl
        new_urls = []
        for link in page_data.links:
            if self.should_crawl_url(link, depth + 1):
                new_urls.append(link)
                self.visited_urls.add(self.normalize_url(link))
        
        return new_urls
    
    async def save_page_data(self, page_data: PageData):
        """Save page data to output file"""
        async with aiofiles.open(self.config.output_file, 'a', encoding='utf-8') as f:
            data = asdict(page_data)
            await f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def save_state(self):
        """Save crawl state for resumption"""
        state = {
            'visited_urls': list(self.visited_urls),
            'url_queue': self.url_queue,
            'crawl_stats': self.crawl_stats,
            'domain_delay_tracker': self.domain_delay_tracker
        }
        
        with open('crawl_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        # Save statistics
        with open(self.config.stats_file, 'w') as f:
            stats = self.crawl_stats.copy()
            stats['domains_crawled'] = list(stats['domains_crawled'])
            stats['end_time'] = datetime.utcnow().isoformat()
            json.dump(stats, f, indent=2)
        
        self.logger.info("Crawl state saved successfully")
    
    def load_state(self):
        """Load crawl state for resumption"""
        try:
            with open('crawl_state.pkl', 'rb') as f:
                state = pickle.load(f)
            
            self.visited_urls = set(state['visited_urls'])
            self.url_queue = state['url_queue']
            self.crawl_stats = state['crawl_stats']
            self.domain_delay_tracker = state['domain_delay_tracker']
            
            self.logger.info("Crawl state loaded successfully")
        except FileNotFoundError:
            self.logger.info("No previous state found. Starting fresh crawl.")
    
    async def crawl(self):
        """Main crawl method"""
        await self.init_session()
        
        # Initialize queue with seed URLs
        for url in self.config.seed_urls:
            if self.should_crawl_url(url, 0):
                self.url_queue.append((url, 0))
                self.visited_urls.add(self.normalize_url(url))
        
        self.crawl_stats['start_time'] = datetime.utcnow().isoformat()
        
        self.logger.info(f"Starting crawl with {len(self.url_queue)} seed URLs")
        
        # Pre-fetch robots.txt for seed domains
        for url in self.config.seed_urls:
            await self.get_robots_parser(url)
        
        # Main crawl loop
        while self.url_queue and self.crawl_stats['pages_crawled'] < self.config.max_pages:
            current_batch = []
            
            # Take a batch of URLs to process
            while self.url_queue and len(current_batch) < self.config.max_connections:
                url, depth = self.url_queue.pop(0)
                current_batch.append((url, depth))
            
            # Process batch concurrently
            tasks = []
            for url, depth in current_batch:
                task = self.process_url(url, depth)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Add new URLs to queue
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task failed: {result}")
                    continue
                
                for new_url in result:
                    if (new_url, depth + 1) not in self.url_queue:
                        self.url_queue.append((new_url, depth + 1))
            
            # Log progress
            if self.crawl_stats['pages_crawled'] % 100 == 0:
                self.logger.info(
                    f"Progress: {self.crawl_stats['pages_crawled']} pages crawled, "
                    f"{len(self.url_queue)} URLs in queue"
                )
            
            # Save state periodically
            if self.crawl_stats['pages_crawled'] % 500 == 0:
                self.save_state()
        
        await self.session.close()
        self.save_state()
        
        self.logger.info(
            f"Crawl completed! "
            f"Pages: {self.crawl_stats['pages_crawled']}, "
            f"Failed: {self.crawl_stats['pages_failed']}, "
            f"Domains: {len(self.crawl_stats['domains_crawled'])}"
        )

# Usage example
async def main():
    config = CrawlConfig(
        max_pages=1000,
        max_depth=3,
        politeness_delay=1.0,
        timeout=30,
        max_connections=50,
        allowed_domains=['wikipedia.org', 'github.com', 'stackoverflow.com'],
        seed_urls=[
            'https://en.wikipedia.org/wiki/Python_(programming_language)',
            'https://github.com',
            'https://stackoverflow.com',
        ],
        output_file='crawled_data.jsonl',
        stats_file='crawl_stats.json'
    )
    
    crawler = AdvancedWebCrawler(config)
    
    try:
        await crawler.crawl()
    except Exception as e:
        crawler.logger.error(f"Crawl failed: {e}")
        crawler.save_state()

if __name__ == "__main__":
    asyncio.run(main())
