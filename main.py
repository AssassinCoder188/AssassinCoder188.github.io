import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.database.vector_database import AdvancedVectorDB, Document
from src.crawler.advanced_crawler import AdvancedWebCrawler, CrawlConfig
import asyncio
import numpy as np
from datetime import datetime

def start_real_crawl():
    """Start crawling actual websites"""
    config = CrawlConfig(
        seed_urls=[
            'https://news.ycombinator.com',
            'https://github.com/topics/python',
            'https://stackoverflow.com/questions',
            'https://towardsdatascience.com',
            'https://realpython.com',
            'https://docs.python.org',
            'https://www.w3schools.com/python/',
            'https://www.freecodecamp.org/news/tag/python/'
        ],
        max_pages=1000,
        max_depth=2,
        allowed_domains=[
            'github.com', 'stackoverflow.com', 'realpython.com',
            'towardsdatascience.com', 'news.ycombinator.com',
            'python.org', 'w3schools.com', 'freecodecamp.org'
        ]
    )
    
    crawler = AdvancedWebCrawler(config)
    print("🚀 Starting real web crawl...")
    asyncio.run(crawler.crawl())

def main():
    print("🔍 ADVANCED SEARCH ENGINE - REAL MODE")
    print("=====================================")
    
    # Initialize database
    vector_db = AdvancedVectorDB("./data")
    
    # Option to start crawling
    choice = input("Start real web crawl? (y/n): ")
    if choice.lower() == 'y':
        start_real_crawl()
    
    print("\n✅ System ready!")
    print("🌐 Open index.html in your browser")
    print("📊 Real results from actual websites")

if __name__ == "__main__":
    main()
