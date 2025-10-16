import json
import time
from datetime import datetime

def monitor_crawl():
    try:
        with open('crawl_stats.json', 'r') as f:
            stats = json.load(f)
        
        start_time = datetime.fromisoformat(stats['start_time'])
        current_time = datetime.utcnow()
        duration = current_time - start_time
        
        print("📈 Crawl Monitor")
        print("=" * 40)
        print(f"🕒 Running for: {duration}")
        print(f"📄 Pages crawled: {stats['pages_crawled']}")
        print(f"❌ Pages failed: {stats['pages_failed']}")
        print(f"🌐 Domains visited: {len(stats['domains_crawled'])}")
        print(f"💾 Total data: {stats['total_bytes'] / (1024*1024):.2f} MB")
        
        if stats['pages_crawled'] > 0:
            rate = stats['pages_crawled'] / duration.total_seconds() * 60
            print(f"⚡ Crawl rate: {rate:.2f} pages/minute")
        
    except FileNotFoundError:
        print("No crawl stats found. Start the crawler first.")

if __name__ == "__main__":
    monitor_crawl()
