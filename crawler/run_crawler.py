#!/usr/bin/env python3
import asyncio
import yaml
from advanced_crawler import AdvancedWebCrawler, CrawlConfig

def load_config(config_path: str = "config.yaml"):
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return CrawlConfig(
        max_pages=config_data['crawler']['max_pages'],
        max_depth=config_data['crawler']['max_depth'],
        politeness_delay=config_data['crawler']['politeness_delay'],
        timeout=config_data['crawler']['timeout'],
        max_retries=config_data['crawler']['max_retries'],
        max_connections=config_data['crawler']['max_connections'],
        user_agent=config_data['crawler']['user_agent'],
        respect_robots=config_data['crawler']['respect_robots'],
        allowed_domains=config_data['allowed_domains'],
        seed_urls=config_data['seed_urls'],
        output_file=config_data['output']['data_file'],
        stats_file=config_data['output']['stats_file']
    )

async def main():
    config = load_config()
    crawler = AdvancedWebCrawler(config)
    
    print("🚀 Starting Advanced Web Crawler")
    print(f"📊 Target: {config.max_pages} pages")
    print(f"🌐 Seed URLs: {len(config.seed_urls)}")
    print(f"🔧 Connections: {config.max_connections}")
    print("=" * 50)
    
    await crawler.crawl()

if __name__ == "__main__":
    asyncio.run(main())
