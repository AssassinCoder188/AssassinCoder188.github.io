import requests
import time
import json
from googleapiclient.discovery import build

class GitHubCodespaceCrawler:
    def __init__(self, api_key, search_engine_id):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.service = build("customsearch", "v1", developerKey=api_key)
        
    def search(self, query, num_results=10):
        print(f"🔍 Searching from GitHub Codespace: '{query}'")
        
        try:
            result = self.service.cse().list(
                q=query,
                cx=self.search_engine_id,
                num=num_results
            ).execute()
            
            if 'items' in result:
                print(f"✅ Found {len(result['items'])} results")
                
                # Save results to file (or your database)
                with open('search_results.json', 'w') as f:
                    json.dump(result['items'], f, indent=2)
                
                for i, item in enumerate(result['items'], 1):
                    print(f"{i}. {item['title']}")
                    print(f"   {item['link']}")
                
                return result['items']
            else:
                print("❌ No results found")
                return []
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

def main():
    API_KEY = "AIzaSyCaeTFYYy3KrXI7w5ocYqXcIEd9BQ2-OQ8"
    SEARCH_ENGINE_ID = "7477c71db93fb4b28"
    
    crawler = GitHubCodespaceCrawler(API_KEY, SEARCH_ENGINE_ID)
    
    queries = [
        "web scraping tutorial",
        "Python requests library", 
        "AI developments 2024"
    ]
    
    for query in queries:
        print(f"\n🎯 Query: {query}")
        results = crawler.search(query, 5)
        time.sleep(2)  # Be polite to API

if __name__ == "__main__":
    main()
