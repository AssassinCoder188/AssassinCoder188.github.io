import requests
import json
import time
from urllib.parse import urlencode
import logging

class SearchEngineCrawler:
    def __init__(self, google_api_key=None, google_search_engine_id=None):
        self.google_api_key = AIzaSyCaeTFYYy3KrXI7w5ocYqXcIEd9BQ2-OQ8
        self.google_search_engine_id = 24cd31d13bdd34858
        self.session = requests.Session()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def search_google_custom(self, query, num_results=10):
        """Search using Google Custom Search JSON API"""
        if not self.google_api_key or not self.google_search_engine_id:
            self.logger.error("Google API key or Search Engine ID not provided")
            return []
        
        results = []
        try:
            # Google Custom Search API endpoint
            url = "https://www.googleapis.com/customsearch/v1"
            
            params = {
                'key': self.google_api_key,
                'cx': self.google_search_engine_id,
                'q': query,
                'num': min(num_results, 10)  # Google allows max 10 results per request
            }
            
            response = self.session.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title'),
                    'link': item.get('link'),
                    'snippet': item.get('snippet'),
                    'source': 'google'
                })
                
            self.logger.info(f"Google search returned {len(results)} results")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Google search error: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Google JSON parse error: {e}")
            
        return results

    def search_bing(self, query, num_results=10):
        """Search Bing web results (using their unofficial API)"""
        results = []
        try:
            url = "https://www.bing.com/search"
            params = {
                'q': query,
                'count': num_results
            }
            
            response = self.session.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            # Note: This is a simplified parser. In production, you'd want a more robust HTML parser
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result elements (Bing's structure)
            for result in soup.find_all('li', class_='b_algo')[:num_results]:
                title_elem = result.find('h2')
                link_elem = result.find('a')
                snippet_elem = result.find('p')
                
                if title_elem and link_elem:
                    results.append({
                        'title': title_elem.get_text().strip(),
                        'link': link_elem.get('href'),
                        'snippet': snippet_elem.get_text().strip() if snippet_elem else '',
                        'source': 'bing'
                    })
            
            self.logger.info(f"Bing search returned {len(results)} results")
            
        except Exception as e:
            self.logger.error(f"Bing search error: {e}")
            
        return results

    def search_duckduckgo(self, query, num_results=10):
        """Search DuckDuckGo using their unofficial API"""
        results = []
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = self.session.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            # DuckDuckGo returns RelatedTopics which can include results
            for topic in data.get('RelatedTopics', [])[:num_results]:
                if 'FirstURL' in topic and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                        'link': topic.get('FirstURL'),
                        'snippet': topic.get('Text', ''),
                        'source': 'duckduckgo'
                    })
            
            self.logger.info(f"DuckDuckGo search returned {len(results)} results")
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            
        return results

    def unified_search(self, query, num_results=10, engines=None):
        """Perform unified search across multiple engines"""
        if engines is None:
            engines = ['google', 'bing', 'duckduckgo']
        
        all_results = []
        
        for engine in engines:
            self.logger.info(f"Searching {engine} for: {query}")
            
            if engine == 'google' and self.google_api_key and self.google_search_engine_id:
                results = self.search_google_custom(query, num_results)
            elif engine == 'bing':
                results = self.search_bing(query, num_results)
            elif engine == 'duckduckgo':
                results = self.search_duckduckgo(query, num_results)
            else:
                continue
                
            all_results.extend(results)
            
            # Be polite - add delay between requests to different engines
            time.sleep(1)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            if result['link'] not in seen_urls:
                seen_urls.add(result['link'])
                unique_results.append(result)
        
        self.logger.info(f"Unified search returned {len(unique_results)} unique results")
        return unique_results

    def save_results(self, results, filename='search_results.json'):
        """Save search results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

# Usage example
if __name__ == "__main__":
    # Initialize the crawler with your Google credentials
    crawler = SearchEngineCrawler(
        google_api_key="YOUR_GOOGLE_API_KEY",
        google_search_engine_id="YOUR_SEARCH_ENGINE_ID"
    )
    
    # Perform a unified search
    query = "Python web development"
    results = crawler.unified_search(query, num_results=5)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['source'].upper()}] {result['title']}")
        print(f"   URL: {result['link']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
        print()
    
    # Save results to file
    crawler.save_results(results)
