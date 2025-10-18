from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Import crawler after app to avoid circular imports
try:
    from crawler import SearchEngineCrawler
    crawler = SearchEngineCrawler()
    crawler_available = True
except ImportError as e:
    print(f"Crawler import error: {e}")
    crawler_available = False
except Exception as e:
    print(f"Crawler initialization error: {e}")
    crawler_available = False

@app.route('/')
def home():
    return jsonify({
        "message": "Assassin Search Engine API is running!",
        "crawler_available": crawler_available,
        "endpoints": {
            "search": "/api/search?q=your_query"
        }
    })

@app.route('/api/search')
def search():
    if not crawler_available:
        return jsonify({'error': 'Search crawler is not available'}), 500
        
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        results = crawler.unified_search(query, num_results=10)
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "crawler": crawler_available})

if __name__ == '__main__':
    print("Starting Assassin Search Engine...")
    print(f"Crawler available: {crawler_available}")
    app.run(host='0.0.0.0', port=5000, debug=True)
