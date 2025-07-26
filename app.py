from flask import Flask, render_template, request, jsonify
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)

class WebsiteScraper:
    def __init__(self):
        self.scraped_content = ""
        self.url = ""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def clean_text(self, text):
        """Clean and normalize scraped text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.,!?;:()\-]', '', text)
        return text.strip()
    
    def scrape_website(self, url):
        """Scrape website content with improved text extraction"""
        try:
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post', '.entry', 'section', 'div'
            ]
            
            extracted_text = []
            
            # Try to find main content
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if len(text) > 100:  # Only include substantial text blocks
                        extracted_text.append(text)
                
                if extracted_text:
                    break
            
            # If no main content found, get all text
            if not extracted_text:
                extracted_text = [soup.get_text(separator=' ', strip=True)]
            
            # Combine and clean text
            combined_text = ' '.join(extracted_text)
            cleaned_text = self.clean_text(combined_text)
            
            # Limit text length to avoid token limits
            max_length = 50000  # Adjust based on your needs
            if len(cleaned_text) > max_length:
                cleaned_text = cleaned_text[:max_length] + "..."
            
            self.scraped_content = cleaned_text
            self.url = url
            
            return {
                'success': True,
                'content_length': len(cleaned_text),
                'url': url,
                'preview': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text
            }
            
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f'Failed to fetch URL: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': f'Error processing content: {str(e)}'}
    
    def answer_question(self, question):
        """Answer questions about the scraped content using Gemini"""
        if not self.scraped_content:
            return {'success': False, 'error': 'No website content available. Please scrape a website first.'}
        
        try:
            # Create a prompt for the LLM
            prompt = f"""
            Based on the following website content from {self.url}, please answer the user's question accurately and concisely.
            
            Website Content:
            {self.scraped_content[:8000]}  # Limit content to avoid token limits
            
            User Question: {question}
            
            Please provide a helpful and accurate answer based only on the information available in the website content. If the information is not available in the content, please state that clearly.
            """
            
            response = self.llm.invoke(prompt)
            
            return {
                'success': True,
                'answer': response.content,
                'source_url': self.url
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error generating answer: {str(e)}'}

# Global scraper instance
scraper = WebsiteScraper()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape_url():
    """Endpoint to scrape a website"""
    data = request.get_json()
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'success': False, 'error': 'URL is required'})
    
    result = scraper.scrape_website(url)
    return jsonify(result)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Endpoint to ask questions about the scraped content"""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'success': False, 'error': 'Question is required'})
    
    result = scraper.answer_question(question)
    return jsonify(result)

@app.route('/status')
def get_status():
    """Get current scraper status"""
    return jsonify({
        'has_content': bool(scraper.scraped_content),
        'url': scraper.url,
        'content_length': len(scraper.scraped_content) if scraper.scraped_content else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)