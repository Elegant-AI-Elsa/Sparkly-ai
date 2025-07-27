from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import re
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class SparklyAIAssistant:
    def __init__(self):
        self.scraped_content = ""
        self.url = ""
        self.vector_store = None
        self.qa_chain = None
        self.has_trained_content = False
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Memory for conversation
        # Simple conversation history instead of deprecated memory
        self.conversation_history = []
        
    
    def clean_text(self, text):
        """Clean and normalize scraped text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.,!?;:()\-\'\"@#$%&*+=<>/\\|`~\[\]{}]', '', text)
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
            
            logger.info(f"Scraping URL: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                script.decompose()
            
            # Extract text from main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post', '.entry', 'section', '.main-content',
                '[role="main"]', '.article-content', '.page-content'
            ]
            
            extracted_text = []
            
            # Try to find main content
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if len(text) > 200:  # Only include substantial text blocks
                        extracted_text.append(text)
                
                if extracted_text:
                    logger.info(f"Found content using selector: {selector}")
                    break
            
            # If no main content found, get all text from body
            if not extracted_text:
                body = soup.find('body')
                if body:
                    extracted_text = [body.get_text(separator=' ', strip=True)]
                else:
                    extracted_text = [soup.get_text(separator=' ', strip=True)]
            
            # Combine and clean text
            combined_text = ' '.join(extracted_text)
            cleaned_text = self.clean_text(combined_text)
            
            # Remove very short lines and duplicates
            lines = cleaned_text.split('.')
            meaningful_lines = []
            seen_lines = set()
            
            for line in lines:
                line = line.strip()
                if len(line) > 20 and line not in seen_lines:
                    meaningful_lines.append(line)
                    seen_lines.add(line)
            
            final_text = '. '.join(meaningful_lines)
            
            # Limit text length to avoid token limits
            max_length = 100000  # Increased limit for better content coverage
            if len(final_text) > max_length:
                final_text = final_text[:max_length] + "..."
            
            logger.info(f"Successfully scraped {len(final_text)} characters from {url}")
            
            self.scraped_content = final_text
            self.url = url
            
            return {
                'success': True,
                'content_length': len(final_text),
                'url': url,
                'preview': final_text[:500] + "..." if len(final_text) > 500 else final_text
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {str(e)}")
            return {'success': False, 'error': f'Failed to fetch URL: {str(e)}'}
        except Exception as e:
            logger.error(f"Processing error for {url}: {str(e)}")
            return {'success': False, 'error': f'Error processing content: {str(e)}'}
    
    def create_vector_store(self):
        """Create vector store from scraped content"""
        try:
            if not self.scraped_content:
                return {'success': False, 'error': 'No content to vectorize'}
            
            logger.info("Creating document chunks...")
            
            # Create documents
            documents = [Document(
                page_content=self.scraped_content,
                metadata={'source': self.url, 'type': 'website_content'}
            )]
            
            # Split documents into chunks
            text_chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(text_chunks)} text chunks")
            
            # Create vector store using in-memory Qdrant
            self.vector_store = Qdrant.from_documents(
                text_chunks,
                self.embeddings,
                location=":memory:",  # In-memory vector store
                collection_name="sparkly_ai_docs"
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                return_source_documents=True
            )
            
            self.has_trained_content = True
            logger.info("Vector store created successfully")
            
            return {'success': True, 'chunks_created': len(text_chunks)}
            
        except Exception as e:
            logger.error(f"Vector store creation error: {str(e)}")
            return {'success': False, 'error': f'Error creating vector store: {str(e)}'}
    
    def train_on_website(self, url):
        """Complete training pipeline: scrape + vectorize"""
        # First scrape the website
        scrape_result = self.scrape_website(url)
        
        if not scrape_result['success']:
            return scrape_result
        
        # Then create vector store
        vector_result = self.create_vector_store()
        
        if not vector_result['success']:
            return vector_result
        
        return {
            'success': True,
            'url': self.url,
            'content_length': len(self.scraped_content),
            'chunks_created': vector_result['chunks_created'],
            'message': f'Successfully trained on {self.url}'
        }
    
    def answer_question(self, question, has_trained_content=False):
        """Answer questions using either trained content or general knowledge"""
        try:
            if has_trained_content and self.has_trained_content and self.qa_chain:
                # Use RAG with trained content
                logger.info(f"Answering question using trained content: {question}")
                
                result = self.qa_chain({"query": question})
                answer = result['result']
                
                # Add context about the source
                if self.url:
                    answer += f"\n\n*This answer is based on content from: {self.url}*"
                
                return {
                    'success': True,
                    'answer': answer,
                    'source_url': self.url,
                    'used_trained_content': True
                }
            
            else:
                # General conversation without trained content
                logger.info(f"Thinking: {question}")
                
                general_prompt = f"""
                You are Sparkly-AI-Assistance, a helpful and friendly AI assistant. 
                Answer the following question in a conversational and helpful manner.
                Be informative but keep your response concise and engaging.
                
                Question: {question}
                
                Provide a helpful response:
                """
                
                response = self.llm.invoke(general_prompt)
                
                return {
                    'success': True,
                    'answer': response.content,
                    'source_url': None,
                    'used_trained_content': False
                }
                
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {'success': False, 'error': f'Error generating answer: {str(e)}'}
    
    def reset_training(self):
        """Reset all trained content"""
        self.scraped_content = ""
        self.url = ""
        self.vector_store = None
        self.qa_chain = None
        self.has_trained_content = False
        self.conversation_history = []
        logger.info("Training data reset")
        
# Global assistant instance
assistant = SparklyAIAssistant()

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/train', methods=['POST'])
def train_website():
    """Endpoint to train on a website"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'})
        
        logger.info(f"Training request for URL: {url}")
        result = assistant.train_on_website(url)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Training endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/ask', methods=['POST'])
def ask_question():
    """Endpoint to ask questions"""
    try:
        data = request.get_json()

        # Support both "question" and "query" keys from frontend
        question = data.get('question') or data.get('query') or ''
        question = question.strip()
        has_trained_content = data.get('has_trained_content', False)

        if not question:
            return jsonify({'success': False, 'error': 'Question is required'})
        
        logger.info(f"Question received: {question} (trained_content: {has_trained_content})")
        result = assistant.answer_question(question, has_trained_content)
        
        return jsonify(result)

        
    except Exception as e:
        logger.error(f"Ask endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/status')
def get_status():
    """Get current assistant status"""
    return jsonify({
        'has_content': assistant.has_trained_content,
        'url': assistant.url,
        'content_length': len(assistant.scraped_content) if assistant.scraped_content else 0,
        'vector_store_ready': assistant.vector_store is not None
    })

@app.route('/reset', methods=['POST'])
def reset_training():
    """Reset all training data"""
    try:
        assistant.reset_training()
        return jsonify({'success': True, 'message': 'Training data reset successfully'})
    except Exception as e:
        logger.error(f"Reset endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check for required environment variables
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable is required!")
        print("Error: Please set your GEMINI_API_KEY in the .env file")
        exit(1)
    
    logger.info("Starting Sparkly-AI-Assistance server...")
    app.run(debug=True, host='0.0.0.0', port=5000)