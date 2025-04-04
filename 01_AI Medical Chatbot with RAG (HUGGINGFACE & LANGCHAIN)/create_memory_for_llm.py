from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import sys

# Load environment variables and set them explicitly
load_dotenv()

# Set HuggingFace API token in environment
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

class ProcessTracker:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.steps_completed = 0
        
    def update_progress(self, message, count=None):
        current_time = time.time()
        duration = current_time - self.last_time
        total_duration = current_time - self.start_time
        
        count_info = f" ({count} items)" if count is not None else ""
        print(f"\n✓ Step {self.steps_completed + 1}: {message}{count_info}")
        print(f"   Time taken: {self.format_time(duration)}")
        print(f"   Total time: {self.format_time(total_duration)}")
        
        self.steps_completed += 1
        self.last_time = current_time
        
    @staticmethod
    def format_time(seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        elif minutes > 0:
            return f"{int(minutes)}m {seconds:.2f}s"
        else:
            return f"{seconds:.2f}s"

def setup_environment():
    """Setup and verify environment variables and directories"""
    required_vars = {
        'HUGGINGFACEHUB_API_TOKEN': 'HuggingFace API token',
        'MODEL_PATH': 'Model directory path',
        'DATA_DIR': 'Data directory path',
        'VECTOR_STORE_PATH': 'Vector store path'
    }
    
    # Check environment variables
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    # Setup directories
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        dirs = {
            'DATA_DIR': os.path.join(BASE_DIR, os.getenv('DATA_DIR')),
            'MODEL_PATH': os.path.join(BASE_DIR, os.getenv('MODEL_PATH')),
            'VECTOR_STORE_PATH': os.path.join(BASE_DIR, os.getenv('VECTOR_STORE_PATH'))
        }
        
        for name, path in dirs.items():
            os.makedirs(path, exist_ok=True)
            print(f"✓ Created/verified directory: {name} at {path}")
        
        return dirs
    except Exception as e:
        print(f"❌ Error setting up directories: {str(e)}")
        return False

def load_documents(data_path, tracker):
    """Load PDF documents"""
    try:
        loader = DirectoryLoader(
            data_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        tracker.update_progress("Loaded PDF documents", len(documents))
        return documents
    except Exception as e:
        print(f"❌ Error loading documents: {str(e)}")
        return None

def create_text_chunks(documents, tracker):
    """Create text chunks from documents"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', 500)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 50))
        )
        chunks = text_splitter.split_documents(documents)
        tracker.update_progress("Created text chunks", len(chunks))
        return chunks
    except Exception as e:
        print(f"❌ Error creating chunks: {str(e)}")
        return None

def initialize_embedding_model(tracker):
    """Initialize the embedding model"""
    try:
        # Initialize the embedding model without the API token parameter
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=os.getenv('MODEL_PATH')
        )
        
        # Set the token through environment variable
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        
        tracker.update_progress("Initialized embedding model")
        return model
    except Exception as e:
        print(f"❌ Error initializing embedding model: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Verify your HuggingFace API token is correct")
        print("2. Check your internet connection")
        print("3. Make sure you have sufficient disk space")
        print(f"4. Check if the model cache directory exists: {os.getenv('MODEL_PATH')}")
        return None

def create_vector_store(chunks, embedding_model, vector_store_path, tracker):
    """Create and save vector store"""
    try:
        # Timing for embedding creation
        print("\nCreating FAISS database...")
        print(f"Number of chunks to process: {len(chunks)}")
        
        embedding_start = time.time()
        print("Step 1/2: Creating embeddings...")
        db = FAISS.from_documents(chunks, embedding_model)
        embedding_duration = time.time() - embedding_start
        
        # Approximate time per chunk
        time_per_chunk = embedding_duration / len(chunks)
        print(f"\nEmbedding Statistics:")
        print(f"├── Total chunks processed: {len(chunks)}")
        print(f"├── Time per chunk: {time_per_chunk:.2f} seconds")
        print(f"├── Total embedding time: {tracker.format_time(embedding_duration)}")
        
        # Timing for saving
        print("\nStep 2/2: Saving FAISS database...")
        save_start = time.time()
        db.save_local(vector_store_path)
        save_duration = time.time() - save_start
        
        total_duration = time.time() - embedding_start
        
        # Save detailed statistics
        stats_path = os.path.join(os.path.dirname(vector_store_path), "vector_store_stats.txt")
        with open(stats_path, "w") as f:
            f.write("=== FAISS Vector Store Creation Statistics ===\n")
            f.write(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total chunks processed: {len(chunks)}\n")
            f.write(f"Average time per chunk: {time_per_chunk:.2f} seconds\n")
            f.write(f"Embedding creation time: {tracker.format_time(embedding_duration)}\n")
            f.write(f"Database saving time: {tracker.format_time(save_duration)}\n")
            f.write(f"Total process time: {tracker.format_time(total_duration)}\n")
            f.write("\nSystem Information:\n")
            f.write(f"Python version: {sys.version}\n")
            if hasattr(os, 'cpu_count'):
                f.write(f"CPU cores: {os.cpu_count()}\n")
        
        tracker.update_progress("Created and saved vector store", len(chunks))
        
        print(f"\nDetailed timing breakdown:")
        print(f"├── Embedding creation: {tracker.format_time(embedding_duration)}")
        print(f"├── Database saving: {tracker.format_time(save_duration)}")
        print(f"└── Total process: {tracker.format_time(total_duration)}")
        
        return db
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        return None

def main():
    print("\n=== Starting Medical Chatbot Setup ===\n")
    
    # Initialize progress tracker
    tracker = ProcessTracker()
    
    # Step 1: Setup environment
    print("Setting up environment...")
    dirs = setup_environment()
    if not dirs:
        print("❌ Environment setup failed")
        return
    tracker.update_progress("Environment setup completed")
    
    # Step 2: Load documents
    print("\nLoading documents...")
    documents = load_documents(dirs['DATA_DIR'], tracker)
    if not documents:
        print("❌ Document loading failed")
        return
    
    # Step 3: Create text chunks
    print("\nCreating text chunks...")
    chunks = create_text_chunks(documents, tracker)
    if not chunks:
        print("❌ Chunk creation failed")
        return
    
    # Step 4: Initialize embedding model
    print("\nInitializing embedding model...")
    embedding_model = initialize_embedding_model(tracker)
    if not embedding_model:
        print("❌ Embedding model initialization failed")
        return
    
    # Step 5: Create and save vector store
    print("\nCreating vector store...")
    db = create_vector_store(chunks, embedding_model, dirs['VECTOR_STORE_PATH'], tracker)
    if not db:
        print("❌ Vector store creation failed")
        return
    
    total_time = time.time() - tracker.start_time
    print(f"\n✨ Process completed successfully in {tracker.format_time(total_time)}!")
    print(f"📁 Vector store saved at: {dirs['VECTOR_STORE_PATH']}")

if __name__ == "__main__":
    main()