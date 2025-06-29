import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

def load_env_file(env_path: str = ".env") -> None:
    env_file = Path(env_path)
    if not env_file.exists():
        return
    
    with open(env_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if ((value.startswith('"') and value.endswith('"')) or 
                        (value.startswith("'") and value.endswith("'"))):
                        value = value[1:-1]
                    
                    os.environ[key] = value
                except ValueError:
                    print(f"Warning: Skipping malformed line {line_num}: {line}")


@dataclass
class AppSettings:    
    openai_api_key: str = ""
    google_search_api_key: str = ""
    google_search_engine_id: str = ""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4000
    temperature: float = 0.3
    top_k_results: int = 5
    openai_model: str = "gpt-4.1-mini-2025-04-14"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    data_directory: str = "./data"
    documents_directory: str = "./documents"
    vector_db_path: str = "./data/vector_db"
    
    log_level: str = "INFO"
    log_file: str = "./data/logs/app.log"
    
    def __post_init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY", self.google_search_api_key)
        self.google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", self.google_search_engine_id)
        
        self.chunk_size = int(os.getenv("CHUNK_SIZE", self.chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", self.chunk_overlap))
        self.max_tokens = int(os.getenv("MAX_TOKENS", self.max_tokens))
        self.temperature = float(os.getenv("TEMPERATURE", self.temperature))
        self.top_k_results = int(os.getenv("TOP_K_RESULTS", self.top_k_results))
        
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        
        self.data_directory = os.getenv("DATA_DIRECTORY", self.data_directory)
        self.documents_directory = os.getenv("DOCUMENTS_DIRECTORY", self.documents_directory)
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", self.vector_db_path)
        
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.log_file = os.getenv("LOG_FILE", self.log_file)
        
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)
        Path(self.documents_directory).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in environment or .env file"
            )