import time
from pathlib import Path
from typing import Dict, Any, Optional

from .config import AppSettings
from .models import Document, QueryRequest, TaskType
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .web_search import WebSearchProvider
from .llm_provider import LLMProvider, TaskClassifier
from .logging_config import get_logger

logger = get_logger(__name__)

class AgenticAISystem:    
    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or AppSettings()
        
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore(self.settings)
        self.web_search = WebSearchProvider(self.settings)
        self.llm_provider = LLMProvider(self.settings)
        self.task_classifier = TaskClassifier()
        
        self.documents: Dict[str, Document] = {}
        logger.info("[OK] Support Agent initialized")
    
    async def load_documents(self, path: str) -> Dict[str, str]:
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise ValueError(f"Path not found: {path}")
        
        file_paths = []
        
        if path_obj.is_file():
            if self.doc_processor.supports_file(str(path_obj)):
                file_paths = [path_obj]
                logger.info(f"[FILE] Processing single file: {path_obj.name}")
            else:
                raise ValueError(f"Unsupported file type: {path_obj.suffix}")
        
        elif path_obj.is_dir():
            for ext in self.doc_processor.supported_extensions:
                file_paths.extend(path_obj.glob(f"*{ext}"))
            
            if not file_paths:
                raise ValueError(f"No supported files found in directory: {path}")
            
            logger.info(f"[DIR] Found {len(file_paths)} files to process")
        
        else:
            raise ValueError(f"Path is neither file nor directory: {path}")
        
        results = {}
        for file_path in file_paths:
            try:
                document = await self.doc_processor.process_document(file_path)
                self.documents[document.id] = document
                await self.vector_store.add_document(document)
                results[str(file_path)] = f"[OK] Success: {document.word_count} words"
            except Exception as e:
                results[str(file_path)] = f"[ERROR] Error: {str(e)}"
                logger.error(f"Failed to process {file_path}: {e}")
        
        return results
    
    async def ask_question(self, question: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            request = QueryRequest(question=question, **kwargs)
            task_type = self.task_classifier.classify(question)
            
            all_results = []
            
            if request.include_documents and task_type != TaskType.WEB_SEARCH:
                doc_results = await self.vector_store.search(question, request.max_results)
                all_results.extend(doc_results)
            
            if request.include_web and task_type in [TaskType.WEB_SEARCH, TaskType.HYBRID_SEARCH]:
                async with self.web_search as search_provider:
                    web_results = await search_provider.search(question, request.max_results)
                    all_results.extend(web_results)
            
            context_parts = []
            for result in all_results[:request.max_results]:
                source_info = f"[{result.source_type.value.upper()}] {result.title}"
                context_parts.append(f"{source_info}\n{result.content}\n")
            
            context = "\n".join(context_parts) if context_parts else "No relevant context found."
            
            answer = await self.llm_provider.generate_response(request, context)
            
            confidence = 85 if all_results else 0
            
            return {
                'answer': answer,
                'task_type': task_type.value,
                'processing_time': f"{time.time() - start_time:.2f}s",
                'confidence': confidence,
                'sources': [
                    {
                        'type': r.source_type.value,
                        'title': r.title,
                        'source': r.source,
                        'score': r.score,
                    }
                    for r in all_results[:5]
                ],
                'has_sources': len(all_results) > 0,
                'total_sources': len(all_results)
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                'answer': f"Error processing question: {str(e)}",
                'task_type': 'error',
                'processing_time': f"{time.time() - start_time:.2f}s",
                'confidence': 0,
                'sources': [],
                'has_sources': False,
                'total_sources': 0
            }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'loaded_documents': len(self.documents),
            'system_ready': len(self.documents) > 0,
            'supported_file_types': sorted(self.doc_processor.supported_extensions)
        }