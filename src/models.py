import time
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

class TaskType(Enum):
    DOCUMENT_QUERY = "document_query"
    WEB_SEARCH = "web_search"
    HYBRID_SEARCH = "hybrid_search"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"

class SourceType(Enum):
    DOCUMENT = "document"
    WEB = "web"

@dataclass
class Document:
    id: str
    title: str
    content: str
    markdown_content: str
    metadata: Dict[str, Any]
    processing_time: float
    created_at: float = field(default_factory=time.time)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())
    
    @property
    def file_name(self) -> str:
        return Path(self.metadata.get("source_path", "")).name


@dataclass
class SearchResult:
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]
    source_type: SourceType
    rank: int = 0
    
    @property
    def title(self) -> str:
        """Get the title from metadata."""
        return self.metadata.get('title', self.metadata.get('document_title', 'Unknown'))


@dataclass
class QueryRequest:
    question: str
    task_type: Optional[TaskType] = None
    max_results: int = 5
    include_web: bool = True
    include_documents: bool = True


@dataclass
class QueryResponse:
    answer: str
    sources: List[SearchResult]
    task_type: TaskType
    processing_time: float
    context_used: int
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    content: str
    start_pos: int
    end_pos: int