import openai
from typing import List

from .models import QueryRequest, TaskType
from .config import AppSettings
from .logging_config import get_logger

logger = get_logger(__name__)


class LLMProvider:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        
        self.model = self._get_available_model(settings.openai_model)
        logger.info(f"[OK] OpenAI provider initialized with model: {self.model}")
    
    def _get_available_model(self, preferred_model: str) -> str:
        model_options = [
            preferred_model,
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4",
            "gpt-4.1-mini-2025-04-14"
        ]
        
        for model in model_options:
            try:
                test_response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                logger.info(f"[OK] Model {model} is available and working")
                return model
            except Exception as e:
                if "model_not_found" in str(e) or "does not exist" in str(e):
                    logger.warning(f"[WARN] Model {model} not available: {e}")
                    continue
                else:
                    logger.error(f"[ERROR] Error testing model {model}: {e}")
                    continue
        
        raise Exception("No OpenAI models are available for your account")
    
    async def generate_response(self, request: QueryRequest, context: str) -> str:
        try:
            system_prompt = self._get_system_prompt(request.task_type)
            user_prompt = f"""Context:\n{context}\n\nQuestion: {request.question}\n\nPlease provide a comprehensive answer based on the context above."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.settings.temperature,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"I encountered an error generating the response: {str(e)}"
    
    def _get_system_prompt(self, task_type: TaskType = None) -> str:
        base_prompt = "You are an expert real estate agent and your task is documents analysis and providing insights. Provide clear, accurate answers based on the provided context. Always cite your sources and be specific about what information comes from which document."
        
        if task_type == TaskType.SUMMARIZATION:
            return base_prompt + " Focus on giving concise, comprehensive summaries."
        elif task_type == TaskType.WEB_SEARCH:
            return base_prompt + " Focus on current information and web sources."
        elif task_type == TaskType.ANALYSIS:
            return base_prompt + " Provide detailed analysis and insights."
        else:
            return base_prompt


class TaskClassifier:
    def classify(self, query: str) -> TaskType:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            return TaskType.SUMMARIZATION
        elif any(word in query_lower for word in ['current', 'latest', 'news', 'market']):
            return TaskType.WEB_SEARCH
        elif any(word in query_lower for word in ['contract', 'document', 'clause', 'agreement']):
            return TaskType.DOCUMENT_QUERY
        else:
            return TaskType.HYBRID_SEARCH