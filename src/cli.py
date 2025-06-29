import asyncio
from pathlib import Path

from .config import load_env_file, AppSettings
from .core import AgenticAISystem
from .logging_config import setup_logging


async def main():
    print("RealInsights.ai Support Agent")
    print("=" * 50)
    
    try:
        load_env_file()
        settings = AppSettings()
        setup_logging(settings.log_level, settings.log_file)
        
        if not settings.openai_api_key:
            print("[ERROR] No OpenAI API key found!")
            print("   Create a .env file with: OPENAI_API_KEY=your-key-here")
            print("   Or set environment variable: export OPENAI_API_KEY=your-key")
            return
        
        print(f"[OK] OpenAI API key loaded: {settings.openai_api_key[:8]}...")
        print()
        
        ai_system = AgenticAISystem(settings)
        docs_path = input(f"[INPUT] Documents directory or file [{settings.documents_directory}]: ").strip()
        docs_path = docs_path or settings.documents_directory
        
        print("[PROCESSING] Loading documents...")
        results = await ai_system.load_documents(docs_path)
        
        print("\n[RESULTS] Processing results:")
        for path, result in results.items():
            print(f"   {Path(path).name}: {result}")
        
        status = ai_system.get_status()
        print(f"\n[READY] System ready! Loaded {status['loaded_documents']} documents")
        
        print("\n[CHAT] Ask questions (type 'quit' to exit):")
        
        while True:
            print("\n" + "-" * 40)
            question = input("[Q] Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("[PROCESSING] Analyzing question...")
            response = await ai_system.ask_question(question)
            
            print(f"\n[ANSWER]")
            print(response['answer'])
            
            print(f"\n[TIMING] Processed in {response['processing_time']}")
            
            if response.get('has_sources') and response.get('total_sources', 0) > 0:
                doc_count = len(set(s['title'].split(' - Chunk')[0] for s in response['sources']))
                section_count = response['total_sources']
                
                if section_count > 1:
                    print(f"[SOURCES] Found relevant information in {section_count} sections from {doc_count} document(s)")
                else:
                    print(f"[SOURCES] Found relevant information in the document")
            
            task_type_display = {
                'document_query': 'Document Analysis',
                'web_search': 'Web Search',
                'hybrid_search': 'Document + Web Search',
                'summarization': 'Summary',
                'analysis': 'Analysis'
            }
            print(f"[TASK] {task_type_display.get(response['task_type'], response['task_type'])}")
    
    except KeyboardInterrupt:
        print("\n[EXIT] Goodbye!")
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    asyncio.run(main())
