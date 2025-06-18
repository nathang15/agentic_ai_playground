import asyncio
import numpy as np
import sounddevice as sd
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
from openai import OpenAI
from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import (
    AudioInput, 
    SingleAgentVoiceWorkflow, 
    VoicePipeline, 
    TTSModelSettings, 
    VoicePipelineConfig
)
from agents import Runner, trace
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment():
    print("Loading environment variables")
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(f"{error_msg}")
        print(f"{error_msg}")
        raise ValueError(error_msg)
    
    # Log optional variables
    optional_vars = {
        'VECTOR_STORE_ID': 'File search functionality',
        'SAMPLE_RATE': 'Audio sample rate (default: 16000)',
        'AUDIO_CHANNELS': 'Audio channels (default: 1)',
        'MAX_RECORDING_DURATION': 'Max recording time (default: 30s)',
        'ENABLE_TRACING': 'OpenAI tracing (default: true)',
        'LOG_LEVEL': 'Logging level (default: INFO)'
    }
    
    print("Required environment variables loaded")
    
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"{var}: {value}")
        else:
            print(f"{var} is NOT set: {description}")

class ProductionConfig:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.vector_store_id = os.getenv('VECTOR_STORE_ID')
        if not self.vector_store_id:
            logger.warning("VECTOR_STORE_ID not set. File search will be disabled.")
        
        self.sample_rate = int(os.getenv('SAMPLE_RATE', '16000'))
        self.channels = int(os.getenv('AUDIO_CHANNELS', '1'))
        
        self.max_recording_duration = int(os.getenv('MAX_RECORDING_DURATION', '30'))
        self.enable_tracing = os.getenv('ENABLE_TRACING', 'true').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        logging.getLogger().setLevel(getattr(logging, self.log_level.upper(), logging.INFO))

class VectorStoreManager:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def create_vector_store(self, store_name: str) -> Dict[str, Any]:
        try:
            vector_store = self.client.vector_stores.create(name=store_name)
            details = {
                "id": vector_store.id,
                "name": vector_store.name,
                "created_at": vector_store.created_at,
                "file_count": vector_store.file_counts.completed
            }
            logger.info(f"Vector store created: {details}")
            return details
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def upload_file_to_vector_store(self, file_path: str, vector_store_id: str) -> Dict[str, Any]:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, 'rb') as file:
                file_response = self.client.files.create(
                    file=file, 
                    purpose="assistants"
                )
            
            attach_response = self.client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_response.id
            )
            
            logger.info(f"File uploaded: {file_name}")
            return {"file": file_name, "status": "success", "file_id": file_response.id}
        
        except Exception as e:
            logger.error(f"Error uploading {file_name}: {str(e)}")
            return {"file": file_name, "status": "failed", "error": str(e)}

class ShopDatabase:
    """Mock database"""
    def __init__(self):
        self.accounts = {
            "1234567890": {
                "user_id": "1234567890",
                "name": "Bugs Bunny",
                "account_balance": "£72.50",
                "membership_status": "Gold Executive",
                "email": "bugs@acme.com",
                "last_login": "2025-06-17T10:30:00Z"
            },
            "0987654321": {
                "user_id": "0987654321", 
                "name": "Daffy Duck",
                "account_balance": "£15.25",
                "membership_status": "Silver",
                "email": "daffy@acme.com",
                "last_login": "2025-06-16T14:20:00Z"
            }
        }
    
    def get_account(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.accounts.get(user_id)
    
    def update_account(self, user_id: str, updates: Dict[str, Any]) -> bool:
        if user_id in self.accounts:
            self.accounts[user_id].update(updates)
            return True
        return False

def initialize_components():
    global config, db, vector_manager
    
    config = ProductionConfig()
    db = ShopDatabase()
    vector_manager = VectorStoreManager(config.openai_api_key)
    
    set_default_openai_key(config.openai_api_key)

@function_tool
def get_account_info(user_id: str) -> Dict[str, Any]:
    """Retrieve account information for a given user ID"""
    logger.info(f"Looking up account for user ID: {user_id}")
    
    account = db.get_account(user_id)
    if account:
        safe_account = {
            "user_id": account["user_id"],
            "name": account["name"],
            "account_balance": account["account_balance"],
            "membership_status": account["membership_status"]
        }
        logger.info(f"Account found for {account['name']}")
        return safe_account
    else:
        logger.warning(f"Account not found for user ID: {user_id}")
        return {
            "error": "Account not found",
            "user_id": user_id,
            "message": "Please verify your user ID and try again."
        }

@function_tool
def get_recent_orders(user_id: str, limit: int = 5) -> Dict[str, Any]:
    logger.info(f"Retrieving recent orders for user: {user_id}")
    
    # Mock order data
    mock_orders = [
        {"order_id": "ORD-001", "date": "2025-06-15", "total": "£25.99", "status": "Delivered"},
        {"order_id": "ORD-002", "date": "2025-06-10", "total": "£45.50", "status": "Shipped"},
        {"order_id": "ORD-003", "date": "2025-06-05", "total": "£12.99", "status": "Processing"}
    ]
    
    account = db.get_account(user_id)
    if account:
        return {
            "user_id": user_id,
            "orders": mock_orders[:limit],
            "total_orders": len(mock_orders)
        }
    else:
        return {"error": "User not found", "user_id": user_id}

@function_tool
def search_product_catalog(query: str, max_results: int = 10) -> Dict[str, Any]:
    catalog_path = "product_catalog.json"
    results = []

    if not os.path.exists(catalog_path):
        return {"error": f"Product catalog file '{catalog_path}' not found."}

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    query_lower = query.lower()

    for section, items in catalog.items():
        if query_lower in section.lower():
            results.extend(items)
        else:
            for item in items:
                searchable_text = json.dumps(item).lower()
                if query_lower in searchable_text:
                    results.append(item)

        if len(results) >= max_results:
            break

    if not results:
        return {"message": f"No matching products found for '{query}'."}

    return {"query": query, "matches": results[:max_results]}

class AgentSystem:
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.agents = {}
        self._setup_agents()
    
    def _setup_agents(self):
        """Initialize all agents"""
        logger.info("Setting up agents...")
        
        # Voice optimization prompt
        voice_system_prompt = """
        [Voice Output Guidelines]
        Your responses will be converted to speech. Follow these guidelines:
        1. Use a friendly, conversational tone that sounds natural when spoken
        2. Keep responses concise - 1-3 sentences per main point
        3. Use simple, clear language without technical jargon
        4. Structure information logically with brief pauses between points
        5. Be helpful and engaging while staying professional
        """
        
        # Search Agent
        self.agents['search'] = Agent(
            name="SearchAgent",
            model="gpt-4.1-mini-2025-04-14",
            instructions=(
                "You are a web search specialist. Use the WebSearchTool to find current, "
                "relevant information for user queries. Provide concise, helpful summaries "
                "of your findings with key details highlighted."
            ),
            tools=[WebSearchTool()],
        )
        
        # Knowledge Agent
        if self.config.vector_store_id:
            self.agents['knowledge'] = Agent(
                name="KnowledgeAgent",
                model="gpt-4.1-mini-2025-04-14",
                instructions=(
                    "You are a product specialist for ACME Shop. Use the FileSearchTool to "
                    "find detailed product information from our catalog. Provide accurate, "
                    "helpful product details and recommendations."
                ),
                tools=[FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[self.config.vector_store_id],
                )],
            )
        else:
            # Fallback knowledge agent without file search
            self.agents['knowledge'] = Agent(
                name="KnowledgeAgent",
                model="gpt-4.1-mini-2025-04-14",
                instructions=(
                    "You are a product specialist for ACME Shop. Use the search_product_catalog "
                    "tool to find product information from our catalog file. Provide accurate, "
                    "helpful product details and recommendations."
                ),
                tools=[search_product_catalog],
            )
        
        # Account Agent
        self.agents['account'] = Agent(
            name="AccountAgent",
            model="gpt-4.1-mini-2025-04-14",
            instructions=(
                "You are an account specialist for ACME Shop. Use the available tools to "
                "help customers with account information, order history, and account-related "
                "questions. Be helpful and professional."
            ),
            tools=[get_account_info, get_recent_orders],
        )
        
        # Triage Agent (Text)
        self.agents['triage'] = Agent(
            name="ACMEAssistant",   
            model="gpt-4.1-mini-2025-04-14",
            instructions=prompt_with_handoff_instructions("""
You are the friendly virtual assistant for ACME Shop, the premier destination for unique and exciting products.

Welcome new users warmly and ask how you can help them today.

Based on the user's request, route them to the appropriate specialist:
- AccountAgent: For account balance, order history, membership status, account settings
- KnowledgeAgent: For product information, specifications, recommendations, catalog questions  
- SearchAgent: For current trends, market research, real-time information, comparisons

Always be helpful, professional, and enthusiastic about ACME products!
"""),
            handoffs=[self.agents['account'], self.agents['knowledge'], self.agents['search']],
        )
        
        # Voice-optimized agents
        self.agents['search_voice'] = Agent(
            name="SearchVoiceAgent",
            model="gpt-4.1-mini-2025-04-14",
            instructions=voice_system_prompt + (
                "You are a web search specialist. Use WebSearchTool to find current information "
                "and provide clear, spoken-friendly summaries."
            ),
            tools=[WebSearchTool()],
        )
        
        if self.config.vector_store_id:
            self.agents['knowledge_voice'] = Agent(
                name="KnowledgeVoiceAgent",
                model="gpt-4.1-mini-2025-04-14",
                instructions=voice_system_prompt + (
                    "You are a product specialist. Use FileSearchTool to find product information "
                    "and explain it in a natural, conversational way."
                ),
                tools=[FileSearchTool(
                    max_num_results=3,
                    vector_store_ids=[self.config.vector_store_id],
                )],
            )
        else:
            self.agents['knowledge_voice'] = Agent(
                name="KnowledgeVoiceAgent",
                model="gpt-4.1-mini-2025-04-14",
                instructions=voice_system_prompt + (
                    "You are a product specialist. Provide general product information "
                    "in a natural, conversational way."
                ),
                tools=[],
            )
        
        self.agents['account_voice'] = Agent(
            name="AccountVoiceAgent",
            model="gpt-4.1-mini-2025-04-14",
            instructions=voice_system_prompt + (
                "You are an account specialist. Help customers with account information "
                "using available tools. Speak naturally and be helpful."
            ),
            tools=[get_account_info, get_recent_orders],
        )
        
        # Voice Triage Agent
        self.agents['triage_voice'] = Agent(
            name="ACMEVoiceAssistant",
            model="gpt-4.1-mini-2025-04-14",
            instructions=prompt_with_handoff_instructions("""
You are the friendly voice assistant for ACME Shop.

Greet users warmly and ask how you can help them today.

Based on their request, route to the appropriate specialist:
- AccountVoiceAgent: Account questions, balances, orders
- KnowledgeVoiceAgent: Product information and recommendations
- SearchVoiceAgent: Current trends and real-time information

Keep your responses natural and conversational for voice interaction.
"""),
            handoffs=[self.agents['account_voice'], self.agents['knowledge_voice'], self.agents['search_voice']],
        )
        
        logger.info("All agents configured successfully")

class VoiceAssistant:
    
    def __init__(self, agent_system: AgentSystem, config: ProductionConfig):
        self.agent_system = agent_system
        self.config = config
        self.setup_audio()
    
    def setup_audio(self):
        try:
            device_info = sd.query_devices(kind='input')
            self.sample_rate = int(device_info['default_samplerate'])
            logger.info(f"Audio configured: {self.sample_rate}Hz, {self.config.channels} channel(s)")
        except Exception as e:
            logger.error(f"Audio setup failed: {e}")
            self.sample_rate = self.config.sample_rate
    
    async def test_text_workflow(self):
        logger.info("Testing text workflow...")
        
        test_queries = [
            "Hi there! What's my account balance? My user ID is 1234567890",
            "I'm interested in your product catalog. Please give me all products in Outdoor Gear section and their prices",
            "What are the latest trends in camping equipment?",
            "Can you show me my recent orders? User ID 1234567890"
        ]
        
        triage_agent = self.agent_system.agents['triage']
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- Test Query {i} ---")
            print(f"User: {query}")
            
            try:
                if self.config.enable_tracing:
                    with trace(f"Text Test {i}"):
                        result = await Runner.run(triage_agent, query)
                else:
                    result = await Runner.run(triage_agent, query)
                
                print(f"Assistant: {result.final_output}")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Assistant: I apologize, but I encountered an error processing your request.")
            
            print("-" * 60)
    
    async def run_voice_assistant(self):
        logger.info("Starting voice assistant...")
        tts_settings = TTSModelSettings(
            instructions=(
                "Voice: Warm, friendly, and professional. "
                "Tone: Helpful and enthusiastic about ACME products. "
                "Pace: Natural conversational speed with clear pronunciation. "
                "Style: Approachable customer service representative."
            )
        )
        
        voice_config = VoicePipelineConfig(tts_settings=tts_settings)
        triage_voice_agent = self.agent_system.agents['triage_voice']
        
        print("\n" + "="*60)
        print("SHOP VOICE ASSISTANT ACTIVE")
        print("="*60)
        print("Instructions:")
        print("• Press ENTER to start recording your question")
        print("• Press ENTER again to stop recording")
        print("• Type your question for text response")
        print("• Type 'quit' or 'exit' to end the session")
        print("• Type 'help' for assistance")
        print("-" * 60)
        
        interaction_count = 0
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{datetime.now().strftime('%H:%M:%S')}] Press ENTER to speak or type your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using ACME Shop Assistant. Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Handle text input
                if user_input:
                    print(f"Text Query: {user_input}")
                    print("🔄 Processing your request...")
                    
                    if self.config.enable_tracing:
                        with trace(f"ACME Text Session {interaction_count}"):
                            agent_result = await Runner.run(triage_voice_agent, user_input)
                    else:
                        agent_result = await Runner.run(triage_voice_agent, user_input)
                    
                    print(f"Response: {agent_result.final_output}")
                    print("💬 Text response complete (Voice TTS not available for typed input)")
                    
                    interaction_count += 1
                    continue
                
                # Handle voice input
                print("Listening... (Press ENTER to stop)")
                
                recorded_chunks = []
                recording_active = True
                
                def audio_callback(indata, frames, time, status):
                    if recording_active:
                        recorded_chunks.append(indata.copy())
                
                # Start recording
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.config.channels,
                    dtype='int16',
                    callback=audio_callback
                ):
                    input()
                    recording_active = False
                
                if not recorded_chunks:
                    print("No audio recorded. Please try again.")
                    continue
                
                recording = np.concatenate(recorded_chunks, axis=0)
                audio_input = AudioInput(buffer=recording)
                
                print("🔄 Processing your voice request...")
                
                pipeline = VoicePipeline(
                    workflow=SingleAgentVoiceWorkflow(triage_voice_agent),
                    config=voice_config
                )
                
                # Process voice input and get voice response
                if self.config.enable_tracing:
                    with trace(f"ACME Voice Session {interaction_count}"):
                        result = await pipeline.run(audio_input)
                else:
                    result = await pipeline.run(audio_input)
                
                # Stream and play the voice response
                response_chunks = []
                async for event in result.stream():
                    if event.type == "voice_stream_event_audio":
                        response_chunks.append(event.data)
                
                if response_chunks:
                    response_audio = np.concatenate(response_chunks, axis=0)
                    print("🔊 Playing voice response...")
                    sd.play(response_audio, samplerate=self.sample_rate)
                    sd.wait()
                    print("Voice response complete")
                else:
                    print("No audio response generated")
                
                interaction_count += 1
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Voice assistant error: {e}")
                print(f"I apologize, but I encountered a technical issue. Please try again.")
    
    def _show_help(self):
        help_text = """
        SHOP VOICE ASSISTANT HELP
        
        Voice Commands:
        • Press ENTER to start voice recording
        • Press ENTER again to stop recording and process
        
        Text Commands:
        • Type your question directly instead of recording
        • 'help' - Show this help message
        • 'quit', 'exit', 'q' - End the session
        
        What I can help with:
        • Account information and order history
        • Product details and recommendations  
        • Current trends and market information
        • General shopping assistance
        """
        print(help_text)

async def setup_vector_store():
    print("Setting up vector store...")
    
    store_details = vector_manager.create_vector_store("ACME Shop Product Catalog")

    sample_documents = [
        "product_catalog.json"
    ]
    
    for doc_path in sample_documents:
        if os.path.exists(doc_path):
            vector_manager.upload_file_to_vector_store(doc_path, store_details["id"])
        else:
            logger.warning(f"Document not found: {doc_path}")
    
    print(f"Vector store ready. ID: {store_details['id']}")
    print(f"Set VECTOR_STORE_ID={store_details['id']} in your .env file")
    
    return store_details["id"]

async def main():
    print("Shop Voice Assistant")
    print("="*70)
    
    try:
        load_environment()
        initialize_components()
        agent_system = AgentSystem(config)
        voice_assistant = VoiceAssistant(agent_system, config)
        
        print(f"OpenAI API Key: {'***' + config.openai_api_key[-4:] if config.openai_api_key else 'Not set'}")
        print(f"Vector Store ID: {config.vector_store_id or 'Not configured'}")
        print(f"Audio Settings: {config.sample_rate}Hz, {config.channels} channel(s)")
        print(f"Tracing: {'Enabled' if config.enable_tracing else 'Disabled'}")
        
        while True:
            print("\n" + "="*50)
            print("📋 MAIN MENU")
            print("="*50)
            print("1. Test text workflow")
            print("2. Use shop assistant") 
            print("3. Setup vector store")
            print("4. Show system info")
            print("5. Exit")
            
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == '1':
                await voice_assistant.test_text_workflow()
            elif choice == '2':
                await voice_assistant.run_voice_assistant()
            elif choice == '3':
                if not config.vector_store_id:
                    vector_store_id = await setup_vector_store()
                    print(f"Update your .env file with: VECTOR_STORE_ID={vector_store_id}")
                else:
                    print(f"Vector store already configured: {config.vector_store_id}")
            elif choice == '4':
                show_system_info(config)
            elif choice == '5':
                print("👋 Thank you for using ACME Shop Assistant!")
                break
            else:
                print("Invalid choice. Please select 1-5.")
                
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Fatal error: {e}")
        return 1
    
    return 0

def show_system_info(config: ProductionConfig):
    info = f"""
    SYSTEM INFORMATION
    {'='*50}
    
    API Configuration:
    • OpenAI API Key: {'Configured' if config.openai_api_key else 'Missing'}
    • Vector Store ID: {config.vector_store_id or 'Not set'}
    
    Audio Configuration:
    • Sample Rate: {config.sample_rate}Hz
    • Channels: {config.channels}
    • Max Recording: {config.max_recording_duration}s
    
    System Settings:
    • Tracing Enabled: {config.enable_tracing}
    • Log Level: {config.log_level}
    
    Required Environment Variables:
    • OPENAI_API_KEY (required)
    • VECTOR_STORE_ID (optional)
    • SAMPLE_RATE (optional, default: 16000)
    • ENABLE_TRACING (optional, default: true)
    """
    print(info)

if __name__ == "__main__":
    print("Starting Shop Voice Assistant...")
    exit_code = asyncio.run(main())