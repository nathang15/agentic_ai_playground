import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
import uuid

def setup_environment():
    print("Loading environment variables from .env file...")
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found."
        )
    
    # LangChain tracing
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "simple_chatbot"
        print("LangChain tracing enabled")
    else:
        print("LangChain tracing disabled (LANGCHAIN_API_KEY not found)")

class LangGraphChatbot:
    def __init__(self, model_name="gpt-4.1"):
        """Initialize the chatbot with LangGraph"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.sys_msg = SystemMessage(content="You are a helpful assistant")
        # Persisent memory
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        
        print(f"Chatbot initialized with {model_name}")
    
    def assistant(self, state: MessagesState):
        return {"messages": [self.llm.invoke([self.sys_msg] + state["messages"])]}
    
    def _build_graph(self):
        """Build and compile the LangGraph"""
        # Create the state graph
        builder = StateGraph(MessagesState)

        # Add the assistant node
        builder.add_node("assistant", self.assistant)

        # Add edge from START to assistant
        builder.add_edge(START, "assistant")

        # Compile with memory checkpointer
        graph = builder.compile(checkpointer=self.memory)
        return graph
    
    def visualize_graph(self, output_path: str = "langgraph.png"):
        try:
            png_bytes = self.graph.get_graph(xray=True).draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            print(f"LangGraph diagram saved to {os.path.abspath(output_path)}")
        except Exception as e:
            print(f"Could not save graph image: {e}")
    
    def chat(self, message: str, thread_id: str = None):
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        # Create human message instance
        human_msg = HumanMessage(content=message)
        
        # Invoke the graph with thread_id for memory
        response = self.graph.invoke(
            {"messages": [human_msg]}, 
            config={"thread_id": thread_id}
        )
        
        return response, thread_id
    
    def print_conversation(self, response):
        print("\n" + "="*50)
        for msg in response['messages']:
            msg.pretty_print()
        print("="*50 + "\n")
    
    def interactive_chat(self, thread_id: str = None):
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        print(f"Starting interactive chat (Thread ID: {thread_id})")
        print("Type 'quit', 'exit', or 'bye' to end the conversation\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response, _ = self.chat(user_input, thread_id)
                
                assistant_msg = response['messages'][-1]
                print(f"Assistant: {assistant_msg.content}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def demo_basic_usage():
    print("=== LangGraph Chatbot Demo ===\n")
    
    chatbot = LangGraphChatbot()
    print("Graph structure:")
    chatbot.visualize_graph()
    
    # Single conversation example
    print("Single message example:")
    response, thread_id = chatbot.chat("Hi simple chat bot")
    chatbot.print_conversation(response)
    
    # With persistent memory
    print("With persistent memory")
    response, _ = chatbot.chat("What is my name?", thread_id)
    chatbot.print_conversation(response)
    
    response, _ = chatbot.chat("My name is Nathan", thread_id)
    chatbot.print_conversation(response)
    
    response, _ = chatbot.chat("Do you remember my name?", thread_id)
    chatbot.print_conversation(response)
    
    # Without persistent memory
    print("Without persistent memory")
    response, new_thread = chatbot.chat("Do you remember my name?")
    chatbot.print_conversation(response)

def main():
    try:
        setup_environment()
        demo_basic_usage()
        chatbot = LangGraphChatbot()
        chatbot.interactive_chat()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()