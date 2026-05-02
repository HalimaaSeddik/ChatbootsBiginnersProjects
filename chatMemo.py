import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory


PAGE_CONFIG = {
    "page_title": "🤖 AI Chat Assistant",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

DEFAULT_MODEL = "phi3"
SYSTEM_PROMPT = """You are {name}, a helpful and knowledgeable AI assistant. 
You provide clear, accurate, and engaging responses. 
Be friendly, professional, and concise in your answers."""

class LLMManager:
    """Manages Ollama LLM instances with caching."""
    
    _instances = {}
    
    @classmethod
    def get_llm(cls, model_name: str, temperature: float = 0.7):
        """Get or create an LLM instance."""
        key = f"{model_name}_{temperature}"
        if key not in cls._instances:
            cls._instances[key] = Ollama(
                model=model_name,
                temperature=temperature
            )
        return cls._instances[key]


class SessionStore:
    """Manages chat histories per session."""
    
    _histories = {}
    
    @classmethod
    def get_history(cls, session_id: str):
        """Get or create chat history for a session."""
        if session_id not in cls._histories:
            cls._histories[session_id] = InMemoryChatMessageHistory()
        return cls._histories[session_id]
    
    @classmethod
    def clear_history(cls, session_id: str):
        """Clear history for a session."""
        if session_id in cls._histories:
            cls._histories[session_id] = InMemoryChatMessageHistory()

class ChainBuilder:
    """Builds LangChain conversation chains using LCEL."""
    
    @staticmethod
    def create_prompt_template(ai_name: str):
        """Create a chat prompt template with memory placeholder."""
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT.format(name=ai_name)),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
    
    @classmethod
    def build_chain(cls, llm, ai_name: str = "Claude"):
        """Build a complete conversation chain using LCEL."""
        prompt = cls.create_prompt_template(ai_name)
        
        # LCEL pipeline: prompt -> llm -> parser
        chain = prompt | llm | StrOutputParser()
        
        # Wrap with message history
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: SessionStore.get_history(session_id),
            input_messages_key="input",
            history_messages_key="history"
        )
        return chain_with_history


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "messages": [],
        "chain": None,
        "ai_name": "Claude",
        "model_name": DEFAULT_MODEL,
        "temperature": 0.7,
        "session_id": "default_session",
        "initialized": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_sidebar():
    """Configure the sidebar with settings."""
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        
        # AI Personality
        st.subheader("🎭 AI Personality")
        ai_name = st.text_input(
            "AI Name",
            value=st.session_state.ai_name,
            placeholder="Enter AI name..."
        )
        
        # Model Settings
        st.subheader("🧠 Model Configuration")
        model_name = st.selectbox(
            "Model",
            options=["phi3", "llama2", "mistral", "codellama", "gemma"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        st.markdown("---")
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Apply", use_container_width=True):
                st.session_state.ai_name = ai_name
                st.session_state.model_name = model_name
                st.session_state.temperature = temperature
                st.session_state.initialized = False
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                SessionStore.clear_history(st.session_state.session_id)
                st.session_state.initialized = False
                st.rerun()
        
        st.markdown("---")
        st.caption("Built with ❤️ using LangChain + Streamlit")

def initialize_chain():
    """Initialize or reinitialize the conversation chain."""
    if not st.session_state.initialized:
        with st.spinner(f"🚀 Loading {st.session_state.model_name}..."):
            try:
                llm = LLMManager.get_llm(
                    st.session_state.model_name,
                    st.session_state.temperature
                )
                chain = ChainBuilder.build_chain(
                    llm, 
                    st.session_state.ai_name
                )
                st.session_state.chain = chain
                st.session_state.initialized = True
                
                # Restore chat history to LangChain memory
                history = SessionStore.get_history(st.session_state.session_id)
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        history.add_message(HumanMessage(content=msg["content"]))
                    else:
                        history.add_message(AIMessage(content=msg["content"]))
                        
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
                st.info("💡 Make sure Ollama is running and the model is pulled.")
                return False
    return True

def display_chat_history():
    """Display the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

def handle_user_input():
    """Process user input and generate response."""
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to UI
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "avatar": "👤"
        })
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("💭 Thinking..."):
                    response = st.session_state.chain.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                
                message_placeholder.markdown(response)
                
                # Add AI message to UI history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "avatar": "🤖"
                })
                
            except Exception as e:
                message_placeholder.error(f"❌ Error: {str(e)}")


def main():
    """Main application entry point."""
    st.set_page_config(**PAGE_CONFIG)
    
    # Custom CSS
    st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
    }
    .stChatInputContainer {
        padding-bottom: 1rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 AI Chat Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Powered by LangChain + Ollama + Streamlit</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize
    init_session_state()
    setup_sidebar()
    
    # Initialize chain
    if not initialize_chain():
        st.stop()
    
    # Display chat
    display_chat_history()
    
    # Handle input
    handle_user_input()
    
    # Footer info
    if st.session_state.messages:
        st.markdown("---")
        st.caption(f"💬 {len(st.session_state.messages)//2} conversation turns | Model: {st.session_state.model_name} | Temp: {st.session_state.temperature}")

if __name__ == "__main__":
    main()