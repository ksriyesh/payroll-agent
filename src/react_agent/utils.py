"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
import logging
from langchain_groq import ChatGroq
from langgraph.config import RunnableConfig
from src.react_agent.configuration import Configuration

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(config: RunnableConfig) -> ChatGroq:
    """Load the chat model with the given configuration."""
    logger.info("🤖 Loading chat model")
    
    try:
        # Get configuration - use direct instantiation instead of from_context
        configuration = Configuration()
        logger.debug(f"Configuration loaded: {configuration.model}")
        
        # Create model
        model = ChatGroq(
            model=configuration.model,
            temperature=configuration.temperature,
            max_tokens=configuration.max_tokens,
            timeout=30,
            max_retries=2
        )
        
        logger.info(f"✅ Chat model loaded successfully: {configuration.model}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading chat model: {str(e)}", exc_info=True)
        raise
