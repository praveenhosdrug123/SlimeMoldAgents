from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
import json
from datetime import datetime

class DataManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory="./agent_data")
        
    def store_interaction(self, state: AgentState):
        # Convert messages to documents
        documents = [
            Document(
                page_content=msg.content,
                metadata={
                    "agent": state.current_agent,
                    "timestamp": datetime.now().isoformat(),
                    "message_type": msg.__class__.__name__,
                    "path_history": ",".join(state.path_history)
                }
            ) for msg in state.messages
        ]
        
        # Store in vector database
        self.vector_store.add_documents(documents)
        
        # Store raw data in JSON
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "path_history": state.path_history,
            "context": state.context,
            "performance_metrics": state.performance_metrics,
            "messages": [(msg.__class__.__name__, msg.content) for msg in state.messages]
        }
        
        with open(f"agent_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(interaction_data, f)
        
    def query_similar_interactions(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)
