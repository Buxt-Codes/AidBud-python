class Config:
    def __init__(self):
        #========== DATA PROCESSING ==========#
        self.llm = {}
        self.llm["fps"] = 1
        self.llm["num_workers"] = 4
        
        #========== RAG ==========#
        self.rag = {}
        self.rag["db_path"] = "./chroma_db"
        self.rag["embedder"] = "BAAI/bge-small-en-v1.5"
        self.rag["embedder_max_tokens"] = 512
        self.rag["tokeniser"] = "cl100k_base"
        self.rag["topK"] = 5

        #========== CONTEXT ==========#
        self.context = {}
        self.context["context_path"] = "./context"

        #========== LLM ==========#
        self.llm = {}
        self.llm["model_id"] = "google/gemma-3n-E4B-it"
        
