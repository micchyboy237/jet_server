import threading
from typing import Callable, Generator, Literal, Optional
from llama_index.core.schema import Document
# from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL
# from jet.llm.ollama.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import initialize_ollama_settings
from jet.llm.query.retrievers import query_llm, setup_index, setup_semantic_search


def remove_substrings(contexts: list[str]) -> list[str]:
    # Sort by length to ensure that substrings are checked after the longer strings
    contexts.sort(key=len, reverse=True)

    result = []
    for context in contexts:
        # Add the context to result if it's not a substring of any existing item
        if not any(context in other for other in result):
            result.append(context)

    return result


class RAG:
    def __init__(
        self,
        path_or_docs: str | list[Document],
        system: Optional[str] = None,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        **kwargs,
    ):
        self.path_or_docs = path_or_docs
        self.system = system
        self.model = model
        self.mode = mode
        self.store_path = kwargs.get("store_path")
        self.embed_model = kwargs.get("embed_model")
        self.chunk_size = kwargs.get("chunk_size")
        self.chunk_overlap = kwargs.get("chunk_overlap")
        self.overwrite = kwargs.get("overwrite")

        initialize_ollama_settings({
            "llm_model": self.model,
            "embedding_model": self.embed_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        })

        if mode in ["faiss", "graph_nx"]:
            self.query_nodes = setup_semantic_search(
                path_or_docs,
                mode=mode,
                **kwargs,
            )
        else:
            self.query_nodes = setup_index(
                path_or_docs,
                mode=mode,
                **kwargs,
            )

    def query(self, query: str, contexts: list[str] = [], system: Optional[str] = None, stop_event: Optional[threading.Event] = None, **kwargs) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        if not contexts:
            result = self.query_nodes(
                query, **kwargs)
            contexts = result['texts']
            contexts = remove_substrings(contexts)

        yield from query_llm(query, contexts, model=self.model, system=system, stop_event=stop_event)

    def get_results(self, query: str, **kwargs) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        options = {
            "query": query,
            **kwargs
        }

        result = self.query_nodes(**options)

        return result
