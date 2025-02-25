import threading
from typing import Callable, Generator, Literal, Optional
from jet.file.utils import get_file_last_modified
from jet.logger import logger
from jet.memory.lru_cache import LRUCache
from llama_index.core.schema import Document
# from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL
# from jet.llm.ollama.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import initialize_ollama_settings
from jet.llm.query.retrievers import load_documents, query_llm, setup_index, setup_semantic_search


_active_search_documents = LRUCache(max_size=1)


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

        self.mode = mode
        self.setup_args = kwargs

        self.last_modified: Optional[float] = None
        self.query_nodes: Optional[Callable] = None

        self._check_documents_cache()

    def _check_documents_cache(self):
        self.documents = self._load_documents()

    def _load_documents(self) -> list[Document]:
        global _active_search_documents

        documents: list[Document]
        if isinstance(self.path_or_docs, str):
            current_modified = get_file_last_modified(self.path_or_docs)

            if not self.last_modified or current_modified > self.last_modified:
                if not self.last_modified:
                    logger.debug("Creating document embeddings from file...")
                else:
                    logger.warning("File has changed, reloading index...")
                documents = load_documents(
                    self.path_or_docs, **self.setup_args)

                self._setup_query_callback(documents)

                self.last_modified = current_modified
                _active_search_documents.put(
                    str(self.last_modified), documents)
            else:
                documents = _active_search_documents.get(
                    str(self.last_modified))

        elif isinstance(self.path_or_docs, list):
            documents = self.path_or_docs
        return documents

    def _setup_query_callback(self, documents: list[Document]):
        if self.mode in ["faiss", "graph_nx"]:
            self.query_nodes = setup_semantic_search(
                documents,
                mode=self.mode,
                **self.setup_args,
            )
        else:
            self.query_nodes = setup_index(
                documents,
                mode=self.mode,
                **self.setup_args,
            )

    def query(self, query: str, contexts: list[str] = [], system: Optional[str] = None, stop_event: Optional[threading.Event] = None, **kwargs) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        self._check_documents_cache()

        if not contexts:
            result = self.query_nodes(
                query, **self.setup_args)
            contexts = result['texts']
            contexts = remove_substrings(contexts)

        yield from query_llm(query, contexts, model=self.model, system=system, stop_event=stop_event)

    def get_results(self, query: str, **kwargs) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        self._check_documents_cache()

        options = {
            "query": query,
            **self.setup_args
        }

        result = self.query_nodes(**options)

        return result
