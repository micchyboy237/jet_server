from typing import Callable, Generator, Literal, Optional
# from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL
# from jet.llm.ollama.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from llama_index.core.schema import Document
from jet.llm.query.retrievers import query_llm, setup_deeplake_query, setup_index


class RAG:
    def __init__(
        self,
        path_or_docs: str | list[Document],
        system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        self.path_or_docs = path_or_docs
        self.system = system
        self.model = model
        self.store_path = kwargs.get("store_path")
        self.embed_model = kwargs.get("embed_model")
        self.overwrite = kwargs.get("overwrite")
        self.query_nodes = setup_index(
            path_or_docs,
            **kwargs,
        )

    def query(self, query: str, **kwargs) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        result = self.query_nodes(
            query, **kwargs)

        yield from query_llm(query, result['texts'], model=self.model)

    def get_results(self, query: str, **kwargs) -> str | Generator[str, None, None]:
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        result = self.query_nodes(
            query, **kwargs)

        return result

    def setup_deeplake_vectors(self) -> Callable:
        query_nodes = setup_deeplake_query(
            data_dir=self.path_or_docs,
            store_path=self.store_path,
            embed_model=self.embed_model,
            overwrite=self.overwrite
        )

        return query_nodes
