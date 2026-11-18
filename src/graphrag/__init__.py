"""GraphRAG retrieval: articles ranked by a Personalized PageRank walk over the
legal knowledge graph, served through the same interface as ``LawRetriever``.
"""

from src.graphrag.graph_retriever import GraphRetriever

__all__ = ["GraphRetriever"]
