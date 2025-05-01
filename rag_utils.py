from qdrant_client import models, QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.llms.sambanovasystems import SambaNovaCloud
from langchain.schema import SystemMessage
import requests
import time
import uuid
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility function to process items in batches
def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:
    def __init__(self, embed_model_name="intfloat/multilingual-e5-large", batch_size=32):
        """
        Initialize embedding model for text.
        Args:
            embed_model_name: Name of the HuggingFace embedding model.
            batch_size: Size of batches for embedding.
        """
        self.embed_model_name = embed_model_name
        self.batch_size = batch_size
        self.embed_model = self._load_embed_model()
        self.embeddings = []
        self.contexts = []

    def _load_embed_model(self):
        """Load the embedding model."""
        try:
            return HuggingFaceEmbeddings(model_name=self.embed_model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate_embedding(self, contexts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            return self.embed_model.embed_documents(contexts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def embed(self, contexts: List[str]):
        """Embed contexts in batches and store results."""
        self.contexts = contexts
        self.embeddings = []
        for batch_context in batch_iterate(contexts, self.batch_size):
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)
        logger.info(f"Embedded {len(self.contexts)} documents.")

class QdrantVDB_QB:
    def __init__(self, collection_name: str, vector_dim=1024, batch_size=512):
        """
        Initialize Qdrant vector database client.
        Args:
            collection_name: Name of the Qdrant collection.
            vector_dim: Dimension of the embedding vectors.
            batch_size: Size of batches for ingestion.
        """
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.client = None

    def define_client(self):
        """Connect to Qdrant server, trying gRPC first and falling back to HTTP."""
        try:
            # Try gRPC connection
            self.client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)
            # Test connection
            self.client.get_collections()
            logger.info("Connected to Qdrant using gRPC")
        except Exception as e:
            logger.warning(f"gRPC connection failed: {e}. Falling back to HTTP.")
            try:
                # Fallback to HTTP
                self.client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
                self.client.get_collections()
                logger.info("Connected to Qdrant using HTTP")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise

    def reset_collection(self):
        """Delete and recreate the collection."""
        try:
            if self.client.collection_exists(collection_name=self.collection_name):
                self.client.delete_collection(collection_name=self.collection_name)
            self.create_collection()
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise

    def create_collection(self):
        """Create a new Qdrant collection if it doesn't exist."""
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=5,
                        indexing_threshold=0
                    ),
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(always_ram=True)
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def ingest_data(self, embeddata: EmbedData):
        """Ingest embeddings and contexts into Qdrant."""
        try:
            for batch_context, batch_embeddings in zip(
                batch_iterate(embeddata.contexts, self.batch_size),
                batch_iterate(embeddata.embeddings, self.batch_size)
            ):
                batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_context))]
                self.client.upload_collection(
                    collection_name=self.collection_name,
                    vectors=batch_embeddings,
                    payload=[{"context": context} for context in batch_context],
                    ids=batch_ids
                )
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
            )
            logger.info(f"Ingested {len(embeddata.contexts)} vectors into {self.collection_name}")
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise

class Retriever:
    def __init__(self, vector_db: QdrantVDB_QB, embeddata: EmbedData):
        """Initialize retriever for searching Qdrant."""
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query: str, top_k: int = 2) -> List[Dict]:
        """Search Qdrant for relevant contexts."""
        try:
            query_embedding = self.embeddata.embed_model.embed_query(query)
            result = self.vector_db.client.search(
                collection_name=self.vector_db.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
                timeout=1000,
            )
            return [dict(r) for r in result]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

class RAG:
    def __init__(self, retriever: Retriever, llm_model: str = "Llama-4-Scout-17B-16E-Instruct"):
        """
        Initialize RAG pipeline with retriever and LLM.
        Args:
            retriever: Retriever instance for context retrieval.
            llm_model: Name of the SambaNovaCloud LLM model.
        """
        self.retriever = retriever
        self.system_message = SystemMessage(
            content="You are a helpful assistant that answers questions about the user's document in Turkish."
        )
        try:
            self.llm = SambaNovaCloud(model=llm_model, temperature=0.7, context_window=100000)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        self.qa_prompt_tmpl_str = (
            "Aşağıda bağlam bilgileri verilmiştir.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Yukarıdaki bağlama göre, soruyu adım adım yanıtlayın. Bilmiyorsanız, 'Bilmiyorum!' deyin.\n"
            "Soru: {query}\n"
            "Cevap: "
        )

    def generate_context(self, query: str) -> str:
        """Generate context from search results."""
        try:
            result = self.retriever.search(query)
            context_items = [item["payload"]["context"] for item in result]
            return "\n\n---\n\n".join(context_items) if context_items else "Bağlam bulunamadı."
        except Exception as e:
            logger.error(f"Context generation failed: {e}")
            raise

    def query(self, query: str) -> str:
        """Generate an answer for the query."""
        try:
            context = self.generate_context(query)
            prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)
            response = self.llm.complete(prompt)
            return str(response) if response else "Yanıt oluşturulamadı."
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

class Transcribe:
    def __init__(self, api_key: str):
        """Initialize AssemblyAI client."""
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"

    def transcribe_audio(self, audio_path: str) -> List[Dict[str, str]]:
        """Transcribe audio with AssemblyAI."""
        try:
            headers = {"authorization": self.api_key}
            with open(audio_path, "rb") as f:
                response = requests.post(
                    f"{self.base_url}/upload",
                    headers=headers,
                    data=f
                )
                response.raise_for_status()
                audio_url = response.json()["upload_url"]

            transcription_request = {
                "audio_url": audio_url,
                "speaker_labels": True,
                "language_code": "tr",
                "speakers_expected": 2,
            }
            response = requests.post(
                f"{self.base_url}/transcript",
                json=transcription_request,
                headers=headers
            )
            response.raise_for_status()
            transcript_id = response.json()["id"]

            while True:
                response = requests.get(
                    f"{self.base_url}/transcript/{transcript_id}",
                    headers=headers
                )
                result = response.json()
                if result["status"] == "completed":
                    break
                elif result["status"] == "error":
                    raise Exception(f"Transcription failed: {result['error']}")
                time.sleep(5)

            utterances = result.get("utterances", [])
            transcripts = [
                {"speaker": f"Konuşmacı {utterance['speaker']}", "text": utterance["text"]}
                for utterance in utterances
            ]
            logger.info(f"Transcribed {len(transcripts)} utterances.")
            return transcripts
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise