"""
Centralized Encoder Actor using Ray for distributed problem encoding.

Supports two encoder backends:
- "sentence_transformers": Lightweight text-only encoding (default, fast)
- "qwen3_vl": Unified multimodal encoding via Qwen3-VL-Embedding
  (text, image, video, and arbitrary combinations)

All environment workers call this single actor to encode problems,
avoiding multiple model loads and GPU memory conflicts.
"""

import ray
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_ACTOR_NAME = "encoder_actor_global"

# Supported encoder backends
ENCODER_SENTENCE_TRANSFORMERS = "sentence_transformers"
ENCODER_TRANSFORMERS = "transformers"  # native transformers AutoModel (no extra deps)
ENCODER_QWEN3_VL = "qwen3_vl"
ENCODER_DUMMY = "dummy"  # random embeddings, for debug/testing (no model loading)


@ray.remote(num_cpus=1, num_gpus=0)
class EncoderActor:
    """
    Centralized problem encoder service using Ray Actor.

    Supports pluggable encoder backends:
    - sentence_transformers: text-only, lightweight (~90MB), requires sentence-transformers
    - transformers: text-only via AutoModel+AutoTokenizer (no extra deps)
    - qwen3_vl: multimodal (text/image/video), Qwen3-VL-Embedding (~4GB for 2B)
    - dummy: random hash-based embeddings, no model loading (for debug/testing)

    All environment workers call this single actor, avoiding:
    - Multiple model loads (memory exhaustion)
    - Thread safety issues
    - GPU memory conflicts with vLLM
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        encoder_type: str = ENCODER_SENTENCE_TRANSFORMERS,
        embedding_dim: Optional[int] = None,
    ):
        """
        Initialize encoder actor.

        Args:
            model_name_or_path: Model path or HuggingFace model name.
                - sentence_transformers: e.g. "all-MiniLM-L6-v2"
                - transformers: e.g. local path to any causal/embedding model
                - qwen3_vl: e.g. "Qwen/Qwen3-VL-Embedding-2B"
            device: Device to load model on (default: cpu)
            encoder_type: Backend type ("sentence_transformers", "transformers", or "qwen3_vl")
            embedding_dim: Output embedding dimension (qwen3_vl only, default: model max)
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.encoder = None
        self.context_dim = None

        self._load_model()

        logger.info(
            f"[EncoderActor] Initialized: type={encoder_type}, model={model_name_or_path}, "
            f"device={device}, context_dim={self.context_dim}"
        )

    def _load_model(self):
        """Load encoder model based on encoder_type."""
        if self.encoder_type == ENCODER_SENTENCE_TRANSFORMERS:
            self._load_sentence_transformers()
        elif self.encoder_type == ENCODER_TRANSFORMERS:
            self._load_transformers_native()
        elif self.encoder_type == ENCODER_QWEN3_VL:
            self._load_qwen3_vl()
        else:
            raise ValueError(
                f"Unknown encoder_type: {self.encoder_type}. "
                f"Supported: {ENCODER_SENTENCE_TRANSFORMERS}, {ENCODER_TRANSFORMERS}, {ENCODER_QWEN3_VL}"
            )

    def _load_sentence_transformers(self):
        """Load SentenceTransformer model (text-only)."""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name_or_path, device=self.device)
            test_emb = self.encoder.encode("test", convert_to_numpy=True)
            self.context_dim = test_emb.shape[0]
            logger.info(f"[EncoderActor] SentenceTransformer loaded, context_dim={self.context_dim}")
        except Exception as e:
            logger.error(f"[EncoderActor] Failed to load SentenceTransformer: {e}")
            raise

    def _load_transformers_native(self):
        """Load model via transformers AutoModel (no extra deps, text-only)."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device).eval()

            # Probe context_dim with a test encoding
            inputs = self._tokenizer("test", return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                test_emb = outputs.last_hidden_state[:, -1, :]  # [EOS] token
            full_dim = test_emb.shape[1]
            self.context_dim = self.embedding_dim or full_dim
            self.encoder = "transformers_native"
            logger.info(
                f"[EncoderActor] transformers AutoModel loaded, "
                f"full_dim={full_dim}, context_dim={self.context_dim}"
            )
        except Exception as e:
            logger.error(f"[EncoderActor] Failed to load transformers model: {e}")
            raise

    def _load_qwen3_vl(self):
        """Load Qwen3-VL-Embedding model (multimodal)."""
        try:
            import torch
            from qwen3_vl_embedding import Qwen3VLEmbedder
            self.encoder = Qwen3VLEmbedder(
                model_name_or_path=self.model_name_or_path,
                torch_dtype=torch.bfloat16,
            )
            # Probe context_dim with a test encoding
            test_emb = self.encoder.process([{"text": "test"}])
            full_dim = test_emb.shape[1]
            self.context_dim = self.embedding_dim or full_dim
            logger.info(
                f"[EncoderActor] Qwen3-VL-Embedding loaded, "
                f"full_dim={full_dim}, context_dim={self.context_dim}"
            )
        except ImportError:
            logger.warning(
                "[EncoderActor] qwen3_vl_embedding not found, "
                "falling back to transformers-based loading"
            )
            self._load_qwen3_vl_transformers()
        except Exception as e:
            logger.error(f"[EncoderActor] Failed to load Qwen3-VL-Embedding: {e}")
            raise

    def _load_qwen3_vl_transformers(self):
        """Fallback: Load Qwen3-VL-Embedding via transformers AutoModel."""
        import torch
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()

        # Probe context_dim
        inputs = self._processor(
            text=["test"], return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use last hidden state [EOS] token as embedding
            test_emb = outputs.last_hidden_state[:, -1, :]
        full_dim = test_emb.shape[1]
        self.context_dim = self.embedding_dim or full_dim
        self.encoder = "transformers_fallback"
        logger.info(
            f"[EncoderActor] Qwen3-VL loaded via transformers, "
            f"full_dim={full_dim}, context_dim={self.context_dim}"
        )

    def encode(self, data: Union[str, Dict[str, Any]]) -> bytes:
        """
        Encode input to embedding.

        Args:
            data: Input to encode. Accepts:
                - str: Pure text (works with all backends)
                - Dict: Multimodal input (qwen3_vl backend), e.g.:
                    {"text": "solve this"}
                    {"image": "path/to/img.png"}
                    {"text": "describe", "image": "path/to/img.png"}
                    {"video": "path/to/video.mp4"}

        Returns:
            Pickled numpy array (bytes)
        """
        embedding = self._encode_impl(data)
        # Truncate to context_dim if needed (MRL support)
        if self.context_dim and embedding.shape[0] > self.context_dim:
            embedding = embedding[:self.context_dim]
        return pickle.dumps(embedding)

    def _encode_impl(self, data: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Internal encoding dispatch."""
        if self.encoder_type == ENCODER_SENTENCE_TRANSFORMERS:
            return self._encode_sentence_transformers(data)
        elif self.encoder_type == ENCODER_TRANSFORMERS:
            return self._encode_transformers_native(data)
        else:
            return self._encode_qwen3_vl(data)

    def _encode_sentence_transformers(self, data: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Encode with SentenceTransformer (text-only)."""
        # Extract text from Dict if needed
        if isinstance(data, dict):
            text = data.get("text", "")
            if not text:
                logger.warning(
                    "[EncoderActor] SentenceTransformer received non-text input, "
                    "using empty string. Consider switching to qwen3_vl encoder."
                )
        else:
            text = data
        return self.encoder.encode(text, convert_to_numpy=True)

    def _encode_transformers_native(self, data: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Encode with transformers AutoModel (text-only, no extra deps)."""
        import torch

        if isinstance(data, dict):
            text = data.get("text", "")
        else:
            text = data
        if not text:
            text = " "

        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            embedding = outputs.last_hidden_state[:, -1, :].squeeze(0)
        return embedding.float().cpu().numpy()

    def _encode_qwen3_vl(self, data: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Encode with Qwen3-VL-Embedding (multimodal)."""
        # Normalize input to Dict format
        if isinstance(data, str):
            input_dict = {"text": data}
        else:
            input_dict = data

        if self.encoder != "transformers_fallback":
            # Use Qwen3VLEmbedder
            embeddings = self.encoder.process([input_dict])
            return embeddings[0].cpu().numpy() if hasattr(embeddings[0], 'cpu') else embeddings[0]
        else:
            # Transformers fallback
            return self._encode_qwen3_vl_transformers(input_dict)

    def _encode_qwen3_vl_transformers(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """Encode using transformers AutoModel fallback."""
        import torch
        from PIL import Image

        text = input_dict.get("text", "")
        images = None

        # Handle image input
        if "image" in input_dict:
            img = input_dict["image"]
            if isinstance(img, str):
                images = [Image.open(img)]
            elif isinstance(img, np.ndarray):
                images = [Image.fromarray(img, mode="RGB")]
            elif isinstance(img, Image.Image):
                images = [img]
            else:
                images = [img]

        inputs = self._processor(
            text=[text] if text else None,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            embedding = outputs.last_hidden_state[:, -1, :].squeeze(0)
        return embedding.float().cpu().numpy()

    def encode_batch(self, data_list: List[Union[str, Dict[str, Any]]]) -> List[bytes]:
        """
        Encode multiple inputs to embeddings.

        Args:
            data_list: List of inputs (str or Dict)

        Returns:
            List of pickled numpy arrays (bytes)
        """
        if self.encoder_type == ENCODER_SENTENCE_TRANSFORMERS:
            # Batch encode for text
            texts = [
                (d.get("text", "") if isinstance(d, dict) else d)
                for d in data_list
            ]
            embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            if self.context_dim and embeddings.shape[1] > self.context_dim:
                embeddings = embeddings[:, :self.context_dim]
            return [pickle.dumps(emb) for emb in embeddings]
        elif self.encoder_type == ENCODER_TRANSFORMERS:
            # Batch one-by-one for transformers native
            return [self.encode(d) for d in data_list]
        else:
            # Qwen3-VL: encode one by one (model handles internally)
            return [self.encode(d) for d in data_list]

    def encode_numpy(self, data: Union[str, Dict[str, Any]]) -> np.ndarray:
        """
        Encode and return numpy array directly.

        Args:
            data: Input to encode (str or Dict)

        Returns:
            Numpy array embedding
        """
        embedding = self._encode_impl(data)
        if self.context_dim and embedding.shape[0] > self.context_dim:
            embedding = embedding[:self.context_dim]
        return embedding

    def get_context_dim(self) -> int:
        """Get embedding dimension."""
        return self.context_dim

    def health_check(self) -> dict:
        """Health check for the encoder."""
        return {
            "status": "healthy",
            "model": self.model_name_or_path,
            "device": self.device,
            "encoder_type": self.encoder_type,
            "context_dim": self.context_dim,
        }


def get_encoder_actor_by_name(actor_name: str = DEFAULT_ENCODER_ACTOR_NAME):
    """
    Get EncoderActor using Ray Named Actor pattern.

    Args:
        actor_name: Name of the Ray actor to retrieve

    Returns:
        Ray actor handle or None if not found
    """
    try:
        actor = ray.get_actor(actor_name)
        logger.info(f"Successfully retrieved EncoderActor: {actor_name}")
        return actor
    except ValueError:
        logger.debug(f"EncoderActor '{actor_name}' not found")
        return None
    except Exception as e:
        logger.warning(f"Failed to get EncoderActor '{actor_name}': {e}")
        return None


def create_encoder_actor(
    model_name_or_path: str,
    actor_name: str = DEFAULT_ENCODER_ACTOR_NAME,
    device: str = "cpu",
    encoder_type: str = ENCODER_SENTENCE_TRANSFORMERS,
    embedding_dim: Optional[int] = None,
):
    """
    Create EncoderActor as a Ray Named Actor.

    Args:
        model_name_or_path: Model path or name
        actor_name: Name for the Ray actor
        device: Device to load model on
        encoder_type: Backend type ("sentence_transformers" or "qwen3_vl")
        embedding_dim: Output embedding dimension (qwen3_vl only)

    Returns:
        Ray actor handle
    """
    logger.info(f"Creating EncoderActor: name={actor_name}, type={encoder_type}")

    # Kill existing actor if any
    try:
        existing_actor = ray.get_actor(actor_name)
        logger.info(f"Found existing EncoderActor, killing it...")
        ray.kill(existing_actor)
    except ValueError:
        pass

    encoder_actor = EncoderActor.options(
        name=actor_name,
        lifetime="detached",
    ).remote(
        model_name_or_path=model_name_or_path,
        device=device,
        encoder_type=encoder_type,
        embedding_dim=embedding_dim,
    )

    logger.info(f"EncoderActor created successfully: {actor_name}")
    return encoder_actor
