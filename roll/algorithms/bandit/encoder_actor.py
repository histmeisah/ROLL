"""
Centralized Encoder Actor using Ray for distributed problem encoding.

This actor runs as a single Ray actor that all environment instances can call
to encode problems, avoiding the overhead of loading SentenceTransformer in each worker.
"""

import ray
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_ACTOR_NAME = "encoder_actor_global"


@ray.remote(num_cpus=1, num_gpus=0)
class EncoderActor:
    """
    Centralized problem encoder service using Ray Actor.

    Responsibilities:
    1. Load SentenceTransformer model once
    2. Provide encoding API for all environment workers
    3. Batch encoding support for efficiency

    All environment workers call this single actor, avoiding:
    - Multiple model loads (memory exhaustion)
    - Thread safety issues with SentenceTransformer
    - GPU memory conflicts with vLLM
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
    ):
        """
        Initialize encoder actor.

        Args:
            model_name_or_path: Path to SentenceTransformer model
            device: Device to load model on (default: cpu to avoid GPU conflicts)
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.encoder = None
        self.context_dim = None

        # Load model
        self._load_model()

        logger.info(
            f"[EncoderActor] Initialized with model={model_name_or_path}, "
            f"device={device}, context_dim={self.context_dim}"
        )

    def _load_model(self):
        """Load SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name_or_path, device=self.device)
            # Get embedding dimension
            test_emb = self.encoder.encode("test", convert_to_numpy=True)
            self.context_dim = test_emb.shape[0]
            logger.info(f"[EncoderActor] Loaded encoder, context_dim={self.context_dim}")
        except Exception as e:
            logger.error(f"[EncoderActor] Failed to load model: {e}")
            raise

    def encode(self, text: str) -> bytes:
        """
        Encode a single text to embedding.

        Args:
            text: Text to encode

        Returns:
            Pickled numpy array (bytes)
        """
        import pickle
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        return pickle.dumps(embedding)

    def encode_batch(self, texts: List[str]) -> List[bytes]:
        """
        Encode multiple texts to embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            List of pickled numpy arrays (bytes)
        """
        import pickle
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        return [pickle.dumps(emb) for emb in embeddings]

    def encode_numpy(self, text: str) -> np.ndarray:
        """
        Encode text and return numpy array directly.

        Args:
            text: Text to encode

        Returns:
            Numpy array embedding
        """
        return self.encoder.encode(text, convert_to_numpy=True)

    def get_context_dim(self) -> int:
        """Get embedding dimension."""
        return self.context_dim

    def health_check(self) -> dict:
        """Health check for the encoder."""
        return {
            "status": "healthy",
            "model": self.model_name_or_path,
            "device": self.device,
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
):
    """
    Create EncoderActor as a Ray Named Actor.

    Args:
        model_name_or_path: Path to SentenceTransformer model
        actor_name: Name for the Ray actor
        device: Device to load model on

    Returns:
        Ray actor handle
    """
    logger.info(f"Creating EncoderActor with name: {actor_name}")

    # Kill existing actor if any (to ensure fresh start)
    try:
        existing_actor = ray.get_actor(actor_name)
        logger.info(f"Found existing EncoderActor, killing it...")
        ray.kill(existing_actor)
    except ValueError:
        pass  # Actor doesn't exist, that's fine

    encoder_actor = EncoderActor.options(
        name=actor_name,
        lifetime="detached",
    ).remote(
        model_name_or_path=model_name_or_path,
        device=device,
    )

    logger.info(f"EncoderActor created successfully: {actor_name}")
    return encoder_actor
