"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""
from roll.utils.logging import get_logger

# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
# from .bandit.config import BanditEnvConfig
# from .bandit.env import BanditEnv
# from .countdown.config import CountdownEnvConfig
# from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv as OriginalSokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
from .cliff_walking.config import CliffWalkingEnvConfig
from .cliff_walking.env import CliffWalkingEnv
# from .metamathqa.env import MetaMathQAEnv
# from .metamathqa.config import MetaMathQAEnvConfig

# Jidi environments (disabled - replaced by standalone implementations)
# from .jidi.config import CliffWalkingConfig, GridWorldConfig, MiniGridConfig, SokobanConfig
# from .jidi.env import CliffWalkingEnv, GridWorldEnv, MiniGridEnv, SokobanEnv

# Search environment
from .search.config import SearchEnvConfig
from .search.env import SearchEnv

# NQ Search environment
from .nq_search.config import NQSearchEnvConfig
from .nq_search.env import NQSearchEnv

# SVG Reconstruction environment (VLM)
from .svg_reconstruction.config import SVGReconstructionEnvConfig
from .svg_reconstruction.env import SVGReconstructionEnv

# Game2048 environment (VLM)
from .game2048.config import Game2048EnvConfig
from .game2048.env import Game2048Env

# GameMatch (Shisen-Sho) environment (VLM)
from .game_match.config import GameMatchEnvConfig
from .game_match.env import GameMatchEnv

# VLM QA environment (GeoQA, MathVista, TabMWP, etc.)
from .vlm_qa.config import VLMQAEnvConfig
from .vlm_qa.env import VLMQAEnv

# Math environment (AIME, MATH, GSM8K)
from .math.config import MathEnvConfig, RollMathEnvConfig, RollMathBanditEnvConfig
from .math.env import MathEnv

logger = get_logger()

REGISTERED_ENVS = {
    # "bandit": BanditEnv,
    # "countdown": CountdownEnv,
    "sokoban": OriginalSokobanEnv,
    "frozen_lake": FrozenLakeEnv,
    "cliff_walking": CliffWalkingEnv,
    # 'alfworld': AlfredTXTEnv,
    # "metamathqa": MetaMathQAEnv,
    # Jidi environments (disabled - replaced by standalone implementations)
    # "jidi_cliffwalking": CliffWalkingEnv,
    # "jidi_gridworld": GridWorldEnv,
    # "jidi_minigrid": MiniGridEnv,
    # "jidi_sokoban": SokobanEnv,
    # Search environment
    "search": SearchEnv,
    # NQ Search environment
    "nq_search": NQSearchEnv,
    # SVG Reconstruction environment (VLM)
    "svg_reconstruction": SVGReconstructionEnv,
    # Game2048 environment (VLM)
    "game2048": Game2048Env,
    # GameMatch (Shisen-Sho) environment (VLM)
    "game_match": GameMatchEnv,
    # VLM QA environment (GeoQA, MathVista, TabMWP, etc.)
    "vlm_qa": VLMQAEnv,
    # Math environment (AIME, MATH, GSM8K)
    "math": MathEnv,
}

REGISTERED_ENV_CONFIGS = {
    # "bandit": BanditEnvConfig,
    # "countdown": CountdownEnvConfig,
    "sokoban": SokobanEnvConfig,
    "frozen_lake": FrozenLakeEnvConfig,
    "cliff_walking": CliffWalkingEnvConfig,
    # 'alfworld': AlfredEnvConfig,
    # "metamathqa": MetaMathQAEnvConfig,
    # Jidi environments (disabled - replaced by standalone implementations)
    # "jidi_cliffwalking": CliffWalkingConfig,
    # "jidi_gridworld": GridWorldConfig,
    # "jidi_minigrid": MiniGridConfig,
    # "jidi_sokoban": SokobanConfig,
    # Search environment
    "search": SearchEnvConfig,
    # NQ Search environment
    "nq_search": NQSearchEnvConfig,
    # SVG Reconstruction environment (VLM)
    "svg_reconstruction": SVGReconstructionEnvConfig,
    # Game2048 environment (VLM)
    "game2048": Game2048EnvConfig,
    # GameMatch (Shisen-Sho) environment (VLM)
    "game_match": GameMatchEnvConfig,
    # VLM QA environment (GeoQA, MathVista, TabMWP, etc.)
    "vlm_qa": VLMQAEnvConfig,
    # Math environment (AIME, MATH, GSM8K)
    "math": MathEnvConfig,
    # gem-based math environments
    "roll_math": RollMathEnvConfig,
    "roll_math_bandit": RollMathBanditEnvConfig,
}

# Webshop environment disabled due to missing dependencies
# try:
#     # add webshop-minimal to PYTHONPATH
#     import os
#     import sys
# 
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     relative_path = "../../../third_party/webshop-minimal"
#     module_path = os.path.join(current_dir, relative_path)
#     sys.path.append(module_path)
# 
#     from .webshop.config import WebShopEnvConfig
#     from .webshop.env import WebShopEnv
# 
#     REGISTERED_ENVS["webshop"] = WebShopEnv
#     REGISTERED_ENV_CONFIGS["webshop"] = WebShopEnvConfig
# except Exception as e:
#     logger.info(f"Failed to import webshop: {e}")
