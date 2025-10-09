"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""
from roll.utils.logging import get_logger

logger = get_logger()

# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
# from .bandit.config import BanditEnvConfig
# from .bandit.env import BanditEnv
# from .countdown.config import CountdownEnvConfig
# from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
# from .metamathqa.env import MetaMathQAEnv
# from .metamathqa.config import MetaMathQAEnvConfig
try:
    from .search.config import SearchEnvConfig
    from .search.env import SearchEnv
    _search_import_success = True
except Exception as e:
    logger.info(f"Failed to import search environment: {e}")
    _search_import_success = False

try:
    from .numina_math.config import NuminaMathEnvConfig
    from .numina_math.env import NuminaMathEnv
    _numina_math_import_success = True
except Exception as e:
    logger.info(f"Failed to import numina_math environment: {e}")
    _numina_math_import_success = False

REGISTERED_ENVS = {
    "sokoban": SokobanEnv,
    "frozen_lake": FrozenLakeEnv,
}

REGISTERED_ENV_CONFIGS = {
    "sokoban": SokobanEnvConfig,
    "frozen_lake": FrozenLakeEnvConfig,
}

# Register search environment only if import was successful
if _search_import_success:
    REGISTERED_ENVS["search"] = SearchEnv
    REGISTERED_ENV_CONFIGS["search"] = SearchEnvConfig

# Register numina_math environment only if import was successful
if _numina_math_import_success:
    REGISTERED_ENVS["numina_math"] = NuminaMathEnv
    REGISTERED_ENV_CONFIGS["numina_math"] = NuminaMathEnvConfig

try:
    # add webshop-minimal to PYTHONPATH
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../../third_party/webshop-minimal"
    module_path = os.path.join(current_dir, relative_path)
    sys.path.append(module_path)

    from .webshop.config import WebShopEnvConfig
    from .webshop.env import WebShopEnv

    REGISTERED_ENVS["webshop"] = WebShopEnv
    REGISTERED_ENV_CONFIGS["webshop"] = WebShopEnvConfig
except Exception as e:
    logger.info(f"Failed to import webshop: {e}")
