#!/usr/bin/env python3
"""
Test script to validate the search environment training configuration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf
from roll.pipeline.agentic.agentic_config import AgenticConfig


def test_config():
    """Test if the configuration can be loaded correctly"""
    print("Testing search environment training configuration...")
    
    try:
        # Initialize hydra with the current directory as config path
        config_path = "."
        initialize(config_path=config_path, job_name="test_app", version_base=None)
        cfg = compose(config_name="agentic_search_train")
        
        print("✓ Configuration loaded successfully")
        
        # Convert to AgenticConfig
        agentic_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))
        
        print("✓ Configuration converted to AgenticConfig successfully")
        
        # Print key configuration details
        print(f"\nConfiguration Summary:")
        print(f"  Experiment name: {agentic_config.exp_name}")
        print(f"  Max steps: {agentic_config.max_steps}")
        print(f"  Rollout batch size: {agentic_config.rollout_batch_size}")
        print(f"  Pretrain model: {agentic_config.pretrain}")
        
        print(f"\nTrain Environment Manager:")
        print(f"  Num env groups: {agentic_config.train_env_manager.num_env_groups}")
        print(f"  Group size: {agentic_config.train_env_manager.group_size}")
        print(f"  Tags: {agentic_config.train_env_manager.tags}")
        
        print(f"\nValidation Environment Manager:")
        print(f"  Num env groups: {agentic_config.val_env_manager.num_env_groups}")
        print(f"  Group size: {agentic_config.val_env_manager.group_size}")
        print(f"  Tags: {agentic_config.val_env_manager.tags}")
        
        print(f"\nCustom Environments:")
        for env_name, env_config in agentic_config.custom_envs.items():
            print(f"  {env_name}:")
            print(f"    Type: {env_config.get('env_type')}")
            print(f"    Dataset: {env_config.get('env_config', {}).get('dataset_path', 'N/A')}")
            print(f"    Max steps: {env_config.get('env_config', {}).get('max_steps', 'N/A')}")
            print(f"    Max search calls: {env_config.get('env_config', {}).get('max_search_calls', 'N/A')}")
        
        print("\n✓ All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_import():
    """Test if search environment can be imported"""
    print("\nTesting search environment import...")
    
    try:
        from roll.agentic.env.search import SearchEnv, SearchEnvConfig
        print("✓ Search environment imported successfully")
        
        # Test basic environment creation
        config = SearchEnvConfig(
            dataset_path="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
            max_instances=1,
            use_mock_api=True,
            disable_limiter=True
        )
        
        env = SearchEnv(config)
        print("✓ Search environment created successfully")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print("✓ Environment reset successful")
        print(f"  Initial observation length: {len(obs)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_registration():
    """Test if search environment is properly registered"""
    print("\nTesting environment registration...")

    try:
        from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS

        print(f"Available environments: {list(REGISTERED_ENVS.keys())}")
        print(f"Available configs: {list(REGISTERED_ENV_CONFIGS.keys())}")

        if "search" in REGISTERED_ENVS:
            print("✓ Search environment is registered in REGISTERED_ENVS")
        else:
            print("✗ Search environment not found in REGISTERED_ENVS")
            print("  This might be due to import errors during environment registration")
            return False

        if "search" in REGISTERED_ENV_CONFIGS:
            print("✓ Search environment config is registered in REGISTERED_ENV_CONFIGS")
        else:
            print("✗ Search environment config not found in REGISTERED_ENV_CONFIGS")
            return False

        return True

    except Exception as e:
        print(f"✗ Environment registration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Search Environment Training Configuration Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Configuration loading
    success &= test_config()
    
    # Test 2: Environment import
    success &= test_environment_import()
    
    # Test 3: Environment registration
    success &= test_environment_registration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Configuration is ready for training.")
    else:
        print("❌ Some tests failed. Please check the configuration.")
    print("=" * 60)
