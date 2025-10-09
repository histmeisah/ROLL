#!/usr/bin/env python3
"""
Test script for dual environment configuration
"""

import sys
import os
import yaml

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_environment_registration():
    """Test if both environments are properly registered"""
    print("=== Testing Environment Registration ===")
    
    try:
        from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
        
        # Check search environment
        if "search" in REGISTERED_ENVS:
            print("✓ Search environment registered")
            print(f"  Class: {REGISTERED_ENVS['search']}")
            print(f"  Config: {REGISTERED_ENV_CONFIGS['search']}")
        else:
            print("✗ Search environment not registered")
            return False
        
        # Check numina_math environment
        if "numina_math" in REGISTERED_ENVS:
            print("✓ Numina Math environment registered")
            print(f"  Class: {REGISTERED_ENVS['numina_math']}")
            print(f"  Config: {REGISTERED_ENV_CONFIGS['numina_math']}")
        else:
            print("✗ Numina Math environment not registered")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Environment registration test failed: {e}")
        return False


def test_config_file():
    """Test the dual environment configuration file"""
    print("\n=== Testing Configuration File ===")
    
    config_path = "agentic_search_train.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration file loaded successfully")
        
        # Check experiment name
        exp_name = config.get("exp_name", "")
        if "dual_env" in exp_name:
            print(f"✓ Experiment name: {exp_name}")
        else:
            print(f"⚠ Experiment name doesn't indicate dual env: {exp_name}")
        
        # Check train environment manager
        train_env_manager = config.get("train_env_manager", {})
        train_tags = train_env_manager.get("tags", [])
        train_partitions = train_env_manager.get("num_groups_partition", [])
        
        if "SearchEnvTrain" in train_tags and "MathEnvTrain" in train_tags:
            print("✓ Train environment tags include both Search and Math")
        else:
            print(f"✗ Train environment tags missing: {train_tags}")
            return False
        
        if len(train_partitions) == 2 and train_partitions[0] == train_partitions[1]:
            print(f"✓ Train environment 1:1 ratio: {train_partitions}")
        else:
            print(f"✗ Train environment ratio not 1:1: {train_partitions}")
            return False
        
        # Check validation environment manager
        val_env_manager = config.get("val_env_manager", {})
        val_tags = val_env_manager.get("tags", [])
        val_partitions = val_env_manager.get("num_groups_partition", [])
        
        if "SearchEnvVal" in val_tags and "MathEnvVal" in val_tags:
            print("✓ Val environment tags include both Search and Math")
        else:
            print(f"✗ Val environment tags missing: {val_tags}")
            return False
        
        if len(val_partitions) == 2 and val_partitions[0] == val_partitions[1]:
            print(f"✓ Val environment 1:1 ratio: {val_partitions}")
        else:
            print(f"✗ Val environment ratio not 1:1: {val_partitions}")
            return False
        
        # Check custom environments
        custom_envs = config.get("custom_envs", {})
        required_envs = ["SearchEnvTrain", "MathEnvTrain", "SearchEnvVal", "MathEnvVal"]
        
        for env_name in required_envs:
            if env_name in custom_envs:
                env_config = custom_envs[env_name]
                env_type = env_config.get("env_type", "")
                dataset_path = env_config.get("env_config", {}).get("dataset_path", "")
                max_concurrent = env_config.get("env_config", {}).get("max_concurrent_calls", 0)
                
                print(f"✓ {env_name}: type={env_type}, dataset={os.path.basename(dataset_path)}, concurrent={max_concurrent}")
                
                # Check rate limiting
                if max_concurrent == 20000:
                    print(f"  ✓ Rate limit set to 20k/min")
                else:
                    print(f"  ⚠ Rate limit not 20k: {max_concurrent}")
                
            else:
                print(f"✗ Missing environment: {env_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration file test failed: {e}")
        return False


def test_dataset_paths():
    """Test if dataset paths exist"""
    print("\n=== Testing Dataset Paths ===")
    
    datasets = {
        "XHPANG_TRAIN": "/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
        "NUMINA_TRAIN": "/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet",
        "XHPANG_VAL": "/mnt/chensiheng/ai_researcher_data/xhpang_search/validation.parquet",
        "NUMINA_VAL": "/mnt/chensiheng/ai_researcher_data/numinamath_new/validation_subset.parquet"
    }
    
    all_exist = True
    for name, path in datasets.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {name}: {path} (NOT FOUND)")
            all_exist = False
    
    return all_exist


def test_environment_creation():
    """Test creating both environments"""
    print("\n=== Testing Environment Creation ===")
    
    try:
        from roll.agentic.env.search import SearchEnv, SearchEnvConfig
        from roll.agentic.env.numina_math import NuminaMathEnv, NuminaMathEnvConfig
        
        # Test search environment
        search_config = SearchEnvConfig(
            dataset_path="/mnt/chensiheng/ai_researcher_data/xhpang_search/train.parquet",
            max_instances=5,
            disable_limiter=True,
            use_mock_api=True
        )
        search_env = SearchEnv(search_config)
        print("✓ Search environment created successfully")
        
        # Test math environment
        math_config = NuminaMathEnvConfig(
            dataset_path="/mnt/chensiheng/ai_researcher_data/numinamath_new/train_subset.parquet",
            max_instances=5,
            disable_limiter=True,
            use_mock_api=True,
            use_xverify=False,
            use_otc=False
        )
        math_env = NuminaMathEnv(math_config)
        print("✓ Math environment created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Dual Environment Configuration Test")
    print("=" * 50)
    
    success = True
    
    success &= test_environment_registration()
    success &= test_config_file()
    success &= test_dataset_paths()
    success &= test_environment_creation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Dual environment configuration is ready.")
        print("\nNext steps:")
        print("1. Run: cd experiments/train_test_0802")
        print("2. Run: bash run_search_training.sh")
        print("3. Monitor training with both search and math environments")
    else:
        print("❌ Some tests failed! Please fix the issues before training.")
        sys.exit(1)
