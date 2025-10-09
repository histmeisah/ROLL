#!/usr/bin/env python3
"""
Test script for async training configuration
"""

import sys
import os
import yaml

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_async_configuration():
    """Test async training configuration"""
    print("=== Testing Async Training Configuration ===")
    
    config_path = "agentic_search_train.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration file loaded successfully")
        
        # Check async_generation_ratio
        async_ratio = config.get("async_generation_ratio", 0)
        if async_ratio > 0:
            print(f"✓ Async training enabled: async_generation_ratio = {async_ratio}")
        else:
            print("✗ Async training not enabled")
            return False
        
        # Check schedule configuration
        schedule_config = config.get("schedule_config", {})
        if schedule_config:
            print("✓ Schedule configuration found:")
            print(f"  generate_opt_level: {schedule_config.get('generate_opt_level', 'N/A')}")
            print(f"  max_running_requests: {schedule_config.get('max_running_requests', 'N/A')}")
            print(f"  is_use_additional_prompts: {schedule_config.get('is_use_additional_prompts', 'N/A')}")
            print(f"  max_additional_running_prompts: {schedule_config.get('max_additional_running_prompts', 'N/A')}")
        else:
            print("⚠ Schedule configuration not found")
        
        # Check environment manager configuration
        train_env_manager = config.get("train_env_manager", {})
        if train_env_manager:
            print("✓ Train environment manager configuration:")
            print(f"  max_env_num_per_worker: {train_env_manager.get('max_env_num_per_worker', 'N/A')}")
            print(f"  num_env_groups: {train_env_manager.get('num_env_groups', 'N/A')}")
            print(f"  group_size: {train_env_manager.get('group_size', 'N/A')}")
            print(f"  max_traj_per_env: {train_env_manager.get('max_traj_per_env', 'N/A')}")
        else:
            print("✗ Train environment manager configuration not found")
            return False
        
        # Check actor_infer configuration for async
        actor_infer = config.get("actor_infer", {})
        if actor_infer:
            strategy_config = actor_infer.get("strategy_args", {}).get("strategy_config", {})
            print("✓ Actor inference configuration:")
            print(f"  max_num_seqs: {strategy_config.get('max_num_seqs', 'N/A')}")
            print(f"  max_num_batched_tokens: {strategy_config.get('max_num_batched_tokens', 'N/A')}")
            print(f"  tensor_parallel_size: {strategy_config.get('tensor_parallel_size', 'N/A')}")
        else:
            print("✗ Actor inference configuration not found")
            return False
        
        # Check batch sizes for async training
        rollout_batch_size = config.get("rollout_batch_size", 0)
        val_batch_size = config.get("val_batch_size", 0)
        sequence_length = config.get("sequence_length", 0)
        
        print("✓ Batch configuration:")
        print(f"  rollout_batch_size: {rollout_batch_size}")
        print(f"  val_batch_size: {val_batch_size}")
        print(f"  sequence_length: {sequence_length}")
        
        # Validate async-specific requirements
        if rollout_batch_size < 512:
            print("⚠ rollout_batch_size might be too small for async training")
        
        if train_env_manager.get('max_env_num_per_worker', 0) < 8:
            print("⚠ max_env_num_per_worker might be too small for async training")
        
        # Check dual environment configuration
        tags = train_env_manager.get('tags', [])
        num_groups_partition = train_env_manager.get('num_groups_partition', [])
        
        if len(tags) == 2 and len(num_groups_partition) == 2:
            print("✓ Dual environment configuration:")
            print(f"  Environments: {tags}")
            print(f"  Group partition: {num_groups_partition}")
            
            if num_groups_partition[0] == num_groups_partition[1]:
                print("✓ 1:1 ratio maintained between environments")
            else:
                print("⚠ Environment ratio is not 1:1")
        else:
            print("✗ Dual environment configuration incorrect")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_async_compatibility():
    """Test compatibility with async training requirements"""
    print("\n=== Testing Async Training Compatibility ===")
    
    try:
        # Test if required modules are available
        from roll.configs.base_config import BaseConfig
        print("✓ ROLL base config module available")
        
        # Check if async_generation_ratio is supported
        base_config = BaseConfig(sequence_length=1024)  # Provide required parameter
        if hasattr(base_config, 'async_generation_ratio'):
            print("✓ async_generation_ratio parameter supported")
        else:
            print("✗ async_generation_ratio parameter not supported")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        return False


def test_async_performance_settings():
    """Test performance-related settings for async training"""
    print("\n=== Testing Async Performance Settings ===")
    
    config_path = "agentic_search_train.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check GPU allocation
        num_gpus_per_node = config.get("num_gpus_per_node", 0)
        num_nodes = config.get("num_nodes", 1)
        total_gpus = num_gpus_per_node * num_nodes
        
        print(f"✓ Total GPU resources: {total_gpus} GPUs ({num_nodes} nodes × {num_gpus_per_node} GPUs)")
        
        # Check device mapping
        actor_train_devices = config.get("actor_train", {}).get("device_mapping", [])
        actor_infer_devices = config.get("actor_infer", {}).get("device_mapping", [])
        reference_devices = config.get("reference", {}).get("device_mapping", [])
        
        if isinstance(actor_train_devices, str):
            actor_train_devices = eval(actor_train_devices)
        if isinstance(actor_infer_devices, str):
            actor_infer_devices = eval(actor_infer_devices)
        if isinstance(reference_devices, str):
            reference_devices = eval(reference_devices)
        
        print(f"✓ Device allocation:")
        print(f"  Training: {len(actor_train_devices)} GPUs")
        print(f"  Inference: {len(actor_infer_devices)} GPUs")
        print(f"  Reference: {len(reference_devices)} GPUs")
        
        # Check for optimal async settings
        rollout_batch_size = config.get("rollout_batch_size", 0)
        max_running_requests = config.get("schedule_config", {}).get("max_running_requests", 0)
        
        if rollout_batch_size >= 1024:
            print("✓ Large rollout batch size for async training")
        else:
            print("⚠ Consider increasing rollout_batch_size for better async performance")
        
        if max_running_requests >= 256:
            print("✓ Sufficient max_running_requests for async training")
        else:
            print("⚠ Consider increasing max_running_requests for better async performance")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance settings test failed: {e}")
        return False


def generate_async_recommendations():
    """Generate recommendations for async training optimization"""
    print("\n=== Async Training Optimization Recommendations ===")
    
    recommendations = [
        "🚀 Async Training Optimizations:",
        "",
        "1. **Batch Size Tuning**:",
        "   - rollout_batch_size: 1024+ (current setting is good)",
        "   - Larger batches improve GPU utilization in async mode",
        "",
        "2. **Environment Management**:",
        "   - max_env_num_per_worker: 16+ for better parallelism",
        "   - group_size: 4-8 for optimal batching",
        "",
        "3. **Inference Optimization**:",
        "   - max_num_seqs: 512+ for vLLM async processing",
        "   - tensor_parallel_size: 2+ for large models",
        "",
        "4. **Memory Management**:",
        "   - gpu_memory_utilization: 0.8 (good balance)",
        "   - sequence_length: 10240 (suitable for complex tasks)",
        "",
        "5. **Monitoring**:",
        "   - Watch GPU utilization across all devices",
        "   - Monitor queue lengths in async mode",
        "   - Track throughput vs sync baseline",
        "",
        "6. **Dual Environment Benefits**:",
        "   - Async mode especially beneficial for mixed workloads",
        "   - Search and math tasks can overlap efficiently",
        "   - Better resource utilization with diverse task types"
    ]
    
    for rec in recommendations:
        print(rec)


if __name__ == "__main__":
    print("🚀 Async Training Configuration Test")
    print("=" * 50)
    
    success = True
    
    success &= test_async_configuration()
    success &= test_async_compatibility()
    success &= test_async_performance_settings()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All async configuration tests passed!")
        generate_async_recommendations()
        print("\nReady for async dual environment training!")
    else:
        print("❌ Some async configuration tests failed!")
        print("\nPlease fix the configuration issues before training.")
        sys.exit(1)
