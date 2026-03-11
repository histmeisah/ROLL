import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import ray

from roll.agentic.rollout.rollout_scheduler import GroupQueueManager

TEST_EXCEPTION = False

class AgenticConfig:
    pass

class EnvManagerConfig:
    pass

async def async_test_GroupQueueManager(rollout_batch_size, async_generation_ratio):
    print(f">>>>>>>>>>>>>>>>>>>>>>>> TEST rollout_batch_size {rollout_batch_size} async_generation_ratio {async_generation_ratio}")
    config = AgenticConfig()
    config.async_generation_ratio = async_generation_ratio

    env_manager_config = EnvManagerConfig()
    env_manager_config.world_size = 1
    env_manager_config.env_groups = 2
    env_manager_config.group_size = 8 # grpo
    train_env_num = env_manager_config.env_groups * env_manager_config.group_size
    env_manager_config.max_env_num_per_worker = train_env_num
    env_manager_config.env_configs = {0: {i: {"group_id": i // env_manager_config.group_size} for i in range(train_env_num)}}

    traj_per_env = (rollout_batch_size + train_env_num - 1) // train_env_num
    env_manager_config.max_traj_per_env = traj_per_env

    env_num = env_manager_config.world_size * env_manager_config.max_env_num_per_worker

    env_output_queue = GroupQueueManager.options(
        max_concurrency = env_num + 2
    ).remote(
        config,
        env_manager_config,
        "train"
    )

    def run_rollout_loop(thread_id, group_id, output_queue):
        """Simulate a single env manager's rollout loop for one step."""
        if TEST_EXCEPTION:
            raise Exception("test exception")

        while True:
            # Get episode_id from central queue (replaces local episode_id tracking)
            episode_id = ray.get(output_queue.get_episode_id.remote(group_id, thread_id))
            if episode_id is None:
                break  # Shutdown signaled, no more episodes available
            rollout = f"rollout_{thread_id}_{episode_id}"
            ray.get(output_queue.put.remote(group_id, episode_id, 0, rollout, thread_id))

    for step in range(10):
        # Each step: advance -> start threads -> get_batch -> shutdown -> clear
        # This mirrors production: _start_env_manager -> get_batch -> _stop_env_manager
        await env_output_queue.advance_step.remote(step)

        with ThreadPoolExecutor(max_workers=env_num) as pool:
            loop = asyncio.get_event_loop()
            try:
                if TEST_EXCEPTION:
                    assert 2 < env_num
                    thread_futures = asyncio.gather(
                        *[loop.run_in_executor(pool, run_rollout_loop, i, i // env_manager_config.group_size, env_output_queue) for i in range(2)]
                    )
                else:
                    thread_futures = asyncio.gather(
                        *[loop.run_in_executor(pool, run_rollout_loop, i, i // env_manager_config.group_size, env_output_queue) for i in range(env_num)]
                    )

                batch = await env_output_queue.get_batch.remote(rollout_batch_size)
                print(f"batch on step({step}): len={len(batch)}")
                assert len(batch) == rollout_batch_size, f"Expected {rollout_batch_size}, got {len(batch)}"

                # Shutdown signals all get_episode_id waiters to return None
                await env_output_queue.shutdown.remote()
                # Wait for all threads to finish (they break on None)
                await thread_futures

            except Exception as e:
                ref = env_output_queue.put_exception.remote(e)
                await asyncio.wrap_future(ref.future())
                raise

        # Clear for next step
        await env_output_queue.clear.remote(rollout_batch_size)

    print(f"  PASSED: batch_size={rollout_batch_size}, async_ratio={async_generation_ratio}")

async def async_test_none_rollouts():
    """Test that None rollouts (from invalid trajectories) are filtered correctly."""
    print(">>>>>>>>>>>>>>>>>>>>>>>> TEST None rollout filtering")
    config = AgenticConfig()
    config.async_generation_ratio = 0

    env_manager_config = EnvManagerConfig()
    env_manager_config.world_size = 1
    env_manager_config.env_groups = 1
    env_manager_config.group_size = 2
    train_env_num = env_manager_config.env_groups * env_manager_config.group_size
    env_manager_config.max_env_num_per_worker = train_env_num
    env_manager_config.env_configs = {0: {i: {"group_id": 0} for i in range(train_env_num)}}
    env_manager_config.max_traj_per_env = 2

    env_output_queue = GroupQueueManager.options(
        max_concurrency = train_env_num + 2
    ).remote(
        config,
        env_manager_config,
        "train"
    )

    def run_env(env_id, output_queue):
        while True:
            episode_id = ray.get(output_queue.get_episode_id.remote(0, env_id))
            if episode_id is None:
                break
            # env_id 0 puts valid rollout, env_id 1 puts None (simulating invalid trajectory)
            rollout = f"valid_rollout_{episode_id}" if env_id == 0 else None
            ray.get(output_queue.put.remote(0, episode_id, 0, rollout, env_id))

    await env_output_queue.advance_step.remote(0)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as pool:
        thread_futures = asyncio.gather(
            *[loop.run_in_executor(pool, run_env, i, env_output_queue) for i in range(train_env_num)]
        )

        batch_size = 2 * 2  # max_traj_per_env * group_size = 4
        batch = await env_output_queue.get_batch.remote(batch_size)
        # With 2 episodes each having 1 valid + 1 None, we get 2 valid rollouts
        print(f"None rollout test: got {len(batch)} valid rollouts (expected 2)")
        assert len(batch) == 2, f"Expected 2 valid rollouts, got {len(batch)}"

        await env_output_queue.shutdown.remote()
        await thread_futures

    await env_output_queue.clear.remote()
    print("None rollout test PASSED")


def test_GroupQueueManager():
    loop = asyncio.get_event_loop()

    # Test sync training (async_generation_ratio = 0)
    loop.run_until_complete(async_test_GroupQueueManager(16, 0))
    loop.run_until_complete(async_test_GroupQueueManager(8, 0))
    loop.run_until_complete(async_test_GroupQueueManager(24, 0))
    loop.run_until_complete(async_test_GroupQueueManager(32, 0))

    # Test None rollout filtering
    loop.run_until_complete(async_test_none_rollouts())

if __name__ == "__main__":
    test_GroupQueueManager()
