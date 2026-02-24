import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray._private import profiling
from tqdm import tqdm

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.functionals import append_to_dict, GenerateRequestType
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class GroupData:
    """Data structure tracking a group of rollouts for the same episode."""
    group_id: int
    episode_id: int
    create_step: int
    rollouts: List[Optional[DataProto]] = field(default_factory=list)
    running_rollouts: int = 0


class GroupQueue:
    """Central episode_id allocation queue for a single group.

    Manages GroupData lifecycle: creation via advance_step,
    episode_id allocation via get_episode_id, rollout collection via put,
    and completed group retrieval via get.
    """

    def __init__(self, group_id: int, progress_bar: tqdm, group_size: int,
                 max_traj_per_env: int, async_generation_ratio: int):
        self.group_id = group_id
        self.group_size = group_size
        self.max_traj_per_env = max_traj_per_env
        self.async_generation_ratio = async_generation_ratio
        self.progress_bar = progress_bar

        self.current_step: Optional[int] = None
        self.next_episode_id: int = 0
        self.groups: Dict[int, GroupData] = {}

        self.progress = asyncio.Event()   # Set when new groups are available
        self.complete = asyncio.Event()   # Set when a group completes
        self.quit = False

    def clear(self, progress_bar: Optional[tqdm] = None):
        """Reset queue state for next training step."""
        self.current_step = None
        self.next_episode_id = 0
        self.groups.clear()
        self.progress = asyncio.Event()
        self.complete = asyncio.Event()
        self.quit = False
        if progress_bar is not None:
            self.progress_bar = progress_bar

    def shutdown(self):
        """Signal shutdown: release all waiters."""
        self.quit = True
        self.groups.clear()
        self.progress.set()
        self.complete.set()

    def advance_group(self, create_step: int):
        """Create a single GroupData entry."""
        assert not self.quit
        self.groups[self.next_episode_id] = GroupData(
            group_id=self.group_id,
            episode_id=self.next_episode_id,
            create_step=create_step
        )
        self.next_episode_id += 1

    def _advance_step(self, create_step: int):
        """Create max_traj_per_env groups for one step."""
        for _ in range(self.max_traj_per_env):
            self.advance_group(create_step)

    def advance_step(self, step: int):
        """Called each training step to create new groups and clean expired ones."""
        if self.current_step is None:
            # First call: create extra groups for async training pipeline
            for _ in range(self.async_generation_ratio):
                self._advance_step(step)
        else:
            # Remove outdated groups from async training
            expired = [eid for eid, g in self.groups.items()
                       if step - g.create_step > self.async_generation_ratio]
            for eid in expired:
                self.groups.pop(eid)

        self.current_step = step
        self._advance_step(step)
        self.progress.set()

    async def get_episode_id(self, env_id: Optional[int] = None) -> Optional[int]:
        """Central episode_id allocation. Returns next available episode_id or None on shutdown."""
        while not self.quit:
            for episode_id, group in self.groups.items():
                if group.running_rollouts < self.group_size:
                    group.running_rollouts += 1
                    return episode_id
            self.progress.clear()
            await self.progress.wait()
        return None

    def put(self, episode_id: int, start_step: int, rollout: Optional[DataProto]):
        """Synchronous put. Appends rollout (or None for skipped trajectory) to the group."""
        if episode_id not in self.groups:
            return  # Expired or already consumed episode
        group = self.groups[episode_id]
        group.rollouts.append(rollout)
        if len(group.rollouts) == self.group_size:
            self.complete.set()
            self.progress_bar.update(self.group_size)

    async def get(self) -> GroupData:
        """Wait and return the first completed group (FIFO by episode_id)."""
        while True:
            for episode_id in list(self.groups.keys()):
                group = self.groups[episode_id]
                if len(group.rollouts) >= self.group_size:
                    self.groups.pop(episode_id)
                    return group
            self.complete.clear()
            await self.complete.wait()


@ray.remote
class GroupQueueManager:
    def __init__(self, config, env_manager_config: EnvManagerConfig, mode):
        self.mode = mode
        self.env_manager_config = env_manager_config
        self.group_size = self.env_manager_config.group_size
        self.progress_bar = tqdm(
            desc=f"{self.mode} rollout progress(trajectory)",
            mininterval=self.env_manager_config.max_traj_per_env
        )
        self.wait_task = None
        self.exception = None
        self.pending_gets = set()

        if self.mode == "train":
            self.async_generation_ratio = config.async_generation_ratio
        else:
            self.async_generation_ratio = 0

        self.group_queue: Dict[int, GroupQueue] = {}
        for rank, rank_env_configs in env_manager_config.env_configs.items():
            for env_id, env_config in rank_env_configs.items():
                group_id = env_config["group_id"]
                if group_id not in self.group_queue:
                    self.group_queue[group_id] = GroupQueue(
                        group_id=group_id,
                        progress_bar=self.progress_bar,
                        group_size=env_manager_config.group_size,
                        max_traj_per_env=env_manager_config.max_traj_per_env,
                        async_generation_ratio=self.async_generation_ratio,
                    )

        # for debug
        self.total = 0
        self.waiting = 0

    def advance_step(self, step: int):
        """Create groups for this training step."""
        for gq in self.group_queue.values():
            gq.advance_step(step)

    async def get_episode_id(self, group_id: int, env_id: Optional[int] = None) -> Optional[int]:
        """Central episode_id allocation for env managers."""
        assert group_id in self.group_queue
        return await self.group_queue[group_id].get_episode_id(env_id)

    def shutdown(self):
        """Signal shutdown to all queues and cancel pending tasks."""
        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()
        for group_queue in self.group_queue.values():
            group_queue.shutdown()

    def clear(self, batch_size=None):
        """Reset all state for next training step."""
        if batch_size is not None:
            self.progress_bar = tqdm(
                total=batch_size,
                desc=f"{self.mode} rollout progress(trajectory)",
                mininterval=self.env_manager_config.max_traj_per_env
            )
        assert self.wait_task is None
        assert self.exception is None
        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()
        for group_queue in self.group_queue.values():
            group_queue.clear(self.progress_bar)

    def put_exception(self, exception):
        self.exception = exception
        if self.wait_task is not None:
            self.wait_task.cancel()

    def _check_exception(self):
        if self.exception is not None:
            raise self.exception

    def put(self, group_id: int, episode_id: int, start_step: int,
            rollout: Optional[DataProto], env_id: Optional[int] = None):
        """Synchronous put. Appends rollout to group."""
        assert group_id in self.group_queue
        self.waiting += 1
        self.group_queue[group_id].put(episode_id, start_step, rollout)
        self.waiting -= 1
        self.total += 1

    async def get_batch(self, batch_size: int) -> List[DataProto]:
        """Collect completed rollout groups until batch_size valid rollouts.

        Filters out None rollouts (from skipped invalid trajectories).
        Returns up to batch_size valid rollouts.
        """
        self._check_exception()
        ret: List[DataProto] = []
        while len(ret) < batch_size:
            # Check if all groups consumed (handles None rollout shortfall)
            remaining = sum(len(gq.groups) for gq in self.group_queue.values())
            if remaining == 0 and not self.pending_gets:
                if len(ret) < batch_size:
                    logger.warning(
                        f"All groups consumed but only {len(ret)}/{batch_size} valid rollouts collected. "
                        f"Some trajectories may have returned None."
                    )
                break

            async def wait_a_episode():
                # Only wait for new episode when there are no pending GroupQueue.get,
                # this way we can avoid starvation of some env.
                if not self.pending_gets:
                    pending = set(
                        asyncio.create_task(self.group_queue[group_id].get())
                        for group_id in self.group_queue
                    )
                else:
                    pending = self.pending_gets
                    self.pending_gets = set()

                while pending and len(ret) < batch_size:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    while done and len(ret) < batch_size:
                        d = done.pop()
                        group: GroupData = await d
                        # Filter None rollouts (from skipped invalid trajectories)
                        group_rollouts = [r for r in group.rollouts if r is not None]
                        self.total -= len(group.rollouts)
                        if group_rollouts:
                            ret.extend(group_rollouts)
                    if done:
                        self.pending_gets.update(done)
                self.pending_gets.update(pending)

            assert self.wait_task is None
            self._check_exception()
            self.wait_task = asyncio.create_task(wait_a_episode())
            try:
                await self.wait_task
            except asyncio.CancelledError:
                self._check_exception()
            self.wait_task = None

        return ret[:batch_size]


@ray.remote
class RolloutScheduler:
    """
    Usage:
        actor_infer
        train_rollout_scheduler = RolloutScheduler(actor_infer)
        val_rollout_scheduler = RolloutScheduler(actor_infer)
        while True:
            ray.get(train_rollout_scheduler.suspend.remote()) # not neccessary in sync traing
            model_update()
            ray.get(train_rollout_scheduler.resume.remote()) # not neccessary in sync traing
            if val:
                ray.get(val_rollout_scheduler.get_batch.remote())
            ray.get(train_rollout_scheduler.get_batch.remote())
            rollout()
        ray.get(train_rollout_scheduler.stop.remote()) # not neccessary in sync traing
    """
    def __init__(self, config, env_manager_config: EnvManagerConfig, resource_manager, infer_cluster, mode, collator=None):
        self.config = config
        self.env_manager_config = env_manager_config
        self.resource_manager = resource_manager
        self.infer_cluster = infer_cluster
        self.mode = mode

        env_num = self.env_manager_config.world_size * self.env_manager_config.max_env_num_per_worker

        self.env_output_queue = GroupQueueManager.options(
            max_concurrency = env_num + 2 # reserve for get_batch + advance_step/get_episode_id
        ).remote(
            self.config,
            self.env_manager_config,
            mode
        )

        self.generate_scheduler = RequestScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                max_concurrency = env_num + 1 # reserve extra one for suspend/resume
            ).remote(infer_cluster=self.infer_cluster, pipeline_config=config)

        self.es_manager: Any = Cluster(
            name=self.env_manager_config.name,
            worker_cls=self.env_manager_config.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.env_manager_config,
        )
        self.es_manager.initialize(
            pipeline_config=self.config,
            generate_scheduler=self.generate_scheduler,
            output_queue=self.env_output_queue,
            collator=collator,
            mode=self.mode,
        )

        self.running = False
        self.rollout_refs = None # only used by async training

        # Padding configuration (set by pipeline)
        self.tokenizer = None
        self.sequence_length = None

    def setup_padding(self, tokenizer, sequence_length):
        """
        Setup padding configuration from pipeline.

        This ensures padding policy is unified in pipeline while execution
        happens in rollout_scheduler to enable successful concat operations.
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        logger.info(f"RolloutScheduler padding setup: sequence_length={sequence_length}")

    def _apply_pipeline_padding(self, data_batch: List[DataProto]) -> List[DataProto]:
        """
        Apply padding to each DataProto in the batch using pipeline's strategy.

        This method applies the same padding logic as pipeline's apply_sequence_padding
        but at the individual DataProto level before concat.
        """
        if not self.tokenizer or not self.sequence_length:
            logger.warning("Padding not configured, skipping padding in rollout_scheduler")
            return data_batch

        from roll.utils.functionals import pad_to_length

        padded_batch = []
        for data_proto in data_batch:
            try:
                # Apply padding to tensor fields
                tensor_fields_to_pad = {
                    "input_ids": self.tokenizer.pad_token_id,
                    "attention_mask": 0,
                    "position_ids": 0,
                    "response_mask": 0,
                    "prompt_mask": 0,
                    "scores": 0.0,
                }

                for field_name, pad_value in tensor_fields_to_pad.items():
                    if field_name in data_proto.batch:
                        original_tensor = data_proto.batch[field_name]
                        padded_tensor = pad_to_length(
                            original_tensor,
                            length=self.sequence_length,
                            pad_value=pad_value
                        )
                        data_proto.batch[field_name] = padded_tensor

                padded_batch.append(data_proto)

            except Exception as e:
                logger.error(f"Failed to apply padding to DataProto: {e}")
                # Fallback: use original data_proto
                padded_batch.append(data_proto)

        return padded_batch

    async def _start_env_manager(self, global_step):
        assert self.running
        if self.config.async_generation_ratio > 0 and self.mode == "train" and self.rollout_refs is None:
            # async training will only call es_manager.run_rollout_loop once
            seed = self.config.seed
            self.rollout_refs: List[ray.ObjectRef] = self.es_manager.run_rollout_loop(None, seed, blocking=False)
        elif self.config.async_generation_ratio == 0 or self.mode != "train":
            # sync and async val will call es_manager.run_rollout_loop every time get_batch called
            assert self.rollout_refs is None
            if self.mode == "train":
                seed = random.randint(0, 1000000)
            else:
                seed = self.config.seed
            self.rollout_refs: List[ray.ObjectRef] = self.es_manager.run_rollout_loop(global_step, seed, blocking=False)

    async def _stop_env_manager(self, batch_size=None):
        assert self.running # generate_scheduler should be running to avoid block env manager
        # First stop env managers (set running=False)
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in self.es_manager.stop(blocking=False)])
        # Then shutdown queue (release any waiting get_episode_id)
        await self.env_output_queue.shutdown.remote()
        # Abort pending generate requests
        await self.generate_scheduler.abort_request.remote()
        # Wait for rollout tasks to complete
        await asyncio.gather(*self.rollout_refs)
        self.rollout_refs = None
        # Reset queue for next step
        await self.env_output_queue.clear.remote(batch_size)

    async def stop(self):
        """
        Stop env manager for async training, called by user!!!
        """
        if self.config.async_generation_ratio > 0 and self.mode == "train" and self.rollout_refs is not None:
            await self._stop_env_manager()
            await self._stop_server()

    async def _stop_server(self):
        if not self.running:
            return
        self.running = False
        stop_server_tasks = [
            asyncio.wrap_future(ref.obj_ref.future())
            for ref in self.infer_cluster.stop_server(blocking=False)
        ]
        if self.config.async_generation_ratio == 0 or self.mode == "train":
            await asyncio.gather(
                self.alive_check_task,
            )
        gen_metrics = await asyncio.gather(*stop_server_tasks)
        gen_metrics = gen_metrics[0]
        return gen_metrics.meta_info.pop("metrics", {})

    async def suspend(self, global_step):
        if self.config.async_generation_ratio == 0 or self.mode != "train":
            return {}

        if not self.running:
            return {}
        # self.running will be set to False in self._stop_server

        await self.generate_scheduler.suspend.remote()
        return await self._stop_server()

    async def _start_server(self, global_step):
        if self.running:
            return
        self.running = True
        data = DataProto()
        data.meta_info["global_step"] = global_step
        data.meta_info["is_offload_states"] = self.config.async_generation_ratio == 0
        await asyncio.gather(
            *[
                asyncio.wrap_future(ref.obj_ref.future())
                for ref in self.infer_cluster.start_server(data, blocking=False)
            ],
        )
        if self.config.async_generation_ratio == 0 or self.mode == "train":
            self.alive_check_task = asyncio.create_task(self.alive_check())

    async def resume(self, global_step):
        if self.config.async_generation_ratio == 0 or self.mode != "train":
            return

        if self.running:
            return
        # self.running will be set to True in self._start_server

        await asyncio.gather(
            self._start_server(global_step),
            *[
                asyncio.wrap_future(ref.future())
                for ref in self.es_manager.update_step(global_step, blocking=False)
            ],
        )
        await self.generate_scheduler.resume.remote()

    async def get_batch(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]

        await self._start_server(global_step)
        # Create groups for this step before starting env managers
        await self.env_output_queue.advance_step.remote(global_step)
        await self._start_env_manager(global_step)

        ref = self.env_output_queue.get_batch.remote(batch_size)
        data_batch: List[DataProto] = await asyncio.wrap_future(ref.future())
        metrics = {}
        [append_to_dict(metrics, meta_info.meta_info["metrics"]) for meta_info in data_batch]

        # Apply padding before concat: use pipeline's padding strategy
        data_batch = self._apply_pipeline_padding(data_batch)

        batch = DataProto.concat(data_batch)

        if self.config.async_generation_ratio == 0 or self.mode != "train":
            await self._stop_env_manager(batch_size)
            # stop server in both async val and sync training, assume train_rollout_manager is suspended or stopped
            actor_infer_metrics = await self._stop_server()
            if self.mode == "train":
                metrics.update(actor_infer_metrics)

        batch.meta_info["metrics"] = metrics
        return batch

    # TODO: do not need alive_check if use async_generate
    async def alive_check(self):
        alive_check_interval = self.config.alive_check_interval
        while self.running:
            await asyncio.sleep(alive_check_interval)
            try:
                outputs: List[DataProto] = await asyncio.gather(
                    *[
                        asyncio.wrap_future(ref.future())
                        for ref in self.infer_cluster.add_request(
                                command=GenerateRequestType.ALIVE_CHECK, data=DataProto(), blocking=False)
                    ]
                )
            except Exception as e:
                if not self.running:
                    return
                self.env_output_queue.put_exception(e)
                return
            request_counts = {key: output.meta_info["request_counts"] for key, output in enumerate(outputs)}
            metrics = {"time": datetime.now().strftime("%Y%m%d-%H%M%S"), "metrics": request_counts}
            logger.debug(f"generate flow: {json.dumps(metrics)}")
