import json
import os.path
import random
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.agentic.rollout.rollout_scheduler import RolloutScheduler
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.configs.model_args import ModelArguments
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.utils import (dump_rollout_render, compute_discounted_returns,
                                         compute_response_level_rewards)
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.functionals import (
    apply_kl_penalty,
    compute_advantage,
    reduce_metrics,
    masked_mean,
    RunningMoments,
    compute_clip_fraction,
    agg_loss,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.agentic.replay_buffer import (
    create_replay_buffer,
    detect_manager_type_from_config,
    BaseReplayBuffer
)
from roll.pipeline.agentic.offpolicy_monitor import (
    compute_offpolicy_metrics,
    validate_replay_batch_fields,
    log_offpolicy_diagnostics
)

logger = get_logger()


class AgenticPipeline(BasePipeline):
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        # Initialize a single tokenizer once and reuse everywhere
        model_path = (
            self.pipeline_config.pretrain or
            getattr(self.pipeline_config.actor_train.model_args, 'model_name_or_path', None)
        )
        if not model_path:
            self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)
        else:
            margs = ModelArguments(model_name_or_path=model_path)
            self.tokenizer = default_tokenizer_provider(margs)
        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=self.pipeline_config.reference.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )

        self.train_rollout_scheduler = RolloutScheduler.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False)).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.train_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="train",
        )
        self.val_rollout_scheduler = RolloutScheduler.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False)).remote(
            config=self.pipeline_config,
            env_manager_config=self.pipeline_config.val_env_manager,
            resource_manager=self.resource_manager,
            infer_cluster=self.actor_infer,
            mode="val",
        )
        
        # 🎯 SETUP PADDING: Provide pipeline's padding strategy to rollout schedulers
        # Padding policy is unified in pipeline, but execution happens in rollout_scheduler
        ray.get([
            self.train_rollout_scheduler.setup_padding.remote(self.tokenizer, self.pipeline_config.sequence_length),
            self.val_rollout_scheduler.setup_padding.remote(self.tokenizer, self.pipeline_config.sequence_length)
        ])
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

        refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = RunningMoments()

        # Initialize replay buffer with new separated architecture
        self.replay_buffer: Optional[BaseReplayBuffer] = None
        rb_cfg = self.pipeline_config.replay
        
        if rb_cfg.enabled:
            # Detect manager type from config
            manager_type = detect_manager_type_from_config(self.pipeline_config)
            
            # Calculate batch size
            batch_size = self.pipeline_config.rollout_batch_size if rb_cfg.use_rollout_batch_size else rb_cfg.minibatch_size
            
            # Always use distributed replay buffer as default
            # It works for both single and multi-machine training
            from roll.agentic.replay_buffer.distributed_buffer_advanced import DistributedReplayBufferWithFaultTolerance

            logger.info("Creating distributed replay buffer with fault tolerance")

            # Get configuration with proper attribute access
            num_shards = getattr(rb_cfg, "num_shards", 4)
            enable_priority = getattr(rb_cfg, "enable_priority", True)
            enable_checkpoint = getattr(rb_cfg, "enable_checkpoint", True)
            enable_rebalancing = getattr(rb_cfg, "enable_rebalancing", False)

            self.replay_buffer = DistributedReplayBufferWithFaultTolerance(
                capacity=rb_cfg.capacity,
                batch_size=batch_size,
                num_shards=num_shards,
                enable_priority=enable_priority,
                enable_checkpoint=enable_checkpoint,
                enable_rebalancing=enable_rebalancing
            )

            logger.info(f"Initialized DistributedReplayBufferWithFaultTolerance for {manager_type} env_manager")
        else:
            self.replay_buffer = None
            # Keep tokenizer for logging/decoding and padding setup even when replay is disabled

    @torch.no_grad()
    def run(self):
        # Calculate tokens-per-second system throughput
        tps_timer = _Timer(window_size=5)

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}
            with tps_timer:
                if self.pipeline_config.adv_estimator == "gae":
                    self.critic.offload_states(blocking=True)
                self.actor_train.offload_states(blocking=True)

                ray.get(self.train_rollout_scheduler.suspend.remote(global_step))
                model_update_metrics: Dict = self.model_update(global_step)
                metrics.update(model_update_metrics)

                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": global_step}

                if global_step % self.pipeline_config.eval_steps == 0:
                    metrics.update(self.val(global_step=global_step))

                ray.get(self.train_rollout_scheduler.resume.remote(global_step))

                with Timer(name="rollout", logger=None) as rollout_timer:
                    batch.meta_info["is_offload_states"] = True
                    batch = ray.get(self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size))
                metrics["time/rollout"] = rollout_timer.last
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                batch.meta_info["global_step"] = global_step

                batch = compute_discounted_returns(batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma)

                # ✨ REPLAY BUFFER INTEGRATION: Mix replay data with fresh rollout data
                batch = self.integrate_replay_buffer_data(batch, global_step)

                # ✅ PADDING HANDLED: Training data padding already applied in rollout_scheduler using pipeline's strategy

                batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                # Debug: batch source & sampling
                try:
                    through_route = 1.0 if batch.meta_info.get("through_route", False) else 0.0
                    sample_method = getattr(self.pipeline_config.replay, 'sample_method', 'lifo') if self.pipeline_config.replay.enabled else 'none'
                    sample_method_code = 1.0 if sample_method == 'lifo' else (0.0 if sample_method == 'uniform' else -1.0)
                    metrics.update({
                        "debug/through_route": through_route,
                        "debug/replay/sample_method_code": sample_method_code,
                        "debug/batch/size": float(batch.batch.batch_size[0])
                    })
                except Exception:
                    pass

                # 当启用 replay 时，下方 off-policy 训练路径会对采样批次重新计算 log_probs/adv。
                # 为避免重复计算，这里仅在 off-policy 关闭时计算。
                with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                    # Pass old_prob_mode to compute_log_probs
                    batch.meta_info["old_prob_mode"] = getattr(self.pipeline_config, "old_prob_mode", "trajectory")
                    ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)
                    ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                    ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                    # CRITICAL FIX: Preserve non_tensor_batch during union operation 
                    # This ensures state_hash and other metadata are not lost
                    preserved_non_tensor_batch = batch.non_tensor_batch
                    batch = batch.union(ref_log_probs)
                    batch.non_tensor_batch = preserved_non_tensor_batch
                    avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                    metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                metrics["time/ref_log_probs_values_reward"] = cal_timer.last

                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    # TODO: use engine log_probs as old_log_probs
                    batch.meta_info["is_offload_states"] = False
                    batch.meta_info["old_prob_mode"] = getattr(self.pipeline_config, "old_prob_mode", "trajectory")
                    old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                    old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        # CRITICAL FIX: Preserve non_tensor_batch during values union operation
                        preserved_non_tensor_batch = batch.non_tensor_batch
                        batch = batch.union(values)
                        batch.non_tensor_batch = preserved_non_tensor_batch
                        metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))
                    batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                    avg_old_log_prob = masked_mean(batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})

                    agg_entropy = agg_loss(
                        loss_mat=old_log_probs.batch["entropy"],
                        loss_mask=batch.batch["response_mask"][:, 1:],
                        loss_agg_mode="token-mean",
                    )
                    metrics.update({"critic/entropy/mean": agg_entropy.item()})

                    metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                metrics["time/old_log_probs_values"] = cal_old_logpb_timer.last

                with Timer(name="adv", logger=None) as timer:
                    # Rewards need to be processed after grouping
                    # We can group by tag(env_type)/traj_group_id(group)/batch(rollout_batch)... to compute rewards / advantages
                    # The compute_response_level_rewards function injects a response_level_rewards key into batch.batch.
                    batch = compute_response_level_rewards(batch=batch, pipeline_config=self.pipeline_config)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    # Debug: reward & mask snapshot before KL/adv
                    try:
                        scores_sum = batch.batch["scores"].sum(dim=-1).mean().detach().item() if "scores" in batch.batch else 0.0
                        penalty_mean = batch.batch["penalty"].mean().detach().item() if "penalty" in batch.batch else 0.0
                        rlr_mean = batch.batch["response_level_rewards"].mean().detach().item() if "response_level_rewards" in batch.batch else 0.0
                        resp_mask = batch.batch["response_mask"][..., 1:].bool() if "response_mask" in batch.batch else None
                        zero_mask_frac = 1.0 - (resp_mask.sum(dim=-1) > 0).float().mean().detach().item() if resp_mask is not None else 0.0
                        metrics.update({
                            "debug/scores_sum/mean": float(scores_sum),
                            "debug/penalty/mean": float(penalty_mean),
                            "debug/response_level_rewards/mean": float(rlr_mean),
                            "debug/response_mask/zero_frac": float(zero_mask_frac),
                        })
                    except Exception:
                        pass

                    if self.pipeline_config.reward_clip:
                        reward_clip_frac = compute_clip_fraction(
                            values=batch.batch["response_level_rewards"],
                            clip_max=self.pipeline_config.reward_clip,
                            clip_min=-self.pipeline_config.reward_clip,
                        )
                        metrics["critic/reward_clip_frac"] = reward_clip_frac
                        batch.batch["response_level_rewards"] = torch.clamp(
                            batch.batch["response_level_rewards"],
                            min=-self.pipeline_config.reward_clip,
                            max=self.pipeline_config.reward_clip,
                        )

                    # 对齐 KL 输入的时间维，防御性裁剪到相同长度（seq_len-1）
                    try:
                        if "old_log_probs" in batch.batch and "ref_log_probs" in batch.batch:
                            t = min(batch.batch["old_log_probs"].shape[1], batch.batch["ref_log_probs"].shape[1])
                            batch.batch["old_log_probs"] = batch.batch["old_log_probs"][:, :t]
                            batch.batch["ref_log_probs"] = batch.batch["ref_log_probs"][:, :t]
                    except Exception:
                        pass
                    # Expand compute_response_level_rewards and add kl_penalty.
                    batch, kl_metrics = apply_kl_penalty(data=batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.pipeline_config.kl_penalty)
                    # KL debug
                    try:
                        metrics.update({
                            "debug/kl/value": float(kl_metrics.get("critic/kl", 0.0)),
                            "debug/kl/beta": float(kl_metrics.get("critic/kl_coef", 0.0)),
                        })
                    except Exception:
                        pass

                    # Is the advantage calculated globally across the batch, or within each group?
                    batch = compute_advantage(
                        data=batch,
                        gamma=self.pipeline_config.gamma,
                        lambd=self.pipeline_config.lambd,
                        adv_estimator=self.pipeline_config.adv_estimator,
                        advantage_clip=self.pipeline_config.advantage_clip,
                        whiten_advantages=self.pipeline_config.whiten_advantages,
                        whiten_rewards=self.pipeline_config.whiten_rewards,
                    )
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    # Debug diagnostics to verify non-zero signals
                    try:
                        resp_mask = batch.batch["response_mask"][:, 1:].bool()
                        mask_tokens = resp_mask.sum().detach().item()
                        tlr = batch.batch.get("token_level_rewards", None)
                        adv = batch.batch.get("advantages", None)
                        rlr = batch.batch.get("response_level_rewards", None)
                        metrics.update({
                            "debug/response_mask_tokens": float(mask_tokens),
                            "debug/token_level_rewards/sum": float(tlr.sum().detach().item()) if tlr is not None else 0.0,
                            "debug/advantages/sum": float(adv.sum().detach().item()) if adv is not None else 0.0,
                            "debug/response_level_rewards/mean": float(rlr.mean().detach().item()) if rlr is not None else 0.0,
                        })
                    except Exception:
                        pass

                metrics.update(kl_metrics)
                metrics["time/adv"] = timer.last

                # Main training step on the current batch (always run)
                if self.pipeline_config.adv_estimator == "gae":
                    critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                if self.pipeline_config.critic_warmup <= global_step:
                    actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                    actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
                    metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

                if self.pipeline_config.adv_estimator == "gae":
                    critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                    metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))

                # Also compute off-policy metrics for fresh batch if it came from replay buffer
                # This helps track how "off-policy" our echo/through-route batch is
                if batch.meta_info.get("through_route", False) and "behavior_log_probs" in batch.batch:
                    fresh_offpolicy_metrics = compute_offpolicy_metrics(
                        current_batch=batch,
                        actor_train_cluster=self.actor_train,
                        old_prob_mode=getattr(self.pipeline_config, "old_prob_mode", "trajectory"),
                        metric_prefix="fresh/offpolicy",
                        pg_clip=self.pipeline_config.pg_clip
                    )
                    metrics.update(fresh_offpolicy_metrics)

                # Optionally perform additional replay buffer training steps
                if self.pipeline_config.replay.enabled:
                    rb_cfg = self.pipeline_config.replay

                    # Only proceed when buffer ready
                    # Check if buffer has enough data for training
                    training_batch_size = (
                        self.pipeline_config.rollout_batch_size if rb_cfg.use_rollout_batch_size else rb_cfg.minibatch_size
                    )
                    if self.replay_buffer.can_sample(batch_size=training_batch_size):

                        all_actor_refs: List[ray.ObjectRef] = []
                        all_critic_refs: List[ray.ObjectRef] = []

                        replay_train_count = 0  # Track successful training steps from replay
                        for step_idx in range(rb_cfg.train_steps_per_env_step):
                            mb = self.replay_buffer.sample_for_training(
                                batch_size=training_batch_size,
                                device='cpu',
                                tokenizer=self.tokenizer,
                                sequence_length=self.pipeline_config.sequence_length,
                                sampling_mode=rb_cfg.sampling_mode,
                                steps_per_episode=rb_cfg.steps_per_episode,
                                sample_method=getattr(rb_cfg, 'sample_method', 'uniform'),
                                candidates_per_group=getattr(rb_cfg, 'candidates_per_group', 1),
                                group_sampling=getattr(rb_cfg, 'group_sampling', 'uniform'),
                            )
                            if mb is None:
                                logger.warning(f"Replay buffer failed to sample batch at step {step_idx} (global_step={global_step})")
                                break

                            # Validate the sampled batch
                            validation = validate_replay_batch_fields(mb)
                            if not validation.get("is_valid", False):
                                logger.warning(f"Invalid replay batch at step {step_idx}: {validation}")

                            replay_train_count += 1

                            # Compute ref/old log_probs and advantages for replay mb
                            mb.meta_info["old_prob_mode"] = getattr(self.pipeline_config, "old_prob_mode", "trajectory")
                            ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(mb, blocking=False)
                            ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                            ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                            mb = mb.union(ref_log_probs)

                            mb.meta_info["is_offload_states"] = False
                            # Only compute old_log_probs if not already provided (fallback to actor_train for stability)
                            if "old_log_probs" not in mb.batch:
                                behavior_old_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(mb, blocking=False)
                                behavior_old = DataProto.materialize_concat(data_refs=behavior_old_refs)
                                mb.batch["old_log_probs"] = behavior_old.batch["log_probs"]

                            # Unified off-policy monitoring for replay training
                            offpolicy_metrics = compute_offpolicy_metrics(
                                current_batch=mb,
                                actor_train_cluster=self.actor_train,
                                old_prob_mode=getattr(self.pipeline_config, "old_prob_mode", "trajectory"),
                                metric_prefix="replay/offpolicy",
                                pg_clip=self.pipeline_config.pg_clip
                            )
                            metrics.update(offpolicy_metrics)

                            # Log diagnostics for debugging
                            if global_step % self.pipeline_config.logging_steps == 0 and offpolicy_metrics:
                                log_offpolicy_diagnostics(
                                    metrics=offpolicy_metrics,
                                    batch=mb,
                                    global_step=global_step,
                                    logger_func=logger.debug
                                )

                            mb = compute_discounted_returns(mb, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma)
                            mb = compute_response_level_rewards(batch=mb, pipeline_config=self.pipeline_config)
                            # Defensive alignment on time dimension (next-token length)
                            try:
                                if "old_log_probs" in mb.batch and "ref_log_probs" in mb.batch:
                                    t = min(mb.batch["old_log_probs"].shape[1], mb.batch["ref_log_probs"].shape[1])
                                    mb.batch["old_log_probs"] = mb.batch["old_log_probs"][:, :t]
                                    mb.batch["ref_log_probs"] = mb.batch["ref_log_probs"][:, :t]
                            except Exception:
                                pass
                            mb, _ = apply_kl_penalty(data=mb, kl_ctrl=self.kl_ctrl, kl_penalty=self.pipeline_config.kl_penalty)
                            mb = compute_advantage(
                                data=mb,
                                gamma=self.pipeline_config.gamma,
                                lambd=self.pipeline_config.lambd,
                                adv_estimator=self.pipeline_config.adv_estimator,
                                advantage_clip=self.pipeline_config.advantage_clip,
                                whiten_advantages=self.pipeline_config.whiten_advantages,
                                whiten_rewards=self.pipeline_config.whiten_rewards,
                            )

                            if self.pipeline_config.adv_estimator == "gae":
                                critic_refs = self.critic.train_step(mb, blocking=False)
                                all_critic_refs.extend(critic_refs)
                            if self.pipeline_config.critic_warmup <= global_step:
                                actor_refs = self.actor_train.train_step(mb, blocking=False)
                                all_actor_refs.extend(actor_refs)

                        if all_actor_refs:
                            actor_metrics = DataProto.materialize_concat(data_refs=all_actor_refs)
                            metrics.update(reduce_metrics(actor_metrics.meta_info.pop("metrics", {})))

                        if all_critic_refs and self.pipeline_config.adv_estimator == "gae":
                            critic_metrics = DataProto.materialize_concat(data_refs=all_critic_refs)
                            metrics.update(reduce_metrics(critic_metrics.meta_info.pop("metrics", {})))

                        buffer_stats = self.replay_buffer.get_stats()
                        metrics.update({
                            "replay_buffer/total_stored": buffer_stats["total_stored"],
                            "replay_buffer/capacity": buffer_stats["capacity"],
                            "replay_buffer/utilization": buffer_stats["utilization"],
                            "replay_buffer/buffer_type": buffer_stats["buffer_type"],
                            "replay/train_steps": replay_train_count,  # Number of successful training steps
                            "replay/train_steps_target": rb_cfg.train_steps_per_env_step,  # Target training steps
                        })

                        # Note: Off-policy monitoring is now done inside the replay training loop
                        # This ensures we compute metrics for each sampled batch, not just the last one
                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

            data_metrics = compute_data_metrics(batch=batch)
            metrics.update(data_metrics)
            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["system/samples"] = (global_step + 1) * self.pipeline_config.rollout_batch_size

            # do ckpt
            self.state.step = global_step
            self.state.log_history.append(metrics)

            self.do_checkpoint(global_step=global_step)

            self.tracker.log(values=metrics, step=global_step)

            if global_step % self.pipeline_config.logging_steps == 0:
                if int(os.environ.get("RAY_PROFILING", "0")):
                    timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                    os.makedirs(timeline_dir, exist_ok=True)
                    ray.timeline(
                        filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                    )

                log_res = []
                batch_grouped = batch.group_by(keys="traj_id")
                for group_name, group_batch in batch_grouped.items():
                    group_batch = group_batch.select_idxs(idxs=[random.choice(range(len(group_batch)))])
                    prompt_mask = group_batch.batch["prompt_mask"]
                    non_prompt_mask = torch.logical_not(group_batch.batch["prompt_mask"])
                    input_ids = group_batch.batch["input_ids"]
                    prompt_ids = torch.where(
                        prompt_mask.bool(), input_ids, torch.full_like(input_ids, self.tokenizer.pad_token_id)
                    )
                    response_ids = torch.where(
                        non_prompt_mask.bool(), input_ids, torch.full_like(input_ids, self.tokenizer.pad_token_id)
                    )
                    prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                    episode_scores = group_batch.non_tensor_batch["episode_scores"].tolist()
                    penalties = group_batch.batch["penalty"].tolist()
                    for prompt, prompt_id, response, response_id, episode_score, penalty in zip(
                            prompts, prompt_ids, responses, response_ids, episode_scores, penalties
                    ):
                        log_res.append(
                            {
                                "prompt": prompt,
                                "response": response,
                                "episode_score": episode_score,
                                "penalty": penalty,
                            }
                        )
                    if len(log_res) >= 10:
                        break
                logger.info(json.dumps(log_res, ensure_ascii=False))
                logger.info(json.dumps(metrics, ensure_ascii=False))

            logger.info(f"pipeline step {global_step} finished")
            global_step += 1
            logger.info(f"epoch {global_step} finished")

        ray.get([
            self.train_rollout_scheduler.stop.remote(),
            self.val_rollout_scheduler.stop.remote()
        ])
        logger.info("pipeline complete!")

    def val(self, global_step):
        batch = DataProto()
        metrics = {}
        batch.meta_info["is_offload_states"] = False
        batch.meta_info["global_step"] = global_step
        eval_batch = ray.get(self.val_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.val_batch_size))
        
        # ✅ PADDING HANDLED: Padding already applied in rollout_scheduler using pipeline's strategy
        eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
        eval_score = get_episode_scores(eval_batch)
        eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
        eval_metrics["score/max"] = torch.max(eval_score).detach().item()
        eval_metrics["score/min"] = torch.min(eval_score).detach().item()

        batch_grouped = eval_batch.group_by(keys="tags")
        for group_name, group_batch in batch_grouped.items():
            traj_group_scores = []
            batch_traj_grouped = group_batch.group_by(keys="traj_group_id")
            for batch_traj_group_name, batch_traj_group in batch_traj_grouped.items():
                traj_group_score = get_episode_scores(batch_traj_group)
                traj_group_scores.append(traj_group_score.mean().item())
            eval_score = torch.tensor(traj_group_scores, dtype=torch.float)
            eval_metrics[f"{group_name}/score/mean"] = torch.mean(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/max"] = torch.max(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/min"] = torch.min(eval_score).detach().item()

        metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})

        if self.pipeline_config.render_save_dir and "frames" in eval_batch.non_tensor_batch:
            self.executor.submit(
                dump_rollout_render,
                save_dir=self.pipeline_config.render_save_dir,
                step=global_step,
                frames=eval_batch.non_tensor_batch["frames"],
                env_ids=eval_batch.non_tensor_batch["env_ids"],
                tags=eval_batch.non_tensor_batch["tags"],
                episode_scores=eval_batch.non_tensor_batch["episode_scores"],
            )
        return metrics

    def adjust_batch(self, data: DataProto, mode="copy") -> DataProto:
        """
        ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/agent_system/multi_turn_rollout/utils.py#L86
        """
        actor_train_train_bsz = self.pipeline_config.actor_train.training_args.per_device_train_batch_size * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps * self.actor_train.dp_size
        actor_train_infer_bsz = self.pipeline_config.actor_train.infer_batch_size * self.actor_train.dp_size
        ref_infer_bsz = self.pipeline_config.reference.infer_batch_size * self.reference.dp_size
        critic_train_bsz = 1
        critic_infer_bsz = 1
        if self.pipeline_config.adv_estimator == "gae":
            critic_train_bsz = self.pipeline_config.critic.training_args.per_device_train_batch_size * self.pipeline_config.critic.training_args.gradient_accumulation_steps * self.critic.dp_size
            critic_infer_bsz = self.pipeline_config.critic.infer_batch_size * self.critic.dp_size

        size_divide = np.lcm.reduce(np.array([actor_train_train_bsz, actor_train_infer_bsz, ref_infer_bsz, critic_infer_bsz, critic_train_bsz])).item()
        batch_size = data.batch.batch_size[0]
        threshold = batch_size % size_divide

        if threshold == 0:
            return data

        if mode == "auto":
            if threshold >= 0.5 * batch_size or  batch_size // size_divide == 0:
                mode = "copy"
            else:
                mode = "delete"

        metrics = data.meta_info.get("metrics", {})
        metrics["system/batch_add_count"] = 0
        metrics["system/batch_remove_count"] = 0
        if mode == "delete":
            remove_indices = np.random.choice(batch_size, threshold, replace=False)
            remove_indices = np.sort(remove_indices)
            keep_mask = np.ones(batch_size, dtype=bool)
            keep_mask[remove_indices] = False
            keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
            tensor_data = data.batch[keep_mask_tensor]
            non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
            adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
            metrics["system/batch_remove_count"] = len(remove_indices)
        elif mode == "copy":
            to_add = size_divide - threshold
            dup_indices = np.random.choice(batch_size, to_add, replace=False)
            dup_proto = data.select_idxs(dup_indices)
            # TODO: set dup_proto response_mask to 0
            adjusted_batch = DataProto.concat([data, dup_proto])
            metrics["system/batch_add_count"] = to_add
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        adjusted_batch.meta_info["metrics"] = metrics

        return adjusted_batch

    def integrate_replay_buffer_data(self, fresh_batch: DataProto, global_step: int) -> DataProto:
        """
        Route training data through the replay buffer (through/echo mode).

        Fresh rollout is first stored into the replay buffer, then we immediately
        sample a replay batch of the same size to be used for training. This keeps
        a stable "data conduit" while maintaining near on-policy freshness.

        Fallback to the original fresh batch when replay cannot return a batch.

        Args:
            fresh_batch: Fresh rollout data from real environments
            global_step: Current training step

        Returns:
            DataProto used for training (prefer replay echo, fallback to fresh)
        """
        if self.replay_buffer is None:
            return fresh_batch
        
        try:
            # 0) Validate fresh batch consistency
            self._validate_batch_consistency(fresh_batch, "fresh_batch")
            
            # 1) Store fresh data to replay buffer
            self.store_fresh_data_to_replay_buffer(fresh_batch, global_step)

            # 2) Sample a replay batch with the SAME size (echo)
            fresh_batch_size = fresh_batch.batch.batch_size[0]
            # Ensure device is not None - fall back to 'cpu' if needed
            target_device = fresh_batch.batch.device if fresh_batch.batch.device is not None else 'cpu'
            with Timer(name="replay_buffer_sample", logger=None) as timer:
                replay_batch = self.replay_buffer.sample_for_training(
                    batch_size=fresh_batch_size,
                    device=target_device,
                    tokenizer=self.tokenizer,
                    sequence_length=self.pipeline_config.sequence_length,
                    sampling_mode=self.pipeline_config.replay.sampling_mode,
                    steps_per_episode=self.pipeline_config.replay.steps_per_episode,
                    sample_method=getattr(self.pipeline_config.replay, 'sample_method', 'lifo'),
                )

            if replay_batch is None:
                logger.debug("Replay buffer returned None, using fresh batch for training")
                return fresh_batch

            # 3) Validate replay batch consistency
            self._validate_batch_consistency(replay_batch, "replay_batch")
            
            # 3.5) Apply compute_discounted_returns for gigpo if needed
            if self.pipeline_config.adv_estimator == "gigpo":
                replay_batch = compute_discounted_returns(
                    replay_batch, 
                    self.pipeline_config.adv_estimator, 
                    self.pipeline_config.step_reward_gamma
                )

            # 4) Attach simple metrics and return replay echo batch
            replay_batch.meta_info.update({
                "replay_buffer_sample_time": timer.last,
                "fresh_batch_size": fresh_batch_size,
                "through_route": True,
            })
            return replay_batch

        except Exception as e:
            logger.error(f"Replay buffer integration failed: {e}")
            logger.debug("Falling back to fresh data only")
            return fresh_batch

    def store_fresh_data_to_replay_buffer(self, fresh_batch: DataProto, global_step: int):
        """Store fresh rollout data to replay buffer for future training."""
        try:
            # Decide old prob compute path and scope
            old_prob_compute = getattr(self.pipeline_config, "old_prob_compute", "trainer")
            old_prob_mode = getattr(self.pipeline_config, "old_prob_mode", "trajectory")
            fresh_batch.meta_info["old_prob_compute"] = old_prob_compute
            fresh_batch.meta_info["old_prob_mode"] = old_prob_mode

            # Compute prompt_length if available (useful for step mode in some envs)
            try:
                if "prompt_mask" in fresh_batch.batch:
                    fresh_batch.batch["prompt_length"] = fresh_batch.batch["prompt_mask"].sum(dim=1)
            except Exception:
                pass

            # Compute behavior policy log_probs according to config
            try:
                if old_prob_compute == "engine" and fresh_batch.batch is not None and "generation_log_probs" in fresh_batch.batch:
                    logger.debug(f"Using engine mode for old_prob_compute with generation_log_probs shape {fresh_batch.batch['generation_log_probs'].shape}")
                    # Use engine-provided log probs
                    if old_prob_mode == "turn" and "prompt_mask" in fresh_batch.batch:
                        # Apply turn mask to engine log probs
                        from roll.utils.turn_mode_utils import create_turn_mode_response_mask
                        
                        # Get the original response mask
                        response_mask = fresh_batch.batch.get("response_mask")
                        prompt_mask = fresh_batch.batch.get("prompt_mask")
                        messages_list = fresh_batch.non_tensor_batch.get("messages_list", None) if hasattr(fresh_batch, 'non_tensor_batch') else None
                        
                        if response_mask is not None:
                            # Create turn-specific mask
                            turn_mask, _ = create_turn_mode_response_mask(
                                response_mask=response_mask,
                                prompt_mask=prompt_mask,
                                messages_list=messages_list
                            )
                            
                            # Apply mask to engine log probs
                            engine_log_probs = fresh_batch.batch["generation_log_probs"]
                            
                            # Ensure shapes match
                            if engine_log_probs.shape != turn_mask.shape:
                                logger.warning(f"Shape mismatch: engine_log_probs {engine_log_probs.shape} vs turn_mask {turn_mask.shape}")
                                # If shapes don't match, try to broadcast or truncate
                                if engine_log_probs.shape[1] > turn_mask.shape[1]:
                                    # Truncate log probs to match mask
                                    engine_log_probs = engine_log_probs[:, :turn_mask.shape[1]]
                                elif engine_log_probs.shape[1] < turn_mask.shape[1]:
                                    # Truncate mask to match log probs
                                    turn_mask = turn_mask[:, :engine_log_probs.shape[1]]
                            
                            masked_log_probs = engine_log_probs * turn_mask.float()
                            fresh_batch.batch["behavior_log_probs"] = masked_log_probs
                        else:
                            # Fallback if no response mask
                            fresh_batch.batch["behavior_log_probs"] = fresh_batch.batch["generation_log_probs"]
                    else:
                        # Trajectory mode: use engine log probs directly
                        fresh_batch.batch["behavior_log_probs"] = fresh_batch.batch["generation_log_probs"]
                else:
                    # Fallback or preferred path: trainer-side recomputation
                    # For step mode, env/data flow ensures response_mask already marks current generation span
                    behavior_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(fresh_batch, blocking=False)
                    behavior = DataProto.materialize_concat(data_refs=behavior_refs)
                    if behavior.batch is not None and "log_probs" in behavior.batch:
                        fresh_batch.batch["behavior_log_probs"] = behavior.batch["log_probs"]
            except Exception as e:
                logger.warning(f"Failed to compute behavior log_probs for replay storage (mode={old_prob_mode}, compute={old_prob_compute}): {e}")

            # Push once per fresh batch (new interface doesn't need tokenizer)
            self.replay_buffer.push_from_dataproto(fresh_batch, global_step)

            # IMPORTANT: Do NOT delete behavior_log_probs here!
            # The replay buffer needs this field for off-policy monitoring.
            # The field will be properly managed by the replay buffer itself.
            
            logger.debug(f"Stored fresh batch to replay buffer (buffer_type={self.replay_buffer.buffer_type})")
            
        except Exception as e:
            logger.error(f"Failed to store fresh data to replay buffer: {e}")
            logger.debug(f"Error details: {str(e)}", exc_info=True)

    def apply_sequence_padding(self, batch: DataProto) -> DataProto:
        """
        Apply unified sequence padding to all tensors in the batch.
        
        This method implements the centralized padding strategy, moving padding logic
        from env_manager and replay_buffer to the pipeline for consistency and efficiency.
        
        Args:
            batch: DataProto with potentially variable-length sequences
            
        Returns:
            DataProto with all sequences padded to pipeline_config.sequence_length
        """
        from roll.utils.functionals import pad_to_length
        
        try:
            sequence_length = self.pipeline_config.sequence_length
            
            # Check if batch is already padded (optimization)
            if hasattr(batch.batch, "input_ids") and batch.batch["input_ids"].size(-1) == sequence_length:
                logger.debug(f"Batch already padded to sequence_length {sequence_length}, skipping")
                return batch
            
            # Get tokenizer for pad_token_id
            tokenizer = self.tokenizer
            
            # Apply padding to all tensor fields that need it
            tensor_fields_to_pad = {
                "input_ids": tokenizer.pad_token_id,
                "attention_mask": 0,
                "position_ids": 0,
                "response_mask": 0,
                "prompt_mask": 0,
                "scores": 0.0,
            }
            
            padded_tensors = {}
            for field_name, pad_value in tensor_fields_to_pad.items():
                if field_name in batch.batch:
                    original_tensor = batch.batch[field_name]
                    padded_tensor = pad_to_length(
                        original_tensor, 
                        length=sequence_length, 
                        pad_value=pad_value
                    )
                    padded_tensors[field_name] = padded_tensor
                    
                    logger.debug(f"Padded {field_name}: {original_tensor.shape} -> {padded_tensor.shape}")
            
            # Update batch with padded tensors
            batch.batch.update(padded_tensors)
            
            # Add padding metadata
            batch.meta_info.update({
                "padding_applied": True,
                "sequence_length": sequence_length,
                "padded_fields": list(padded_tensors.keys())
            })
            
            logger.debug(f"Applied unified padding to {len(padded_tensors)} tensor fields")
            return batch
            
        except Exception as e:
            logger.error(f"Failed to apply sequence padding: {e}")
            logger.debug(f"Error details: {str(e)}", exc_info=True)
            return batch


    def _validate_batch_consistency(self, batch: DataProto, batch_source: str = "unknown"):
        """
        Validate batch data consistency for debugging.
        
        Args:
            batch: DataProto to validate
            batch_source: Source description for logging
        """
        try:
            # Check penalty ranges (only warn on unusual values)
            if "penalty" in batch.batch:
                penalty_mean = batch.batch["penalty"].mean().item()
                if abs(penalty_mean) > 5.0:  # Based on -0.7 normal value
                    logger.warning(f"Unusual penalty values in {batch_source}: mean={penalty_mean:.3f}")
                    
        except Exception as e:
            logger.debug(f"Batch validation failed for {batch_source}: {e}")

def get_episode_scores(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
    scores = []
    for traj_id,  traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["episode_scores"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)

def compute_data_metrics(batch):
    # token_level_scores are per-token scores assigned by the reward model, possibly after normalization/clipping
    # score denotes the raw environment reward
    episode_scores = get_episode_scores(batch)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    advantages = batch.batch["advantages"]
    # fix: https://github.com/volcengine/verl/pull/60
    prompt_mask = batch.batch["prompt_mask"].bool()
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    prompt_lengths = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    returns = batch.batch["returns"]
    non_prompt_mask = torch.logical_not(batch.batch["prompt_mask"]).float()
    penalty: torch.Tensor = batch.batch["penalty"]

    metrics = {
        # score, sequence_score from env
        "critic/score/mean": torch.mean(episode_scores).detach().item(),
        "critic/score/max": torch.max(episode_scores).detach().item(),
        "critic/score/min": torch.min(episode_scores).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # penalty
        "critic/penalty/mean": torch.mean(penalty).detach().item(),
        "critic/penalty/max": torch.max(penalty).detach().item(),
        "critic/penalty/min": torch.min(penalty).detach().item(),
        # adv
        "critic/advantages/mean": masked_mean(advantages, response_mask).detach().item(),
        "critic/advantages/max": torch.max(advantages[response_mask]).detach().item(),
        "critic/advantages/min": torch.min(advantages[response_mask]).detach().item(),
        # returns
        "critic/returns/mean": masked_mean(returns, response_mask).detach().item(),
        "critic/returns/max": torch.max(returns[response_mask]).detach().item(),
        "critic/returns/min": torch.min(returns[response_mask]).detach().item(),
        # response length
        "tokens/response_length/mean": torch.mean(response_length).detach().item(),
        "tokens/response_length/max": torch.max(response_length).detach().item(),
        "tokens/response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "tokens/prompt_length/mean": torch.mean(prompt_lengths).detach().item(),
        "tokens/prompt_length/max": torch.max(prompt_lengths).detach().item(),
        "tokens/prompt_length/min": torch.min(prompt_lengths).detach().item(),
        # non-prompt length
        "tokens/non_prompt_length/mean": torch.mean(non_prompt_mask).detach().item(),
        "tokens/non_prompt_length/max": torch.max(non_prompt_mask).detach().item(),
        "tokens/non_prompt_length/min": torch.min(non_prompt_mask).detach().item(),
    }
    if "values" in batch.batch.keys():
        values = batch.batch["values"]
        # values
        metrics.update(
            {
                "critic/values/mean": masked_mean(values, response_mask).detach().item(),
                "critic/values/max": torch.max(values[response_mask]).detach().item(),
                "critic/values/min": torch.min(values[response_mask]).detach().item(),
            }
        )
    if "episode_rewards_norm" in batch.batch.keys():
        episode_rewards_norm = batch.batch["episode_rewards_norm"]
        step_rewards_norm = batch.batch["step_rewards_norm"]
        metrics.update({
            "critic/episode_rewards_norm/mean": episode_rewards_norm.mean().detach().item(),
            "critic/episode_rewards_norm/max": episode_rewards_norm.max().detach().item(),
            "critic/episode_rewards_norm/min": episode_rewards_norm.min().detach().item(),
            "critic/step_rewards_norm/mean": step_rewards_norm.mean().detach().item(),
            "critic/step_rewards_norm/max": step_rewards_norm.max().detach().item(),
            "critic/step_rewards_norm/min": step_rewards_norm.min().detach().item(),
        })
    return metrics
