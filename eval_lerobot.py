"""Evaluate a lerobot policy checkpoint (ACT, Diffusion, …) in the ABC MuJoCo-Warp put-bottles sim.

Reuses abc_minimal's PutBottlesEnv, rollout, video/JSON output, and image
geometry (resize_with_pad). Loads the policy and its pre/post processors from a
lerobot checkpoint directory and adapts them to the sim's `infer(obs)` contract.

Example:
    uv run eval_lerobot.py \
        --checkpoint outputs/act/bottles-act-test2/checkpoints/last \
        --num-worlds 20 \
        --save-video --log-every-chunk
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from abc_minimal.eval_policy import (
    PutBottlesEnv,
    jsonable,
    require_mjwarp,
    resolve_device,
    video_frame,
)
from abc_minimal.config import PutBottlesSimConfig
from abc_minimal.preprocess import resize_with_pad

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE

torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parent


@dataclass
class LeRobotSimEvalConfig:
    """MuJoCo-Warp put-bottles evaluation for a lerobot checkpoint."""

    # Path to a lerobot checkpoint dir. Either the checkpoint root (containing
    # `pretrained_model/`) or the `pretrained_model/` dir itself.
    checkpoint: str

    output_dir: str = field(
        default_factory=lambda: str(ROOT / "outputs" / "sim_eval_lerobot")
    )
    num_worlds: int = 5
    seed: int = 20260511
    num_chunks: int = 120
    execute_chunk_dim: int = 15

    camera_keys: tuple[str, ...] = ("top", "left", "right")
    image_size: tuple[int, int] = (224, 224)  # (H, W) the policy expects
    camera_height: int = 224  # sim render height
    camera_width: int = 224   # sim render width

    device: str = "auto"
    gpu_id: int | None = None
    vanilla_physics: bool = False
    log_every_chunk: bool = False
    save_video: bool = False
    video_fps: int = 30
    video_every_n_actions: int = 1
    prompt: str = "sim put the plastic bottles in the bin"

    scene: PutBottlesSimConfig = field(default_factory=PutBottlesSimConfig)


class LeRobotSimPolicy:
    """Wraps a lerobot policy + processors behind the sim's `infer(obs)` contract."""

    def __init__(self, checkpoint: Path, config: LeRobotSimEvalConfig, device: str):
        self.config = config
        self.device = torch.device(device)
        self.camera_keys = tuple(config.camera_keys)
        self.target_h, self.target_w = config.image_size

        pretrained = checkpoint / "pretrained_model"
        if not pretrained.is_dir():
            # Allow passing the pretrained_model dir directly.
            pretrained = checkpoint
        if not (pretrained / "config.json").is_file():
            raise FileNotFoundError(
                f"No lerobot config.json under {pretrained}. Point --checkpoint at a "
                "checkpoint dir containing pretrained_model/."
            )
        self.pretrained = pretrained

        cfg = PreTrainedConfig.from_pretrained(str(pretrained))
        cfg.device = str(self.device)
        self.policy_cfg = cfg

        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(str(pretrained), config=cfg)
        self.policy.to(self.device)
        self.policy.eval()

        # Load the checkpoint's own pre/post processors (normalization is baked in).
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(pretrained),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        # Validate camera keys against what the policy expects.
        expected = {k.split(".")[-1] for k in cfg.input_features if "images" in k}
        missing = expected - set(self.camera_keys)
        if missing:
            raise ValueError(
                f"Policy expects camera(s) {sorted(missing)} not in --camera-keys {self.camera_keys}"
            )

    def _prep_image(self, img_chw: np.ndarray) -> torch.Tensor:
        """CHW uint8 sim frame -> float [0,1] CHW tensor at target size (no imagenet norm)."""
        x = torch.as_tensor(img_chw).float()
        if x.max() > 1.0:
            x = x / 255.0
        # resize_with_pad works on HWC.
        x = resize_with_pad(x.permute(1, 2, 0), self.target_h, self.target_w)
        return x.permute(2, 0, 1).contiguous()

    @torch.inference_mode()
    def infer(self, obs: dict[str, Any], noise: np.ndarray | None = None) -> np.ndarray:
        # `noise` is unused for ACT/diffusion policies loaded here; kept for the
        # sim's infer(obs, noise=...) contract compatibility.
        state = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32))
        batch: dict[str, Any] = {OBS_STATE: state}
        for cam in self.camera_keys:
            batch[f"observation.images.{cam}"] = self._prep_image(obs["images"][cam])
        batch["task"] = self.config.prompt

        # Preprocessor: batches, moves to device, and normalizes (imagenet for
        # images, dataset mean/std for state) using the checkpoint's own stats.
        batch = self.preprocessor(batch)
        actions = self.policy.predict_action_chunk(batch)  # (B, chunk, action_dim)
        actions = self.postprocessor(actions)              # unnormalize
        return actions[0].to("cpu").numpy()


def local_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p


def run_eval(config: LeRobotSimEvalConfig) -> dict[str, Any]:
    require_mjwarp()
    ckpt_path = local_checkpoint(config.checkpoint)
    device = resolve_device(config.device)
    policy = LeRobotSimPolicy(ckpt_path, config, device)
    env = PutBottlesEnv(
        height=config.camera_height,
        width=config.camera_width,
        camera_keys=config.camera_keys,
        prompt=config.prompt,
        scene=config.scene,
        gpu_id=config.gpu_id,
    )
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    worlds = []

    try:
        for world_index in range(config.num_worlds):
            video = None
            video_path = None
            t0 = time.perf_counter()
            seed = int(config.seed + world_index)
            obs = env.reset(seed=seed)

            if config.save_video:
                import imageio.v2 as imageio

                video_path = out_dir / f"world_{world_index:03d}.mp4"
                video = imageio.get_writer(str(video_path), fps=config.video_fps, macro_block_size=1)
                video.append_data(video_frame(obs["images"], config.camera_keys))

            final_eval = env.evaluate_vanilla() if config.vanilla_physics else env.evaluate()
            steps = 0
            chunk_metrics = []
            try:
                obs_fn = env.obs_vanilla_state if config.vanilla_physics else env.obs
                eval_fn = env.evaluate_vanilla if config.vanilla_physics else env.evaluate
                step_fn = env.step_one_vanilla if config.vanilla_physics else env.step_one
                render_fn = (
                    env.render_cameras_vanilla_state
                    if config.vanilla_physics
                    else env.render_cameras
                )

                t_infer = time.perf_counter()
                actions = policy.infer(obs)
                current_infer_s = time.perf_counter() - t_infer

                for chunk in range(config.num_chunks):
                    t_chunk = time.perf_counter()
                    chunk_infer_s = current_infer_s
                    t_steps = time.perf_counter()
                    for action in actions[: config.execute_chunk_dim]:
                        step_fn(action)
                        final_eval = eval_fn()
                        steps += 1
                        if video is not None and steps % config.video_every_n_actions == 0:
                            video.append_data(video_frame(render_fn(), config.camera_keys))
                        if final_eval["ever_success"]:
                            break
                    steps_s = time.perf_counter() - t_steps
                    if final_eval["ever_success"]:
                        break
                    if chunk + 1 < config.num_chunks:
                        t_obs = time.perf_counter()
                        obs = obs_fn()
                        obs_s = time.perf_counter() - t_obs
                        t_infer = time.perf_counter()
                        actions = policy.infer(obs)
                        current_infer_s = time.perf_counter() - t_infer
                    else:
                        obs_s = 0.0
                    metric = {
                        "chunk": chunk,
                        "infer_s": float(chunk_infer_s),
                        "steps_s": float(steps_s),
                        "obs_render_s": float(obs_s),
                        "wall_s": float(time.perf_counter() - t_chunk),
                        "bottles": int(final_eval["num_bottles_in_bin"]),
                        "max_bottles": int(final_eval["max_bottles_in_bin_so_far"]),
                    }
                    chunk_metrics.append(metric)
                    if config.log_every_chunk:
                        print(
                            f"world={world_index:03d} chunk={chunk:02d} "
                            f"infer={metric['infer_s'] * 1000:.0f}ms "
                            f"steps={metric['steps_s'] * 1000:.0f}ms "
                            f"bottles={final_eval['num_bottles_in_bin']}/{final_eval['num_active_bottles']} "
                            f"success={final_eval['ever_success']}",
                            flush=True,
                        )
            finally:
                if video is not None:
                    video.close()

            world = {
                "world_index": world_index,
                "world_seed": seed,
                "success": bool(final_eval["ever_success"]),
                "final_success": bool(final_eval["success"]),
                "reward": float(final_eval["reward"]),
                "steps": steps,
                "wall_s": time.perf_counter() - t0,
                "chunk_metrics": chunk_metrics,
                "randomization": env.randomization,
                "final_task_eval": final_eval,
                "video_path": str(video_path) if video_path is not None else None,
            }
            worlds.append(world)
            print(
                f"world={world_index:03d} done success={world['success']} "
                f"bottles={final_eval['max_bottles_in_bin_so_far']}/{final_eval['num_active_bottles']} "
                f"steps={steps}",
                flush=True,
            )
    finally:
        env.close()

    success = np.asarray([w["success"] for w in worlds], dtype=bool)
    rewards = np.asarray([w["reward"] for w in worlds], dtype=np.float32)
    max_bottles = np.asarray(
        [w["final_task_eval"]["max_bottles_in_bin_so_far"] for w in worlds],
        dtype=np.float32,
    )
    summary = {
        "format": "abc_minimal_put_bottles_eval/v1",
        "checkpoint": str(ckpt_path),
        "policy_type": policy.policy_cfg.type,
        "prompt": config.prompt,
        "config": asdict(config),
        "resolved_device": device,
        "success_rate": float(success.mean()) if success.size else None,
        "num_success": int(success.sum()),
        "num_worlds": len(worlds),
        "mean_reward": float(rewards.mean()) if rewards.size else None,
        "mean_max_bottles_in_bin": float(max_bottles.mean()) if max_bottles.size else None,
        "worlds": worlds,
    }
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    print(
        f"summary: success_rate={summary['success_rate']} "
        f"num_success={summary['num_success']}/{summary['num_worlds']} "
        f"mean_reward={summary['mean_reward']} "
        f"mean_max_bottles={summary['mean_max_bottles_in_bin']}",
        flush=True,
    )
    print(f"wrote {out_dir / 'summary.json'}", flush=True)
    return summary


def main(config: LeRobotSimEvalConfig) -> None:
    run_eval(config)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(LeRobotSimEvalConfig))
