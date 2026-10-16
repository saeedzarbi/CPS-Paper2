"""
Clean-slate wall-clock ABR simulator (replaces abr_multi_env_v5..v18 in Paper 2).

Second-by-second trace integration, optional intra-chunk early abort, log-utility or VMAF rewards.
35-dim observation compatible with existing MPC / BBA / BOLA policies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CleanABREnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    CHUNK_DURATION = 4.0
    BUFFER_MAX = 12.0
    BUFFER_TARGET = 6.0
    B_REF = 4.0
    RTT_SECONDS = 0.04

    MIN_NETWORK_THROUGHPUT = 10.0
    MAX_NETWORK_THROUGHPUT = 50000.0
    MAX_CHUNK_SIZE_BITS = 80_000_000.0
    MAX_DL_RATIO = 5.0

    LOG_REBUF_WEIGHT = 4.3
    LOG_SMOOTH_WEIGHT = 1.0
    VMAF_REBUF_WEIGHT = 100.0
    VMAF_SMOOTH_WEIGHT = 1.0

    REBUF_PENALTY_BASE = 6.0
    SMOOTH_PENALTY_WEIGHT = 1.0
    BUFFER_DEV_WEIGHT = 0.05
    LYAPUNOV_BETA = 1.0

    def __init__(
        self,
        video_names: Union[str, List[str]] = "bigbuckbunny",
        trace_dir: str | Path = "data/standardized/test_traces_v18",
        vmaf_dir: str | Path | None = None,
        siti_dir: str | Path | None = None,
        max_chunks: int = 48,
        random_seed: int | None = None,
        use_future: bool = True,
        use_lyapunov: bool = True,
        reward_mode: str = "log_utility",
        enable_early_abort: bool = False,
        abort_margin: float = 0.5,
        probe_time: float = 0.4,
    ):
        super().__init__()
        from configs.paths import get_paths

        P = get_paths()
        self.reward_mode = str(reward_mode)
        self.enable_early_abort = bool(enable_early_abort)
        self.abort_margin = float(abort_margin)
        self.probe_time = float(probe_time)

        self.use_future = use_future
        self.use_lyapunov = use_lyapunov

        if isinstance(video_names, str):
            self.video_names = [video_names]
        else:
            self.video_names = list(video_names)

        self.trace_dir = Path(trace_dir)
        self.vmaf_dir = Path(vmaf_dir or P["vmaf_scores"])
        self.siti_dir = Path(siti_dir or P["content_features"])
        self.max_chunks = int(max_chunks)

        if random_seed is not None:
            self.np_random = np.random.default_rng(random_seed)

        self.video_assets: dict = {}
        self._load_traces()
        self._load_all_video_data()
        self._generate_vbr_profiles()
        self._load_perchunk_vmaf()

        self.action_space = spaces.Discrete(len(self.BITRATE_LEVELS))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32
        )

        self._reset_runtime()

    def _reset_runtime(self):
        self.current_video_name = None
        self.current_vmaf_scores = {}
        self.current_vmaf_norm = {}
        self.perchunk_vmaf: dict[tuple[int, int], float] = {}
        self.current_si_norm = 0.0
        self.current_ti_norm = 0.0
        self.current_trace = None
        self.current_trace_idx = 0
        self.chunk_idx = 0
        self.buffer_level = 0.0
        self.prev_buffer_level = 0.0
        self.last_bitrate_idx = 0
        self.last_vmaf = 35.0
        self.last_log_u = 0.0
        self.last_raw_throughput = 2000.0
        self.throughput_history: list[float] = []
        self.wall_time = 0.0
        self.trace_offset = 0.0
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
        self.abort_count = 0

    def _load_traces(self):
        trace_files = sorted(self.trace_dir.glob("*.json"))
        loaded = []
        for f in trace_files:
            if f.name.startswith("trace_stats"):
                continue
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
                if "throughput_kbps" in j:
                    loaded.append(j)
            except (json.JSONDecodeError, OSError):
                continue
        self.traces = loaded if loaded else [{"throughput_kbps": [2000.0] * 1000}]

    def _tp_at_wall(self, t: float) -> float:
        trace_tp = self.current_trace["throughput_kbps"]
        idx = int(np.floor(t + self.trace_offset)) % len(trace_tp)
        return float(np.clip(trace_tp[idx], self.MIN_NETWORK_THROUGHPUT, self.MAX_NETWORK_THROUGHPUT))

    def _norm_tp(self, kbps: float) -> float:
        log_tp = np.log(max(kbps, self.MIN_NETWORK_THROUGHPUT) / self.MIN_NETWORK_THROUGHPUT)
        log_scale = np.log(self.MAX_NETWORK_THROUGHPUT / self.MIN_NETWORK_THROUGHPUT)
        return float(np.clip(log_tp / log_scale, 0.0, 1.0))

    def _load_all_video_data(self):
        import pandas as pd

        default_vmaf = {300: 35.0, 750: 58.0, 1200: 74.0, 1850: 84.0, 2850: 91.0, 6000: 97.0}
        ladder = [int(b) for b in self.BITRATE_LEVELS]
        legacy_csv_names: dict = {}
        try:
            from configs.videos import LEGACY_ALIASES

            for old, new in LEGACY_ALIASES.items():
                legacy_csv_names.setdefault(new, []).append(old)
        except Exception:
            pass

        vmaf_file = self.vmaf_dir / "vmaf_summary.csv"
        try:
            all_vmaf_data = pd.read_csv(vmaf_file)
        except Exception:
            all_vmaf_data = None

        for vid in self.video_names:
            self.video_assets[vid] = {}
            vmaf_scores = dict(default_vmaf)
            if all_vmaf_data is not None and "video" in all_vmaf_data.columns:
                aliases = legacy_csv_names.get(vid, [])
                names = [vid, *aliases]
                df = all_vmaf_data[all_vmaf_data["video"].isin(names)]
                for _, row in df.iterrows():
                    try:
                        br = int(row["bitrate_kbps"])
                        vmaf_scores[br] = float(row["vmaf"])
                    except (TypeError, ValueError):
                        continue
            for br in ladder:
                vmaf_scores.setdefault(br, default_vmaf[br])
            self.video_assets[vid]["vmaf"] = vmaf_scores
            self.video_assets[vid]["vmaf_norm"] = {k: v / 100.0 for k, v in vmaf_scores.items()}

            siti_file = self.siti_dir / f"{vid}_siti.json"
            si, ti = 50, 10
            if siti_file.exists():
                try:
                    data = json.loads(siti_file.read_text(encoding="utf-8"))
                    si = data.get("mean_si", 50)
                    ti = data.get("mean_ti", 10)
                except Exception:
                    pass
            self.video_assets[vid]["si_norm"] = float(np.clip(si / 150.0, 0, 1))
            self.video_assets[vid]["ti_norm"] = float(np.clip(ti / 70.0, 0, 1))

    def _load_perchunk_vmaf(self):
        import pandas as pd

        pc = self.vmaf_dir / "vmaf_perchunk.csv"
        if not pc.exists():
            return
        try:
            df = pd.read_csv(pc)
        except Exception:
            return
        if not {"video", "chunk_idx", "bitrate_kbps", "vmaf"}.issubset(df.columns):
            return
        for vid in self.video_names:
            if vid not in self.video_assets:
                continue
            sub = df[df["video"] == vid]
            perchunk = {}
            for _, row in sub.iterrows():
                try:
                    key = (int(row["chunk_idx"]), int(row["bitrate_kbps"]))
                    perchunk[key] = float(row["vmaf"])
                except (TypeError, ValueError):
                    continue
            self.video_assets[vid]["perchunk"] = perchunk

    @staticmethod
    def _stable_video_seed(video_name: str) -> int:
        digest = hashlib.sha256(video_name.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (2**31 - 1)

    def _generate_vbr_profiles(self):
        for vid in self.video_names:
            rng = np.random.RandomState(self._stable_video_seed(vid))
            n = self.max_chunks + 10
            profile = np.ones(n, dtype=np.float32)
            for i in range(n):
                profile[i] = rng.uniform(1.5, 2.2) if i % 4 == 0 else rng.uniform(0.6, 1.3)
            phase = rng.uniform(0, 2 * np.pi)
            wave = 0.85 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 20.0 + phase)
            profile = np.clip(profile * wave, 0.4, 2.5).astype(np.float32)
            self.video_assets[vid]["vbr_profile"] = profile

    def _get_vbr_multiplier(self, chunk_idx=None):
        idx = chunk_idx if chunk_idx is not None else self.chunk_idx
        profile = self.video_assets[self.current_video_name]["vbr_profile"]
        return float(profile[min(idx, len(profile) - 1)])

    def get_chunk_size_bits(self, bitrate_kbps, chunk_idx):
        vbr_mult = self._get_vbr_multiplier(chunk_idx)
        return float(bitrate_kbps) * 1000.0 * self.CHUNK_DURATION * vbr_mult

    def get_vbr_profile(self):
        return self.video_assets[self.current_video_name]["vbr_profile"]

    def _vmaf_for_chunk(self, bitrate_kbps: float, chunk_idx: int) -> float:
        pc = self.video_assets[self.current_video_name].get("perchunk", {})
        key = (chunk_idx, int(bitrate_kbps))
        if key in pc:
            return float(pc[key])
        return float(self.current_vmaf_scores.get(int(bitrate_kbps), 35.0))

    def _get_next_chunk_sizes(self):
        if not self.use_future:
            return np.zeros(len(self.BITRATE_LEVELS), dtype=np.float32)
        next_idx = min(self.chunk_idx, self.max_chunks - 1)
        vbr_mult = self._get_vbr_multiplier(next_idx)
        return np.array(
            [br * 1000 * self.CHUNK_DURATION * vbr_mult for br in self.BITRATE_LEVELS],
            dtype=np.float32,
        )

    def _get_download_ratios(self):
        if not self.use_future:
            return np.zeros(len(self.BITRATE_LEVELS), dtype=np.float32)
        next_idx = min(self.chunk_idx, self.max_chunks - 1)
        vbr_mult = self._get_vbr_multiplier(next_idx)
        buf = max(self.buffer_level, 0.5)
        tp_bps = self.last_raw_throughput * 1000.0 + 1e-6
        ratios = np.array(
            [
                min((br * 1000 * self.CHUNK_DURATION * vbr_mult) / tp_bps / buf, self.MAX_DL_RATIO)
                for br in self.BITRATE_LEVELS
            ],
            dtype=np.float32,
        )
        return ratios / self.MAX_DL_RATIO

    def _simulate_download(self, action: int, chunk_size_bits: float) -> tuple[float, float, float, dict]:
        """Returns (download_time, effective_tp_kbps, rebuffer_time, extra_info)."""
        dt = 0.05
        max_time = 60.0
        bits_remaining = float(chunk_size_bits)
        bits_downloaded = 0.0
        t_start = self.wall_time
        self.wall_time += self.RTT_SECONDS
        elapsed_dl = 0.0
        wasted_bits = 0.0
        aborted = 0

        def integrate(bits_target: float, allow_abort: bool) -> tuple[float, float, bool]:
            nonlocal bits_remaining, bits_downloaded, elapsed_dl, wasted_bits, aborted
            bits_remaining = bits_target
            chunk_bits_done = 0.0
            while bits_remaining > 0 and elapsed_dl < max_time:
                tp = self._tp_at_wall(self.wall_time)
                step_bits = tp * 1000.0 * dt
                take = min(step_bits, bits_remaining)
                bits_remaining -= take
                chunk_bits_done += take
                bits_downloaded += take
                self.wall_time += dt
                elapsed_dl += dt

                if (
                    allow_abort
                    and action > 0
                    and elapsed_dl >= self.probe_time
                    and chunk_bits_done > 0
                ):
                    avg_kbps = chunk_bits_done / max(elapsed_dl, 1e-6) / 1000.0
                    rem_bits = bits_target - chunk_bits_done
                    est_rem_time = rem_bits / max(avg_kbps * 1000.0, 1e-6)
                    buf_rem = max(self.buffer_level - elapsed_dl - self.abort_margin, 0.15)
                    if est_rem_time > buf_rem:
                        wasted_bits += chunk_bits_done
                        aborted = 1
                        return elapsed_dl, avg_kbps, True
            avg_kbps = chunk_bits_done / max(elapsed_dl, 1e-6) / 1000.0 if chunk_bits_done > 0 else self._tp_at_wall(t_start)
            return elapsed_dl, avg_kbps, False

        allow_abort = self.enable_early_abort and action > 0
        elapsed_dl, eff_kbps, did_abort = integrate(bits_remaining, allow_abort)
        if did_abort:
            self.abort_count += 1
            low_bits = self.get_chunk_size_bits(int(self.BITRATE_LEVELS[0]), self.chunk_idx)
            extra_t, eff_kbps2, _ = integrate(low_bits, False)
            elapsed_dl += extra_t
            eff_kbps = (bits_downloaded + low_bits) / max(elapsed_dl, 1e-6) / 1000.0

        total_time = elapsed_dl
        rebuffer_time = max(0.0, total_time - self.buffer_level)
        effective_tp = bits_downloaded / max(total_time, 1e-6) / 1000.0 if bits_downloaded > 0 else eff_kbps
        extra = {"wasted_bits": wasted_bits, "aborted": aborted, "download_time": total_time}
        return total_time, float(effective_tp), float(rebuffer_time), extra

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "rng" in options:
            self.np_random = options["rng"]
        elif seed is not None:
            self.np_random = np.random.default_rng(int(seed))
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()
        vid_idx = int(self.np_random.integers(0, len(self.video_names)))
        self.current_video_name = self.video_names[vid_idx]
        assets = self.video_assets[self.current_video_name]
        self.current_vmaf_scores = assets["vmaf"]
        self.current_vmaf_norm = assets["vmaf_norm"]
        self.current_si_norm = assets["si_norm"]
        self.current_ti_norm = assets["ti_norm"]

        self.current_trace_idx = int(self.np_random.integers(0, len(self.traces)))
        self.current_trace = self.traces[self.current_trace_idx]

        self.chunk_idx = 0
        self.wall_time = 0.0
        self.trace_offset = 0.0
        self.buffer_level = float(self.np_random.uniform(0.5, 3.0))
        self.prev_buffer_level = self.buffer_level
        self.last_bitrate_idx = 0
        self.last_vmaf = self._vmaf_for_chunk(self.BITRATE_LEVELS[0], 0)

        start_tp = self._tp_at_wall(0.0)
        self.last_raw_throughput = start_tp
        log_obs = self._norm_tp(start_tp)
        self.throughput_history = [log_obs] * 12
        self.last_log_u = 0.0

        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
        self.abort_count = 0

        return self._get_observation(), self._get_info()

    def _get_observation(self):
        tp_obs = np.array(self.throughput_history[-12:], dtype=np.float32)
        buf_obs = np.clip(self.buffer_level / self.BUFFER_MAX, 0, 1)
        buf_trend = np.clip((self.buffer_level - self.prev_buffer_level) / self.CHUNK_DURATION, -1.0, 1.0)
        last_br_obs = self.last_bitrate_idx / (len(self.BITRATE_LEVELS) - 1)
        content_obs = np.array([self.current_si_norm, self.current_ti_norm], dtype=np.float32)
        vmaf_obs = np.array([self.current_vmaf_norm[br] for br in self.BITRATE_LEVELS], dtype=np.float32)
        next_sizes = self._get_next_chunk_sizes()
        sizes_obs = np.clip(next_sizes / self.MAX_CHUNK_SIZE_BITS, 0, 1)
        dl_ratios = self._get_download_ratios()
        return np.concatenate(
            [tp_obs, [buf_obs], [buf_trend], [last_br_obs], content_obs, vmaf_obs, sizes_obs, dl_ratios]
        ).astype(np.float32)

    def step(self, action):
        action = int(action)
        bitrate_kbps = int(self.BITRATE_LEVELS[action])
        chunk_size_bits = self.get_chunk_size_bits(bitrate_kbps, self.chunk_idx)

        download_time, effective_tp, rebuffer_time, dl_extra = self._simulate_download(action, chunk_size_bits)

        self.prev_buffer_level = self.buffer_level
        self.buffer_level = max(0.0, self.buffer_level - download_time) + self.CHUNK_DURATION
        self.buffer_level = min(self.buffer_level, self.BUFFER_MAX)

        br0 = float(self.BITRATE_LEVELS[0])
        log_u = float(np.log(max(bitrate_kbps / br0, 1e-6)))
        current_vmaf = self._vmaf_for_chunk(bitrate_kbps, self.chunk_idx)

        if self.reward_mode == "log_utility":
            smooth_pen = 0.0 if self.chunk_idx == 0 else abs(log_u - self.last_log_u)
            reward = log_u - self.LOG_REBUF_WEIGHT * rebuffer_time - self.LOG_SMOOTH_WEIGHT * smooth_pen
            quality_metric = log_u
        else:
            smooth_pen = 0.0 if self.chunk_idx == 0 else abs(current_vmaf - self.last_vmaf)
            buffer_dev = max(0.0, self.BUFFER_TARGET - self.buffer_level)
            lyap_term = self.LYAPUNOV_BETA * max(0.0, self.B_REF - self.buffer_level) if self.use_lyapunov else 0.0
            reward = (
                current_vmaf
                - self.VMAF_REBUF_WEIGHT * rebuffer_time
                - (self.VMAF_SMOOTH_WEIGHT / 100.0) * smooth_pen
                - self.BUFFER_DEV_WEIGHT * buffer_dev
                - lyap_term
            ) / 100.0
            quality_metric = current_vmaf

        self.last_raw_throughput = float(effective_tp)
        self.throughput_history.append(self._norm_tp(effective_tp))
        self.last_bitrate_idx = action
        self.last_vmaf = current_vmaf
        self.last_log_u = log_u
        self.chunk_idx += 1
        terminated = self.chunk_idx >= self.max_chunks

        self.total_quality += quality_metric
        self.total_rebuffer += rebuffer_time
        self.total_smooth += smooth_pen

        info = self._get_info()
        info.update(
            {
                "bitrate": float(bitrate_kbps),
                "throughput": effective_tp,
                "buffer": self.buffer_level,
                "rebuffer": rebuffer_time,
                "vmaf": current_vmaf,
                "log_utility": log_u,
                "smooth_penalty": smooth_pen,
                "reward": float(reward),
                "video_name": self.current_video_name,
                "wall_time": self.wall_time,
                **dl_extra,
            }
        )
        return self._get_observation(), float(reward), terminated, False, info

    def _get_info(self):
        return {
            "chunk_idx": self.chunk_idx,
            "total_rebuffer": self.total_rebuffer,
            "total_quality": self.total_quality,
            "avg_quality": self.total_quality / max(1, self.chunk_idx),
            "total_smoothness": self.total_smooth,
            "buffer_level": self.buffer_level,
            "abort_count": self.abort_count,
        }

    def render(self):
        pass


ABREnv = CleanABREnv
__all__ = ["CleanABREnv", "ABREnv"]
