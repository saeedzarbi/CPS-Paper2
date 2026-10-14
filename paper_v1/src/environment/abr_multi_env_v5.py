"""
Multi-Video ABR Environment V5 (Constrained MDP)

Key changes over V3:
1. Rebuffer penalty (6.0) → moderate safety margin above eval weight (4.3)
   V3 used 10.0 which caused overly conservative policies
2. Lyapunov (B_REF=10, BETA=1.0) → V3 used BETA=2.0, too aggressive
3. Buffer-deviation weight (0.05) → V3 used 0.1
4. Chunk 0 smooth penalty = 0 → V3 penalized first chunk relative to
   lowest-bitrate VMAF, creating artificial penalty on episode start
5. Decomposed reward components exposed in info dict (vmaf, smooth_penalty,
   buffer_dev, lyapunov_penalty) → enables Lagrangian wrapper to
   recompute reward with adaptive dual variables
6. obs_dim = 35 (same as V3): +6 download-time ratios
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Union, List


class ABREnv(gym.Env):

    metadata = {'render_modes': ['human']}

    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    CHUNK_DURATION = 4.0

    BUFFER_TARGET = 15.0
    BUFFER_MAX = 30.0

    REBUF_PENALTY_BASE = 6.0
    SMOOTH_PENALTY_WEIGHT = 1.0
    BUFFER_DEV_WEIGHT = 0.05

    B_REF = 10.0
    LYAPUNOV_BETA = 1.0

    MAX_NETWORK_THROUGHPUT = 20000.0
    MIN_NETWORK_THROUGHPUT = 10.0
    MAX_CHUNK_SIZE_BITS = 80_000_000.0
    MAX_DL_RATIO = 5.0

    def __init__(self, video_names: Union[str, List[str]] = 'bigbuckbunny',
                 trace_dir='/root/new/ABR-Awareness/new/data/standardized/train_traces',
                 vmaf_dir='/root/new/ABR-Awareness/new/data/vmaf_scores',
                 siti_dir='/root/new/ABR-Awareness/new/data/content_features',
                 max_chunks=48, random_seed=None,
                 use_future=True, use_lyapunov=True):
        super().__init__()

        self.use_future = use_future
        self.use_lyapunov = use_lyapunov

        if isinstance(video_names, str):
            self.video_names = [video_names]
        else:
            self.video_names = video_names

        self.trace_dir = Path(trace_dir)
        self.vmaf_dir = Path(vmaf_dir)
        self.siti_dir = Path(siti_dir)
        self.max_chunks = max_chunks

        if random_seed is not None:
            self.np_random = np.random.default_rng(random_seed)

        self.video_assets = {}
        self._load_traces()
        self._load_all_video_data()
        self._generate_vbr_profiles()

        self.action_space = spaces.Discrete(len(self.BITRATE_LEVELS))
        obs_dim = 35
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_video_name = None
        self.current_vmaf_scores = {}
        self.current_vmaf_norm = {}
        self.current_si_norm = 0.0
        self.current_ti_norm = 0.0
        self.current_trace = None
        self.current_trace_idx = 0
        self.chunk_idx = 0
        self.buffer_level = 0.0
        self.prev_buffer_level = 0.0
        self.last_bitrate_idx = 0
        self.last_vmaf = 35.0
        self.last_raw_throughput = 2000.0
        self.throughput_history = []

    # ------------------------------------------------------------------ data
    def _load_traces(self):
        trace_files = sorted(self.trace_dir.glob("*.json"))
        if not trace_files:
            self.traces = [{'throughput_kbps': [2000] * 1000}]
        else:
            self.traces = [json.load(open(f)) for f in trace_files]

    def _load_all_video_data(self):
        import pandas as pd

        default_vmaf = {300: 35.0, 750: 58.0, 1200: 74.0,
                        1850: 84.0, 2850: 91.0, 6000: 97.0}
        ladder = [int(b) for b in self.BITRATE_LEVELS]

        # Legacy CSV slugs that map to a v19 canonical name (e.g. tearsofsteel_short).
        legacy_csv_names = {}
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

            if all_vmaf_data is not None and 'video' in all_vmaf_data.columns:
                aliases = legacy_csv_names.get(vid, [])
                names = [vid, *aliases]
                df = all_vmaf_data[all_vmaf_data['video'].isin(names)]
                for _, row in df.iterrows():
                    try:
                        br = int(row['bitrate_kbps'])
                        vmaf_scores[br] = float(row['vmaf'])
                    except (TypeError, ValueError):
                        continue

            # Guarantee every rung exists (avoids KeyError in obs construction).
            for br in ladder:
                vmaf_scores.setdefault(br, default_vmaf[br])

            self.video_assets[vid]['vmaf'] = vmaf_scores
            self.video_assets[vid]['vmaf_norm'] = {
                k: v / 100.0 for k, v in vmaf_scores.items()
            }

            siti_file = self.siti_dir / f"{vid}_siti.json"
            si, ti = 50, 10
            if siti_file.exists():
                try:
                    data = json.load(open(siti_file))
                    si = data.get('mean_si', 50)
                    ti = data.get('mean_ti', 10)
                except Exception:
                    pass

            self.video_assets[vid]['si_norm'] = float(np.clip(si / 150.0, 0, 1))
            self.video_assets[vid]['ti_norm'] = float(np.clip(ti / 70.0, 0, 1))

    # ------------------------------------------------------------------ VBR
    def _generate_vbr_profiles(self):
        for vid in self.video_names:
            rng = np.random.RandomState(abs(hash(vid)) % (2 ** 31))
            n = self.max_chunks + 10

            profile = np.ones(n, dtype=np.float32)
            for i in range(n):
                if i % 4 == 0:
                    profile[i] = rng.uniform(1.5, 2.2)
                else:
                    profile[i] = rng.uniform(0.6, 1.3)

            phase = rng.uniform(0, 2 * np.pi)
            wave = 0.85 + 0.3 * np.sin(
                2 * np.pi * np.arange(n) / 20.0 + phase
            )
            profile = np.clip(profile * wave, 0.4, 2.5).astype(np.float32)
            self.video_assets[vid]['vbr_profile'] = profile

    def _get_vbr_multiplier(self, chunk_idx=None):
        idx = chunk_idx if chunk_idx is not None else self.chunk_idx
        profile = self.video_assets[self.current_video_name]['vbr_profile']
        return float(profile[min(idx, len(profile) - 1)])

    def get_chunk_size_bits(self, bitrate_kbps, chunk_idx):
        vbr_mult = self._get_vbr_multiplier(chunk_idx)
        return bitrate_kbps * 1000 * self.CHUNK_DURATION * vbr_mult

    def get_vbr_profile(self):
        return self.video_assets[self.current_video_name]['vbr_profile']

    def _get_next_chunk_sizes(self):
        if not self.use_future:
            return np.zeros(len(self.BITRATE_LEVELS), dtype=np.float32)

        next_idx = min(self.chunk_idx, self.max_chunks - 1)
        vbr_mult = self._get_vbr_multiplier(next_idx)
        return np.array([
            br * 1000 * self.CHUNK_DURATION * vbr_mult
            for br in self.BITRATE_LEVELS
        ], dtype=np.float32)

    def _get_download_ratios(self):
        if not self.use_future:
            return np.zeros(len(self.BITRATE_LEVELS), dtype=np.float32)

        next_idx = min(self.chunk_idx, self.max_chunks - 1)
        vbr_mult = self._get_vbr_multiplier(next_idx)
        buf = max(self.buffer_level, 0.5)
        tp_bps = self.last_raw_throughput * 1000.0 + 1e-6

        ratios = np.array([
            min((br * 1000 * self.CHUNK_DURATION * vbr_mult) / tp_bps / buf,
                self.MAX_DL_RATIO)
            for br in self.BITRATE_LEVELS
        ], dtype=np.float32)

        return ratios / self.MAX_DL_RATIO

    # ------------------------------------------------------------------ core
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        vid_idx = int(self.np_random.integers(0, len(self.video_names)))
        self.current_video_name = self.video_names[vid_idx]
        assets = self.video_assets[self.current_video_name]

        self.current_vmaf_scores = assets['vmaf']
        self.current_vmaf_norm = assets['vmaf_norm']
        self.current_si_norm = assets['si_norm']
        self.current_ti_norm = assets['ti_norm']

        self.current_trace_idx = int(self.np_random.integers(0, len(self.traces)))
        self.current_trace = self.traces[self.current_trace_idx]

        self.chunk_idx = 0
        self.buffer_level = float(self.np_random.uniform(0.5, 3.0))
        self.prev_buffer_level = self.buffer_level
        self.last_bitrate_idx = 0
        self.last_vmaf = self.current_vmaf_scores[self.BITRATE_LEVELS[0]]

        start_tp = float(self.np_random.uniform(200.0, 5000.0))
        self.last_raw_throughput = start_tp
        log_obs = np.log(start_tp / self.MIN_NETWORK_THROUGHPUT) / \
                  np.log(self.MAX_NETWORK_THROUGHPUT / self.MIN_NETWORK_THROUGHPUT)
        log_obs = float(np.clip(log_obs, 0.0, 1.0))
        self.throughput_history = [log_obs] * 12

        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0

        return self._get_observation(), self._get_info()

    def _get_observation(self):
        tp_obs = np.array(self.throughput_history[-12:], dtype=np.float32)
        buf_obs = np.clip(self.buffer_level / self.BUFFER_MAX, 0, 1)
        buf_trend = np.clip(
            (self.buffer_level - self.prev_buffer_level) / self.CHUNK_DURATION,
            -1.0, 1.0,
        )
        last_br_obs = self.last_bitrate_idx / (len(self.BITRATE_LEVELS) - 1)
        content_obs = np.array(
            [self.current_si_norm, self.current_ti_norm], dtype=np.float32
        )
        vmaf_obs = np.array(
            [self.current_vmaf_norm[br] for br in self.BITRATE_LEVELS],
            dtype=np.float32,
        )

        next_sizes = self._get_next_chunk_sizes()
        sizes_obs = np.clip(next_sizes / self.MAX_CHUNK_SIZE_BITS, 0, 1)

        dl_ratios = self._get_download_ratios()

        return np.concatenate([
            tp_obs,         # 12  throughput history
            [buf_obs],      # 1   buffer level
            [buf_trend],    # 1   buffer trend
            [last_br_obs],  # 1   last bitrate
            content_obs,    # 2   SI / TI
            vmaf_obs,       # 6   VMAF per bitrate
            sizes_obs,      # 6   next chunk sizes
            dl_ratios,      # 6   download-time / buffer ratios
        ]).astype(np.float32)

    def step(self, action):
        bitrate_kbps = self.BITRATE_LEVELS[action]

        vbr_mult = self._get_vbr_multiplier(self.chunk_idx)
        chunk_size_bits = bitrate_kbps * 1000 * self.CHUNK_DURATION * vbr_mult

        trace_tp = self.current_trace['throughput_kbps']
        avail_tp = trace_tp[
            int(self.chunk_idx * self.CHUNK_DURATION) % len(trace_tp)
        ]
        effective_tp = np.clip(
            avail_tp, self.MIN_NETWORK_THROUGHPUT, self.MAX_NETWORK_THROUGHPUT
        )

        download_time = chunk_size_bits / (effective_tp * 1000.0)
        download_time = min(download_time, 60.0)

        rebuffer_time = max(0, download_time - self.buffer_level)
        self.prev_buffer_level = self.buffer_level
        self.buffer_level = max(0, self.buffer_level - download_time) \
                            + self.CHUNK_DURATION
        self.buffer_level = min(self.buffer_level, self.BUFFER_MAX)

        current_vmaf = self.current_vmaf_scores.get(bitrate_kbps, 35.0)
        smooth_pen = 0.0 if self.chunk_idx == 0 else abs(current_vmaf - self.last_vmaf)
        buffer_dev = max(0, self.BUFFER_TARGET - self.buffer_level)

        weighted_rebuf = self.REBUF_PENALTY_BASE * rebuffer_time

        lyap_term = 0.0
        if self.use_lyapunov:
            lyap_term = self.LYAPUNOV_BETA * max(0, self.B_REF - self.buffer_level)

        reward = (current_vmaf
                  - weighted_rebuf
                  - self.SMOOTH_PENALTY_WEIGHT * smooth_pen
                  - self.BUFFER_DEV_WEIGHT * buffer_dev
                  - lyap_term) / 100.0

        self.last_raw_throughput = float(effective_tp)

        log_tp = np.log(
            max(effective_tp, self.MIN_NETWORK_THROUGHPUT)
            / self.MIN_NETWORK_THROUGHPUT
        )
        log_scale = np.log(
            self.MAX_NETWORK_THROUGHPUT / self.MIN_NETWORK_THROUGHPUT
        )
        norm_tp = float(np.clip(log_tp / log_scale, 0.0, 1.0))
        self.throughput_history.append(norm_tp)

        self.last_bitrate_idx = action
        self.last_vmaf = current_vmaf
        self.chunk_idx += 1
        terminated = self.chunk_idx >= self.max_chunks

        self.total_quality += current_vmaf
        self.total_rebuffer += rebuffer_time
        self.total_smooth += smooth_pen

        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'bitrate': bitrate_kbps,
            'throughput': avail_tp,
            'buffer': self.buffer_level,
            'rebuffer': rebuffer_time,
            'vmaf': current_vmaf,
            'smooth_penalty': smooth_pen,
            'buffer_dev': buffer_dev,
            'lyapunov_penalty': lyap_term,
            'reward': float(reward),
            'video_name': self.current_video_name,
        })
        return obs, float(reward), terminated, False, info

    def _get_info(self):
        return {
            'chunk_idx': self.chunk_idx,
            'total_rebuffer': self.total_rebuffer,
            'total_quality': self.total_quality,
            'avg_quality': self.total_quality / max(1, self.chunk_idx),
            'total_smoothness': self.total_smooth,
            'buffer_level': self.buffer_level,
        }

    def render(self):
        pass

    @property
    def vmaf_scores(self):
        return self.current_vmaf_scores

    @property
    def video_name(self):
        return self.current_video_name

    @property
    def trace(self):
        return self.current_trace
