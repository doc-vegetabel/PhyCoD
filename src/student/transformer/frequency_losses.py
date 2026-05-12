from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


def make_direction_dof_indices(
    *,
    n_nodes: int,
    component: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    offsets = {
        "x": 0,
        "y": 1,
        "z": 2,
        "rx": 3,
        "ry": 4,
        "rz": 5,
    }
    if component not in offsets:
        raise ValueError(f"Unsupported component: {component}")
    off = offsets[component]
    return torch.tensor([6 * i + off for i in range(int(n_nodes))], dtype=torch.long, device=device)


def _as_btd(response: torch.Tensor) -> torch.Tensor:
    if response.ndim == 2:
        return response.unsqueeze(0)
    if response.ndim == 3:
        return response
    raise ValueError(f"response must be (T,D) or (B,T,D), got shape={tuple(response.shape)}")


def _parse_observations(observations: str | Sequence[str]) -> list[str]:
    if isinstance(observations, str):
        items = [x.strip().lower() for x in observations.split(",") if x.strip()]
    else:
        items = [str(x).strip().lower() for x in observations if str(x).strip()]
    if not items:
        items = ["tip"]
    allowed = {"mean", "all", "full", "tip", "last5", "lastk"}
    bad = [x for x in items if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported observations={bad}. Allowed: {sorted(allowed)}")
    return items


def direction_observation_signal(
    response: torch.Tensor,
    *,
    dof_indices: torch.Tensor,
    observation: str = "mean",
    last_k: int = 5,
) -> torch.Tensor:
    """
    Convert full-order response to a scalar observation time series.

    Args:
        response: (T,D) or (B,T,D)
        dof_indices: direction indices, e.g. all ux or all uy DOFs.
        observation:
            mean/all/full: mean over all direction DOFs;
            tip:          last direction DOF;
            last5/lastk:  mean over last_k direction DOFs.
    Returns:
        signal: (B,T)
    """
    x = _as_btd(response)
    idx = dof_indices.to(device=x.device, dtype=torch.long)
    obs = str(observation).strip().lower()

    if obs in {"mean", "all", "full"}:
        return x[..., idx].mean(dim=-1)
    if obs == "tip":
        return x[..., idx[-1]]
    if obs in {"last5", "lastk"}:
        k = max(1, min(int(last_k), int(idx.numel())))
        return x[..., idx[-k:]].mean(dim=-1)
    raise ValueError(f"Unsupported observation: {observation!r}")


def normalized_power_spectrum(
    signal: torch.Tensor,
    *,
    dt: float,
    freq_min: float = 0.0,
    freq_max: float | None = None,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    signal: (..., T)
    return:
        freqs: (F,)
        power_norm: (..., F)
    """
    if signal.shape[-1] < 4:
        raise ValueError(f"signal length too short: {signal.shape[-1]}")

    x = signal - signal.mean(dim=-1, keepdim=True)
    spec = torch.fft.rfft(x, dim=-1)
    power = spec.real.square() + spec.imag.square()

    freqs = torch.fft.rfftfreq(signal.shape[-1], d=float(dt), device=signal.device)
    freqs = freqs.to(dtype=signal.dtype)

    mask = freqs >= float(freq_min)
    if freq_max is not None:
        mask = mask & (freqs <= float(freq_max))

    freqs = freqs[mask]
    power = power[..., mask]

    power_sum = power.sum(dim=-1, keepdim=True).clamp_min(eps)
    power_norm = power / power_sum
    return freqs, power_norm


def soft_peak_frequency(
    signal: torch.Tensor,
    *,
    dt: float,
    freq_min: float = 0.0,
    freq_max: float | None = None,
    temperature: float = 0.02,
) -> torch.Tensor:
    freqs, power_norm = normalized_power_spectrum(
        signal,
        dt=dt,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    tau = max(float(temperature), 1.0e-8)
    logits = power_norm / tau
    weights = torch.softmax(logits, dim=-1)
    return torch.sum(weights * freqs, dim=-1)


def frequency_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dt: float,
    dof_indices: torch.Tensor,
    freq_min: float = 0.0,
    freq_max: float | None = None,
    peak_temperature: float = 0.02,
    observation: str = "mean",
    last_k: int = 5,
) -> dict[str, torch.Tensor]:
    """
    pred / target: (T, D) or (B, T, D)

    For the selected direction and observation, compare:
      1. normalized power spectrum shape;
      2. soft peak frequency.
    """
    pred_btd = _as_btd(pred)
    target_btd = _as_btd(target)
    if pred_btd.shape != target_btd.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred_btd.shape)} vs {tuple(target_btd.shape)}")

    pred_dir = direction_observation_signal(
        pred_btd,
        dof_indices=dof_indices,
        observation=observation,
        last_k=last_k,
    )
    target_dir = direction_observation_signal(
        target_btd,
        dof_indices=dof_indices,
        observation=observation,
        last_k=last_k,
    )

    _, pred_power = normalized_power_spectrum(
        pred_dir,
        dt=dt,
        freq_min=freq_min,
        freq_max=freq_max,
    )
    _, target_power = normalized_power_spectrum(
        target_dir,
        dt=dt,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    spec_loss = torch.mean((pred_power - target_power) ** 2)

    pred_peak = soft_peak_frequency(
        pred_dir,
        dt=dt,
        freq_min=freq_min,
        freq_max=freq_max,
        temperature=peak_temperature,
    )
    target_peak = soft_peak_frequency(
        target_dir,
        dt=dt,
        freq_min=freq_min,
        freq_max=freq_max,
        temperature=peak_temperature,
    )

    peak_loss = torch.mean((pred_peak - target_peak) ** 2)

    return {
        "spec_loss": spec_loss,
        "peak_loss": peak_loss,
        "pred_peak_hz": pred_peak.mean(),
        "target_peak_hz": target_peak.mean(),
    }


def _zero_like_signal_loss(signal: torch.Tensor) -> torch.Tensor:
    # Keep device/dtype and produce a differentiable scalar when possible.
    return signal.sum() * 0.0


def _candidate_extrema_indices(
    target_1d: torch.Tensor,
    *,
    dt: float,
    start_time: float = 0.0,
    end_time: float | None = None,
    min_distance_seconds: float = 0.3,
    prominence_std: float = 0.15,
    include_troughs: bool = True,
    max_events: int = 16,
) -> list[tuple[int, int]]:
    """
    Return teacher-anchored extrema as a chronological list of (index, kind),
    where kind=+1 is peak and kind=-1 is trough.

    The extrema detection is intentionally non-differentiable because teacher
    anchors are fixed labels. Gradients flow through the predicted soft peak
    time inside each teacher-anchored window.
    """
    y = target_1d.detach()
    T = int(y.numel())
    if T < 5:
        return []

    start_idx = max(1, int(round(float(start_time) / float(dt))))
    if end_time is None:
        end_idx = T - 2
    else:
        end_idx = min(T - 2, int(round(float(end_time) / float(dt))))
    if end_idx <= start_idx:
        return []

    mid = y[1:-1]
    prev = y[:-2]
    nxt = y[2:]

    peaks = ((mid > prev) & (mid >= nxt)).nonzero(as_tuple=False).flatten() + 1
    troughs = ((mid < prev) & (mid <= nxt)).nonzero(as_tuple=False).flatten() + 1

    mean = y[start_idx : end_idx + 1].mean()
    std = y[start_idx : end_idx + 1].std().clamp_min(1.0e-12)
    threshold = float(prominence_std) * std

    events: list[tuple[int, int]] = []
    for idx_t in peaks:
        idx = int(idx_t.item())
        if start_idx <= idx <= end_idx and torch.abs(y[idx] - mean) >= threshold:
            events.append((idx, +1))
    if include_troughs:
        for idx_t in troughs:
            idx = int(idx_t.item())
            if start_idx <= idx <= end_idx and torch.abs(y[idx] - mean) >= threshold:
                events.append((idx, -1))

    events.sort(key=lambda x: x[0])

    min_distance_steps = max(1, int(round(float(min_distance_seconds) / float(dt))))
    filtered: list[tuple[int, int]] = []
    last_idx = -10**9
    for idx, kind in events:
        if idx - last_idx >= min_distance_steps:
            filtered.append((idx, kind))
            last_idx = idx

    if int(max_events) > 0 and len(filtered) > int(max_events):
        filtered = filtered[: int(max_events)]

    return filtered


def peak_time_alignment_loss_from_signal(
    pred_signal: torch.Tensor,
    target_signal: torch.Tensor,
    *,
    dt: float,
    start_time: float = 0.0,
    end_time: float | None = None,
    window_seconds: float = 0.35,
    temperature: float = 0.08,
    min_distance_seconds: float = 0.3,
    prominence_std: float = 0.15,
    max_events: int = 16,
    include_troughs: bool = True,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """
    Teacher-anchored peak/trough time alignment loss.

    1. Detect teacher peaks/troughs in the selected window.
    2. Around each teacher extremum, compute a differentiable softargmax
       time on the predicted signal.
    3. Penalize normalized timing error.
    """
    if pred_signal.ndim == 1:
        pred_signal = pred_signal.unsqueeze(0)
    if target_signal.ndim == 1:
        target_signal = target_signal.unsqueeze(0)
    if pred_signal.shape != target_signal.shape:
        raise ValueError(
            f"pred_signal/target_signal shape mismatch: {tuple(pred_signal.shape)} vs {tuple(target_signal.shape)}"
        )

    B, T = pred_signal.shape
    radius_steps = max(2, int(round(float(window_seconds) / float(dt))))
    tau = max(float(temperature), 1.0e-8)
    window_seconds_safe = max(float(window_seconds), float(dt))

    losses: list[torch.Tensor] = []
    abs_errors: list[torch.Tensor] = []
    n_events = 0

    for b in range(B):
        events = _candidate_extrema_indices(
            target_signal[b],
            dt=dt,
            start_time=start_time,
            end_time=end_time,
            min_distance_seconds=min_distance_seconds,
            prominence_std=prominence_std,
            include_troughs=include_troughs,
            max_events=max_events,
        )
        for anchor_idx, kind in events:
            lo = max(0, anchor_idx - radius_steps)
            hi = min(T - 1, anchor_idx + radius_steps)
            if hi - lo + 1 < 4:
                continue
            pred_win = pred_signal[b, lo : hi + 1]
            score = pred_win if kind > 0 else -pred_win
            score = (score - score.mean()) / score.std().clamp_min(eps)
            weights = torch.softmax(score / tau, dim=0)
            times = torch.arange(lo, hi + 1, device=pred_signal.device, dtype=pred_signal.dtype) * float(dt)
            pred_time = torch.sum(weights * times)
            teacher_time = torch.as_tensor(anchor_idx * float(dt), device=pred_signal.device, dtype=pred_signal.dtype)
            err = pred_time - teacher_time
            losses.append((err / window_seconds_safe) ** 2)
            abs_errors.append(torch.abs(err))
            n_events += 1

    if not losses:
        zero = _zero_like_signal_loss(pred_signal)
        return {
            "loss": zero,
            "mean_abs_time_error_s": zero.detach(),
            "n_events": torch.as_tensor(0.0, device=pred_signal.device, dtype=pred_signal.dtype),
        }

    loss = torch.stack(losses).mean()
    mean_abs_error = torch.stack(abs_errors).mean().detach()
    return {
        "loss": loss,
        "mean_abs_time_error_s": mean_abs_error,
        "n_events": torch.as_tensor(float(n_events), device=pred_signal.device, dtype=pred_signal.dtype),
    }


def local_lag_alignment_loss_from_signal(
    pred_signal: torch.Tensor,
    target_signal: torch.Tensor,
    *,
    dt: float,
    start_time: float = 0.0,
    end_time: float | None = None,
    window_seconds: float = 2.56,
    stride_seconds: float = 1.28,
    max_lag_seconds: float = 0.8,
    temperature: float = 0.05,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """
    Local differentiable lag loss using soft cross-correlation.

    Each local window produces a soft lag. The loss encourages the lag of
    maximum correlation to be zero, directly penalizing local phase delay.
    """
    if pred_signal.ndim == 1:
        pred_signal = pred_signal.unsqueeze(0)
    if target_signal.ndim == 1:
        target_signal = target_signal.unsqueeze(0)
    if pred_signal.shape != target_signal.shape:
        raise ValueError(
            f"pred_signal/target_signal shape mismatch: {tuple(pred_signal.shape)} vs {tuple(target_signal.shape)}"
        )

    B, T = pred_signal.shape
    start_idx = max(0, int(round(float(start_time) / float(dt))))
    end_idx = T if end_time is None else min(T, int(round(float(end_time) / float(dt))) + 1)

    win_steps = max(8, int(round(float(window_seconds) / float(dt))))
    stride_steps = max(1, int(round(float(stride_seconds) / float(dt))))
    max_lag_steps = max(1, int(round(float(max_lag_seconds) / float(dt))))
    tau = max(float(temperature), 1.0e-8)

    if end_idx - start_idx < win_steps:
        zero = _zero_like_signal_loss(pred_signal)
        return {
            "loss": zero,
            "mean_abs_lag_s": zero.detach(),
            "n_windows": torch.as_tensor(0.0, device=pred_signal.device, dtype=pred_signal.dtype),
        }

    losses: list[torch.Tensor] = []
    abs_lags_s: list[torch.Tensor] = []
    n_windows = 0

    lag_values = torch.arange(
        -max_lag_steps,
        max_lag_steps + 1,
        device=pred_signal.device,
        dtype=pred_signal.dtype,
    )

    for b in range(B):
        for lo in range(start_idx, end_idx - win_steps + 1, stride_steps):
            hi = lo + win_steps
            p = pred_signal[b, lo:hi]
            t = target_signal[b, lo:hi]

            p = (p - p.mean()) / p.std().clamp_min(eps)
            t = (t - t.mean()) / t.std().clamp_min(eps)

            corrs: list[torch.Tensor] = []
            valid_lags: list[int] = []
            for lag in range(-max_lag_steps, max_lag_steps + 1):
                if lag < 0:
                    p_l = p[:lag]
                    t_l = t[-lag:]
                elif lag > 0:
                    p_l = p[lag:]
                    t_l = t[:-lag]
                else:
                    p_l = p
                    t_l = t
                if p_l.numel() < 8:
                    continue
                corrs.append(torch.mean(p_l * t_l))
                valid_lags.append(lag)

            if not corrs:
                continue

            corr_tensor = torch.stack(corrs)
            lag_tensor = torch.tensor(valid_lags, device=pred_signal.device, dtype=pred_signal.dtype)
            weights = torch.softmax(corr_tensor / tau, dim=0)
            soft_lag_steps = torch.sum(weights * lag_tensor)
            losses.append((soft_lag_steps / float(max_lag_steps)) ** 2)
            abs_lags_s.append(torch.abs(soft_lag_steps) * float(dt))
            n_windows += 1

    if not losses:
        zero = _zero_like_signal_loss(pred_signal)
        return {
            "loss": zero,
            "mean_abs_lag_s": zero.detach(),
            "n_windows": torch.as_tensor(0.0, device=pred_signal.device, dtype=pred_signal.dtype),
        }

    loss = torch.stack(losses).mean()
    mean_abs_lag_s = torch.stack(abs_lags_s).mean().detach()
    return {
        "loss": loss,
        "mean_abs_lag_s": mean_abs_lag_s,
        "n_windows": torch.as_tensor(float(n_windows), device=pred_signal.device, dtype=pred_signal.dtype),
    }


def _complex_phase_and_amp_loss_from_window(
    pred_win: torch.Tensor,
    target_win: torch.Tensor,
    *,
    dt: float,
    freq_min: float = 0.05,
    freq_max: float | None = 5.0,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """Compare phase of the target-dominant complex spectrum in one local window."""
    if pred_win.ndim != 1 or target_win.ndim != 1:
        raise ValueError("pred_win and target_win must be 1D tensors.")
    if pred_win.shape != target_win.shape:
        raise ValueError(
            f"pred_win/target_win shape mismatch: {tuple(pred_win.shape)} vs {tuple(target_win.shape)}"
        )
    if pred_win.numel() < 8:
        zero = pred_win.sum() * 0.0
        return {
            "phase_loss": zero,
            "amp_guard_loss": zero,
            "rms_amp_ratio": zero.detach(),
            "n_freq_bins": torch.as_tensor(0.0, device=pred_win.device, dtype=pred_win.dtype),
        }

    p = pred_win - pred_win.mean()
    t = target_win - target_win.mean()
    n = int(p.numel())
    window = torch.hann_window(n, periodic=False, device=p.device, dtype=p.dtype)
    p_spec = torch.fft.rfft(p * window, dim=0)
    t_spec = torch.fft.rfft(t * window, dim=0)

    freqs = torch.fft.rfftfreq(n, d=float(dt), device=p.device).to(dtype=p.dtype)
    mask = freqs >= float(freq_min)
    if freq_max is not None:
        mask = mask & (freqs <= float(freq_max))

    if int(mask.sum().item()) <= 0:
        zero = pred_win.sum() * 0.0
        return {
            "phase_loss": zero,
            "amp_guard_loss": zero,
            "rms_amp_ratio": zero.detach(),
            "n_freq_bins": torch.as_tensor(0.0, device=pred_win.device, dtype=pred_win.dtype),
        }

    p_spec = p_spec[mask]
    t_spec = t_spec[mask]
    p_amp = torch.abs(p_spec)
    t_amp = torch.abs(t_spec)

    p_unit = p_spec / p_amp.clamp_min(eps)
    t_unit = t_spec / t_amp.clamp_min(eps)
    target_weights = t_amp.detach()
    target_weights = target_weights / target_weights.sum().clamp_min(eps)

    phase_cos = torch.real(p_unit * torch.conj(t_unit))
    phase_loss = torch.sum(target_weights * (1.0 - phase_cos))

    p_rms = torch.sqrt(torch.mean(p * p).clamp_min(eps))
    t_rms = torch.sqrt(torch.mean(t * t).clamp_min(eps))
    rms_amp_ratio = p_rms / t_rms.clamp_min(eps)
    amp_guard_loss = torch.log(rms_amp_ratio.clamp_min(eps)) ** 2

    return {
        "phase_loss": phase_loss,
        "amp_guard_loss": amp_guard_loss,
        "rms_amp_ratio": rms_amp_ratio.detach(),
        "n_freq_bins": torch.as_tensor(float(mask.sum().item()), device=pred_win.device, dtype=pred_win.dtype),
    }


def _soft_lag_loss_and_score_from_window(
    pred_win: torch.Tensor,
    target_win: torch.Tensor,
    *,
    dt: float,
    max_lag_seconds: float = 0.5,
    temperature: float = 0.04,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """Return differentiable zero-lag loss plus detached hard-mining score."""
    if pred_win.ndim != 1 or target_win.ndim != 1:
        raise ValueError("pred_win and target_win must be 1D tensors.")
    if pred_win.shape != target_win.shape:
        raise ValueError(
            f"pred_win/target_win shape mismatch: {tuple(pred_win.shape)} vs {tuple(target_win.shape)}"
        )

    p = (pred_win - pred_win.mean()) / pred_win.std().clamp_min(eps)
    t = (target_win - target_win.mean()) / target_win.std().clamp_min(eps)
    max_lag_steps = max(1, int(round(float(max_lag_seconds) / float(dt))))
    tau = max(float(temperature), 1.0e-8)

    corrs: list[torch.Tensor] = []
    valid_lags: list[int] = []
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            p_l = p[:lag]
            t_l = t[-lag:]
        elif lag > 0:
            p_l = p[lag:]
            t_l = t[:-lag]
        else:
            p_l = p
            t_l = t
        if p_l.numel() < 8:
            continue
        corrs.append(torch.mean(p_l * t_l))
        valid_lags.append(int(lag))

    if not corrs:
        zero = pred_win.sum() * 0.0
        return {
            "lag_loss": zero,
            "score": zero.detach(),
            "best_abs_lag_s": zero.detach(),
            "best_corr": zero.detach(),
            "corr0": zero.detach(),
        }

    corr_tensor = torch.stack(corrs)
    lag_tensor = torch.tensor(valid_lags, device=pred_win.device, dtype=pred_win.dtype)
    weights = torch.softmax(corr_tensor / tau, dim=0)
    soft_lag_steps = torch.sum(weights * lag_tensor)
    lag_loss = (soft_lag_steps / float(max_lag_steps)) ** 2

    with torch.no_grad():
        corr_detached = corr_tensor.detach()
        best_idx = int(torch.argmax(corr_detached).item())
        best_lag = float(valid_lags[best_idx])
        best_corr = corr_detached[best_idx]
        zero_lag_idx = valid_lags.index(0) if 0 in valid_lags else best_idx
        corr0 = corr_detached[zero_lag_idx]
        lag_score = abs(best_lag) / float(max_lag_steps)
        corr_score = torch.relu(1.0 - best_corr)
        corr0_score = torch.relu(1.0 - corr0)
        score = torch.as_tensor(lag_score, device=pred_win.device, dtype=pred_win.dtype) + corr_score + corr0_score

    return {
        "lag_loss": lag_loss,
        "score": score.detach(),
        "best_abs_lag_s": torch.as_tensor(abs(best_lag) * float(dt), device=pred_win.device, dtype=pred_win.dtype),
        "best_corr": best_corr.to(device=pred_win.device, dtype=pred_win.dtype),
        "corr0": corr0.to(device=pred_win.device, dtype=pred_win.dtype),
    }


def _high_band_power_weight_from_window(
    target_win: torch.Tensor,
    *,
    dt: float,
    freq_min: float = 0.50,
    freq_max: float | None = 1.20,
    threshold: float = 0.20,
    temperature: float = 0.08,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Detached soft weight for windows dominated by sustained high-frequency content."""
    if target_win.numel() < 8:
        return torch.zeros((), device=target_win.device, dtype=target_win.dtype)

    y = target_win.detach() - target_win.detach().mean()
    spec = torch.fft.rfft(y, dim=0)
    power = spec.real.square() + spec.imag.square()
    freqs = torch.fft.rfftfreq(y.numel(), d=float(dt), device=y.device).to(dtype=y.dtype)
    mask = freqs >= float(freq_min)
    if freq_max is not None:
        mask = mask & (freqs <= float(freq_max))
    total = power.sum().clamp_min(eps)
    high_ratio = torch.where(mask.any(), power[mask].sum() / total, torch.zeros_like(total))
    tau = max(float(temperature), 1.0e-8)
    return torch.sigmoid((high_ratio - float(threshold)) / tau).detach()


def _amplitude_weight_from_window(
    target_win: torch.Tensor,
    *,
    reference: float = 0.0,
    strength: float = 0.0,
    power: float = 1.0,
    max_weight: float = 4.0,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Detached multiplicative emphasis for unusually large response windows."""
    if float(strength) <= 0.0 or float(reference) <= 0.0:
        return torch.ones((), device=target_win.device, dtype=target_win.dtype)

    y = target_win.detach() - target_win.detach().mean()
    rms = torch.sqrt(torch.mean(y * y).clamp_min(eps))
    ratio = rms / max(float(reference), eps)
    raw = torch.pow(ratio.clamp_min(eps), float(power))
    emphasized = torch.clamp(raw, min=1.0, max=max(float(max_weight), 1.0))
    return (1.0 + float(strength) * (emphasized - 1.0)).detach()


def _soft_lag_steps_from_window(
    pred_win: torch.Tensor,
    target_win: torch.Tensor,
    *,
    dt: float,
    max_lag_seconds: float = 0.50,
    temperature: float = 0.04,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable soft lag plus detached hard-lag/corr diagnostics for one window."""
    p = (pred_win - pred_win.mean()) / pred_win.std().clamp_min(eps)
    t = (target_win - target_win.mean()) / target_win.std().clamp_min(eps)
    max_lag_steps = max(1, int(round(float(max_lag_seconds) / float(dt))))
    tau = max(float(temperature), 1.0e-8)

    corrs: list[torch.Tensor] = []
    valid_lags: list[int] = []
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            p_l = p[:lag]
            t_l = t[-lag:]
        elif lag > 0:
            p_l = p[lag:]
            t_l = t[:-lag]
        else:
            p_l = p
            t_l = t
        if p_l.numel() < 8:
            continue
        corrs.append(torch.mean(p_l * t_l))
        valid_lags.append(int(lag))

    if not corrs:
        zero = pred_win.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    corr_tensor = torch.stack(corrs)
    lag_tensor = torch.tensor(valid_lags, device=pred_win.device, dtype=pred_win.dtype)
    weights = torch.softmax(corr_tensor / tau, dim=0)
    soft_lag_steps = torch.sum(weights * lag_tensor)

    with torch.no_grad():
        best_idx = int(torch.argmax(corr_tensor.detach()).item())
        best_abs_lag_s = torch.as_tensor(
            abs(float(valid_lags[best_idx])) * float(dt),
            device=pred_win.device,
            dtype=pred_win.dtype,
        )
        best_corr = corr_tensor.detach()[best_idx].to(device=pred_win.device, dtype=pred_win.dtype)

    return soft_lag_steps, best_abs_lag_s, best_corr


def phase_drift_rate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dt: float,
    dof_indices: torch.Tensor,
    observations: str | Sequence[str] = "tip,last5",
    last_k: int = 5,
    start_time: float = 0.0,
    end_time: float | None = None,
    window_seconds: float = 1.54,
    stride_seconds: float = 0.32,
    max_lag_seconds: float = 0.50,
    lag_temperature: float = 0.04,
    freq_min: float = 0.50,
    freq_max: float | None = 1.20,
    high_power_threshold: float = 0.20,
    high_power_temperature: float = 0.08,
    amplitude_reference: float = 0.0,
    amplitude_weight: float = 0.0,
    amplitude_power: float = 1.0,
    amplitude_max_weight: float = 4.0,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """
    Penalize accumulated phase drift in high-frequency local windows.

    The loss first computes a soft lag for consecutive windows, then penalizes:
      1. non-zero lag in high-frequency windows;
      2. changes in lag between consecutive high-frequency windows.

    High-frequency windows are selected by target-window spectral content, not
    by case name. This directly targets late high-frequency phase drift.
    """
    pred_btd = _as_btd(pred)
    target_btd = _as_btd(target)
    if pred_btd.shape != target_btd.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred_btd.shape)} vs {tuple(target_btd.shape)}")

    B, T, _ = pred_btd.shape
    obs_list = _parse_observations(observations)
    idx = dof_indices.to(device=pred_btd.device, dtype=torch.long)

    start_idx = max(0, int(round(float(start_time) / float(dt))))
    end_idx = T if end_time is None else min(T, int(round(float(end_time) / float(dt))) + 1)
    win_steps = max(8, int(round(float(window_seconds) / float(dt))))
    stride_steps = max(1, int(round(float(stride_seconds) / float(dt))))
    max_lag_steps = max(1, int(round(float(max_lag_seconds) / float(dt))))
    zero = _zero_like_signal_loss(pred_btd)

    if end_idx - start_idx < win_steps:
        return {
            "loss": zero,
            "lag_loss": zero,
            "drift_loss": zero,
            "mean_abs_lag_s": zero.detach(),
            "mean_abs_dlag_s": zero.detach(),
            "high_weight_mean": zero.detach(),
            "amplitude_weight_mean": zero.detach(),
            "combined_weight_mean": zero.detach(),
            "n_windows": torch.as_tensor(0.0, device=pred_btd.device, dtype=pred_btd.dtype),
        }

    lag_loss_terms: list[torch.Tensor] = []
    drift_loss_terms: list[torch.Tensor] = []
    abs_lag_terms: list[torch.Tensor] = []
    abs_dlag_terms: list[torch.Tensor] = []
    high_weights_all: list[torch.Tensor] = []
    amplitude_weights_all: list[torch.Tensor] = []
    combined_weights_all: list[torch.Tensor] = []
    n_windows = 0

    for obs in obs_list:
        pred_sig = direction_observation_signal(
            pred_btd,
            dof_indices=idx,
            observation=obs,
            last_k=last_k,
        )
        target_sig = direction_observation_signal(
            target_btd,
            dof_indices=idx,
            observation=obs,
            last_k=last_k,
        )
        for b in range(B):
            window_lags: list[torch.Tensor] = []
            window_weights: list[torch.Tensor] = []
            for lo in range(start_idx, end_idx - win_steps + 1, stride_steps):
                hi = lo + win_steps
                p = pred_sig[b, lo:hi]
                t = target_sig[b, lo:hi]
                soft_lag_steps, hard_abs_lag_s, _ = _soft_lag_steps_from_window(
                    p,
                    t,
                    dt=dt,
                    max_lag_seconds=max_lag_seconds,
                    temperature=lag_temperature,
                    eps=eps,
                )
                weight = _high_band_power_weight_from_window(
                    t,
                    dt=dt,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    threshold=high_power_threshold,
                    temperature=high_power_temperature,
                    eps=eps,
                )
                amp_weight = _amplitude_weight_from_window(
                    t,
                    reference=amplitude_reference,
                    strength=amplitude_weight,
                    power=amplitude_power,
                    max_weight=amplitude_max_weight,
                    eps=eps,
                )
                combined_weight = (weight * amp_weight).detach()
                lag_loss_terms.append(combined_weight * (soft_lag_steps / float(max_lag_steps)) ** 2)
                abs_lag_terms.append(combined_weight * hard_abs_lag_s)
                high_weights_all.append(weight)
                amplitude_weights_all.append(amp_weight)
                combined_weights_all.append(combined_weight)
                window_lags.append(soft_lag_steps)
                window_weights.append(combined_weight)
                n_windows += 1

            if len(window_lags) >= 2:
                lag_t = torch.stack(window_lags)
                weight_t = torch.stack(window_weights)
                dlag = lag_t[1:] - lag_t[:-1]
                pair_weight = torch.minimum(weight_t[1:], weight_t[:-1]).detach()
                drift_loss_terms.append(torch.mean(pair_weight * (dlag / float(max_lag_steps)) ** 2))
                abs_dlag_terms.append(torch.mean(pair_weight * torch.abs(dlag) * float(dt)))

    if not lag_loss_terms:
        return {
            "loss": zero,
            "lag_loss": zero,
            "drift_loss": zero,
            "mean_abs_lag_s": zero.detach(),
            "mean_abs_dlag_s": zero.detach(),
            "high_weight_mean": zero.detach(),
            "amplitude_weight_mean": zero.detach(),
            "combined_weight_mean": zero.detach(),
            "n_windows": torch.as_tensor(0.0, device=pred_btd.device, dtype=pred_btd.dtype),
        }

    lag_loss = torch.stack(lag_loss_terms).mean()
    drift_loss = torch.stack(drift_loss_terms).mean() if drift_loss_terms else zero
    loss = lag_loss + drift_loss
    mean_abs_lag_s = torch.stack(abs_lag_terms).mean().detach()
    mean_abs_dlag_s = torch.stack(abs_dlag_terms).mean().detach() if abs_dlag_terms else zero.detach()
    high_weight_mean = torch.stack(high_weights_all).mean().detach()
    amplitude_weight_mean = torch.stack(amplitude_weights_all).mean().detach()
    combined_weight_mean = torch.stack(combined_weights_all).mean().detach()

    return {
        "loss": loss,
        "lag_loss": lag_loss,
        "drift_loss": drift_loss,
        "mean_abs_lag_s": mean_abs_lag_s,
        "mean_abs_dlag_s": mean_abs_dlag_s,
        "high_weight_mean": high_weight_mean,
        "amplitude_weight_mean": amplitude_weight_mean,
        "combined_weight_mean": combined_weight_mean,
        "n_windows": torch.as_tensor(float(n_windows), device=pred_btd.device, dtype=pred_btd.dtype),
    }


def adaptive_phase_window_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dt: float,
    dof_indices: torch.Tensor,
    observations: str | Sequence[str] = "tip,last5",
    last_k: int = 5,
    gate: torch.Tensor | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
    window_seconds: float = 1.92,
    stride_seconds: float = 0.64,
    top_k: int = 4,
    score_temperature: float = 0.25,
    gate_target_score_ref: float = 0.12,
    max_lag_seconds: float = 0.5,
    lag_temperature: float = 0.04,
    freq_min: float = 0.05,
    freq_max: float | None = 5.0,
    amplitude_reference: float = 0.0,
    amplitude_weight: float = 0.0,
    amplitude_power: float = 1.0,
    amplitude_max_weight: float = 4.0,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """
    Adaptive phase-window hard mining loss.

    The loss scans the full response, scores each local window by detached
    phase drift indicators, then applies differentiable lag/complex-phase
    losses mostly on the highest-score windows. When a phase gate is provided,
    it also returns a soft gate-alignment loss so the fast branch learns to
    open where the current rollout shows local phase drift. The gate target is
    calibrated by ``gate_target_score_ref`` so a meaningful local drift score can
    ask for a clearly active gate instead of being compressed to a near-zero
    target.
    """
    pred_btd = _as_btd(pred)
    target_btd = _as_btd(target)
    if pred_btd.shape != target_btd.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred_btd.shape)} vs {tuple(target_btd.shape)}")

    B, T, _ = pred_btd.shape
    obs_list = _parse_observations(observations)
    idx = dof_indices.to(device=pred_btd.device, dtype=torch.long)

    start_idx = max(0, int(round(float(start_time) / float(dt))))
    end_idx = T if end_time is None else min(T, int(round(float(end_time) / float(dt))) + 1)
    win_steps = max(8, int(round(float(window_seconds) / float(dt))))
    stride_steps = max(1, int(round(float(stride_seconds) / float(dt))))

    zero = _zero_like_signal_loss(pred_btd)
    if end_idx - start_idx < win_steps:
        return {
            "loss": zero,
            "lag_loss": zero,
            "complex_phase_loss": zero,
            "complex_amp_guard_loss": zero,
            "gate_align_loss": zero,
            "score_mean": zero.detach(),
            "score_max": zero.detach(),
            "amplitude_weight_mean": zero.detach(),
            "amplitude_weight_max": zero.detach(),
            "best_abs_lag_s_mean": zero.detach(),
            "best_corr_mean": zero.detach(),
            "corr0_mean": zero.detach(),
            "selected_t_start_mean": zero.detach(),
            "selected_t_start_min": zero.detach(),
            "selected_t_start_max": zero.detach(),
            "selected_gate_mean": zero.detach(),
            "gate_target_mean": zero.detach(),
            "n_windows": torch.as_tensor(0.0, device=pred_btd.device, dtype=pred_btd.dtype),
            "n_selected_windows": torch.as_tensor(0.0, device=pred_btd.device, dtype=pred_btd.dtype),
        }

    gate_btg = None
    if gate is not None:
        gate_btg = gate
        if gate_btg.ndim == 2:
            gate_btg = gate_btg.unsqueeze(-1)
        if gate_btg.ndim != 3:
            raise ValueError(f"gate must have shape (B,T) or (B,T,G), got {tuple(gate.shape)}")
        if tuple(gate_btg.shape[:2]) != (B, T):
            raise ValueError(
                f"gate prefix shape mismatch: expected {(B,T)}, got {tuple(gate_btg.shape[:2])}."
            )
        gate_btg = gate_btg.to(device=pred_btd.device, dtype=pred_btd.dtype)

    lag_losses: list[torch.Tensor] = []
    phase_losses: list[torch.Tensor] = []
    amp_losses: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    best_abs_lags: list[torch.Tensor] = []
    best_corrs: list[torch.Tensor] = []
    corr0s: list[torch.Tensor] = []
    starts: list[float] = []
    gate_means: list[torch.Tensor] = []
    amp_weights: list[torch.Tensor] = []

    for obs in obs_list:
        pred_sig = direction_observation_signal(
            pred_btd,
            dof_indices=idx,
            observation=obs,
            last_k=last_k,
        )
        target_sig = direction_observation_signal(
            target_btd,
            dof_indices=idx,
            observation=obs,
            last_k=last_k,
        )

        for b in range(B):
            for lo in range(start_idx, end_idx - win_steps + 1, stride_steps):
                hi = lo + win_steps
                p = pred_sig[b, lo:hi]
                t = target_sig[b, lo:hi]

                lag = _soft_lag_loss_and_score_from_window(
                    p,
                    t,
                    dt=dt,
                    max_lag_seconds=max_lag_seconds,
                    temperature=lag_temperature,
                    eps=eps,
                )
                complex_loss = _complex_phase_and_amp_loss_from_window(
                    p,
                    t,
                    dt=dt,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    eps=eps,
                )
                amp_weight = _amplitude_weight_from_window(
                    t,
                    reference=amplitude_reference,
                    strength=amplitude_weight,
                    power=amplitude_power,
                    max_weight=amplitude_max_weight,
                    eps=eps,
                )

                lag_losses.append(lag["lag_loss"])
                phase_losses.append(complex_loss["phase_loss"])
                amp_losses.append(complex_loss["amp_guard_loss"])
                scores.append(lag["score"] * amp_weight)
                best_abs_lags.append(lag["best_abs_lag_s"].detach())
                best_corrs.append(lag["best_corr"].detach())
                corr0s.append(lag["corr0"].detach())
                starts.append(float(lo) * float(dt))
                amp_weights.append(amp_weight)
                if gate_btg is not None:
                    gate_means.append(gate_btg[b, lo:hi, :].mean())

    if not lag_losses:
        return {
            "loss": zero,
            "lag_loss": zero,
            "complex_phase_loss": zero,
            "complex_amp_guard_loss": zero,
            "gate_align_loss": zero,
            "score_mean": zero.detach(),
            "score_max": zero.detach(),
            "amplitude_weight_mean": zero.detach(),
            "amplitude_weight_max": zero.detach(),
            "best_abs_lag_s_mean": zero.detach(),
            "best_corr_mean": zero.detach(),
            "corr0_mean": zero.detach(),
            "selected_t_start_mean": zero.detach(),
            "selected_t_start_min": zero.detach(),
            "selected_t_start_max": zero.detach(),
            "selected_gate_mean": zero.detach(),
            "gate_target_mean": zero.detach(),
            "n_windows": torch.as_tensor(0.0, device=pred_btd.device, dtype=pred_btd.dtype),
            "n_selected_windows": torch.as_tensor(0.0, device=pred_btd.device, dtype=pred_btd.dtype),
        }

    lag_tensor = torch.stack(lag_losses)
    phase_tensor = torch.stack(phase_losses)
    amp_tensor = torch.stack(amp_losses)
    score_tensor = torch.stack(scores).detach()
    start_tensor = torch.as_tensor(starts, device=pred_btd.device, dtype=pred_btd.dtype)
    amp_weight_tensor = torch.stack(amp_weights).detach()

    n_windows = int(score_tensor.numel())
    if int(top_k) > 0:
        k = min(int(top_k), n_windows)
        selected_idx = torch.topk(score_tensor, k=k, largest=True).indices
    else:
        selected_idx = torch.arange(n_windows, device=pred_btd.device)

    selected_scores = score_tensor[selected_idx]
    temp = max(float(score_temperature), 1.0e-8)
    selected_weights = torch.softmax(selected_scores / temp, dim=0).detach()

    lag_loss = torch.sum(selected_weights * lag_tensor[selected_idx])
    complex_phase_loss = torch.sum(selected_weights * phase_tensor[selected_idx])
    complex_amp_guard_loss = torch.sum(selected_weights * amp_tensor[selected_idx])
    loss = lag_loss + complex_phase_loss + complex_amp_guard_loss

    gate_align_loss = zero
    selected_gate_mean = zero.detach()
    gate_target_mean = zero.detach()
    if gate_means:
        gate_tensor = torch.stack(gate_means)
        score_ref = max(float(gate_target_score_ref), 1.0e-8)
        gate_target = (score_tensor / score_ref).clamp(0.0, 1.0).detach()
        gate_align_loss = torch.mean((gate_tensor - gate_target) ** 2)
        selected_gate_mean = torch.sum(selected_weights * gate_tensor[selected_idx]).detach()
        gate_target_mean = torch.sum(selected_weights * gate_target[selected_idx]).detach()

    selected_starts = start_tensor[selected_idx]
    return {
        "loss": loss,
        "lag_loss": lag_loss,
        "complex_phase_loss": complex_phase_loss,
        "complex_amp_guard_loss": complex_amp_guard_loss,
        "gate_align_loss": gate_align_loss,
        "score_mean": score_tensor.mean().detach(),
        "score_max": score_tensor.max().detach(),
        "amplitude_weight_mean": amp_weight_tensor.mean().detach(),
        "amplitude_weight_max": amp_weight_tensor.max().detach(),
        "best_abs_lag_s_mean": torch.stack(best_abs_lags).mean().detach(),
        "best_corr_mean": torch.stack(best_corrs).mean().detach(),
        "corr0_mean": torch.stack(corr0s).mean().detach(),
        "selected_t_start_mean": selected_starts.mean().detach(),
        "selected_t_start_min": selected_starts.min().detach(),
        "selected_t_start_max": selected_starts.max().detach(),
        "selected_gate_mean": selected_gate_mean,
        "gate_target_mean": gate_target_mean,
        "n_windows": torch.as_tensor(float(n_windows), device=pred_btd.device, dtype=pred_btd.dtype),
        "n_selected_windows": torch.as_tensor(float(selected_idx.numel()), device=pred_btd.device, dtype=pred_btd.dtype),
    }


def peak_and_lag_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dt: float,
    dof_indices: torch.Tensor,
    observations: str | Sequence[str] = "tip,last5",
    last_k: int = 5,
    peak_start_time: float = 0.0,
    peak_end_time: float | None = None,
    peak_window_seconds: float = 0.35,
    peak_temperature: float = 0.08,
    peak_min_distance_seconds: float = 0.3,
    peak_prominence_std: float = 0.15,
    peak_max_events: int = 16,
    lag_start_time: float = 0.0,
    lag_end_time: float | None = None,
    lag_window_seconds: float = 2.56,
    lag_stride_seconds: float = 1.28,
    max_lag_seconds: float = 0.8,
    lag_temperature: float = 0.05,
) -> dict[str, torch.Tensor]:
    """
    Average peak-time and local-lag losses over selected observations.

    This is intended for tip/last5 alignment, while the global spectrum loss
    can still use full-field mean response.
    """
    pred_btd = _as_btd(pred)
    target_btd = _as_btd(target)
    if pred_btd.shape != target_btd.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred_btd.shape)} vs {tuple(target_btd.shape)}")

    obs_list = _parse_observations(observations)

    peak_losses: list[torch.Tensor] = []
    lag_losses: list[torch.Tensor] = []
    peak_errs: list[torch.Tensor] = []
    lag_errs: list[torch.Tensor] = []
    peak_counts: list[torch.Tensor] = []
    lag_counts: list[torch.Tensor] = []

    for obs in obs_list:
        pred_sig = direction_observation_signal(
            pred_btd,
            dof_indices=dof_indices,
            observation=obs,
            last_k=last_k,
        )
        target_sig = direction_observation_signal(
            target_btd,
            dof_indices=dof_indices,
            observation=obs,
            last_k=last_k,
        )

        peak = peak_time_alignment_loss_from_signal(
            pred_sig,
            target_sig,
            dt=dt,
            start_time=peak_start_time,
            end_time=peak_end_time,
            window_seconds=peak_window_seconds,
            temperature=peak_temperature,
            min_distance_seconds=peak_min_distance_seconds,
            prominence_std=peak_prominence_std,
            max_events=peak_max_events,
            include_troughs=True,
        )
        lag = local_lag_alignment_loss_from_signal(
            pred_sig,
            target_sig,
            dt=dt,
            start_time=lag_start_time,
            end_time=lag_end_time,
            window_seconds=lag_window_seconds,
            stride_seconds=lag_stride_seconds,
            max_lag_seconds=max_lag_seconds,
            temperature=lag_temperature,
        )

        peak_losses.append(peak["loss"])
        lag_losses.append(lag["loss"])
        peak_errs.append(peak["mean_abs_time_error_s"].to(device=pred_btd.device, dtype=pred_btd.dtype))
        lag_errs.append(lag["mean_abs_lag_s"].to(device=pred_btd.device, dtype=pred_btd.dtype))
        peak_counts.append(peak["n_events"].to(device=pred_btd.device, dtype=pred_btd.dtype))
        lag_counts.append(lag["n_windows"].to(device=pred_btd.device, dtype=pred_btd.dtype))

    return {
        "peak_time_loss": torch.stack(peak_losses).mean(),
        "lag_loss": torch.stack(lag_losses).mean(),
        "mean_abs_peak_time_error_s": torch.stack(peak_errs).mean().detach(),
        "mean_abs_lag_s": torch.stack(lag_errs).mean().detach(),
        "n_peak_events": torch.stack(peak_counts).mean().detach(),
        "n_lag_windows": torch.stack(lag_counts).mean().detach(),
    }


# ============================================================
# Cached teacher-side frequency / peak / lag alignment losses
# ============================================================

@dataclass
class FrequencyAlignmentCache:
    """Teacher-side cache for frequency_alignment_loss.

    The cached tensors are fixed labels derived only from the teacher signal and
    the loss configuration. Prediction-side spectra are still computed with
    gradients, so the loss is unchanged.
    """

    dof_indices: torch.Tensor
    observation: str
    last_k: int
    dt: float
    freq_min: float
    freq_max: float | None
    peak_temperature: float
    target_power: torch.Tensor
    target_peak_hz: torch.Tensor


@dataclass
class PeakEventCache:
    observation: str
    anchor_idx: int
    kind: int
    lo: int
    hi: int


@dataclass
class LagWindowCache:
    observation: str
    lo: int
    hi: int
    valid_lags: tuple[int, ...]
    target_norm: torch.Tensor


@dataclass
class PeakLagAlignmentCache:
    """Teacher-side cache for peak_and_lag_alignment_loss.

    Peak/trough anchors, lag windows and normalized teacher lag windows are
    fixed for a case. Prediction-side soft peak and soft correlation are still
    evaluated every pass with gradients.
    """

    dof_indices: torch.Tensor
    observations: tuple[str, ...]
    last_k: int
    dt: float
    peak_window_seconds: float
    peak_temperature: float
    lag_window_seconds: float
    max_lag_seconds: float
    lag_temperature: float
    peak_events: tuple[PeakEventCache, ...]
    lag_windows: tuple[LagWindowCache, ...]


def build_frequency_alignment_cache(
    target: torch.Tensor,
    *,
    dt: float,
    dof_indices: torch.Tensor,
    freq_min: float = 0.0,
    freq_max: float | None = None,
    peak_temperature: float = 0.02,
    observation: str = "mean",
    last_k: int = 5,
) -> FrequencyAlignmentCache:
    target_btd = _as_btd(target)
    target_dir = direction_observation_signal(
        target_btd,
        dof_indices=dof_indices,
        observation=observation,
        last_k=last_k,
    )
    _, target_power = normalized_power_spectrum(
        target_dir,
        dt=dt,
        freq_min=freq_min,
        freq_max=freq_max,
    )
    target_peak = soft_peak_frequency(
        target_dir,
        dt=dt,
        freq_min=freq_min,
        freq_max=freq_max,
        temperature=peak_temperature,
    )
    return FrequencyAlignmentCache(
        dof_indices=dof_indices.detach().clone(),
        observation=str(observation),
        last_k=int(last_k),
        dt=float(dt),
        freq_min=float(freq_min),
        freq_max=None if freq_max is None else float(freq_max),
        peak_temperature=float(peak_temperature),
        target_power=target_power.detach().clone(),
        target_peak_hz=target_peak.detach().clone(),
    )


def frequency_alignment_loss_from_cache(
    pred: torch.Tensor,
    cache: FrequencyAlignmentCache,
) -> dict[str, torch.Tensor]:
    pred_btd = _as_btd(pred)
    idx = cache.dof_indices.to(device=pred_btd.device, dtype=torch.long)
    pred_dir = direction_observation_signal(
        pred_btd,
        dof_indices=idx,
        observation=cache.observation,
        last_k=cache.last_k,
    )
    _, pred_power = normalized_power_spectrum(
        pred_dir,
        dt=cache.dt,
        freq_min=cache.freq_min,
        freq_max=cache.freq_max,
    )
    target_power = cache.target_power.to(device=pred_power.device, dtype=pred_power.dtype)
    spec_loss = torch.mean((pred_power - target_power) ** 2)

    pred_peak = soft_peak_frequency(
        pred_dir,
        dt=cache.dt,
        freq_min=cache.freq_min,
        freq_max=cache.freq_max,
        temperature=cache.peak_temperature,
    )
    target_peak = cache.target_peak_hz.to(device=pred_peak.device, dtype=pred_peak.dtype)
    peak_loss = torch.mean((pred_peak - target_peak) ** 2)

    return {
        "spec_loss": spec_loss,
        "peak_loss": peak_loss,
        "pred_peak_hz": pred_peak.mean(),
        "target_peak_hz": target_peak.mean(),
    }


def build_peak_lag_alignment_cache(
    target: torch.Tensor,
    *,
    dt: float,
    dof_indices: torch.Tensor,
    observations: str | Sequence[str] = "tip,last5",
    last_k: int = 5,
    peak_start_time: float = 0.0,
    peak_end_time: float | None = None,
    peak_window_seconds: float = 0.35,
    peak_temperature: float = 0.08,
    peak_min_distance_seconds: float = 0.3,
    peak_prominence_std: float = 0.15,
    peak_max_events: int = 16,
    lag_start_time: float = 0.0,
    lag_end_time: float | None = None,
    lag_window_seconds: float = 2.56,
    lag_stride_seconds: float = 1.28,
    max_lag_seconds: float = 0.8,
    lag_temperature: float = 0.05,
    eps: float = 1.0e-12,
) -> PeakLagAlignmentCache:
    target_btd = _as_btd(target)
    if int(target_btd.shape[0]) != 1:
        raise ValueError(
            "Cached peak/lag alignment currently expects one target case at a time; "
            f"got batch={target_btd.shape[0]}."
        )

    obs_list = tuple(_parse_observations(observations))
    peak_events: list[PeakEventCache] = []
    lag_windows: list[LagWindowCache] = []

    T = int(target_btd.shape[1])
    win_steps = max(8, int(round(float(lag_window_seconds) / float(dt))))
    stride_steps = max(1, int(round(float(lag_stride_seconds) / float(dt))))
    max_lag_steps = max(1, int(round(float(max_lag_seconds) / float(dt))))
    lag_start_idx = max(0, int(round(float(lag_start_time) / float(dt))))
    lag_end_idx = T if lag_end_time is None else min(T, int(round(float(lag_end_time) / float(dt))) + 1)
    peak_radius_steps = max(2, int(round(float(peak_window_seconds) / float(dt))))

    for obs in obs_list:
        target_sig = direction_observation_signal(
            target_btd,
            dof_indices=dof_indices,
            observation=obs,
            last_k=last_k,
        )[0]

        events = _candidate_extrema_indices(
            target_sig,
            dt=dt,
            start_time=peak_start_time,
            end_time=peak_end_time,
            min_distance_seconds=peak_min_distance_seconds,
            prominence_std=peak_prominence_std,
            include_troughs=True,
            max_events=peak_max_events,
        )
        for anchor_idx, kind in events:
            lo = max(0, int(anchor_idx) - peak_radius_steps)
            hi = min(T - 1, int(anchor_idx) + peak_radius_steps)
            if hi - lo + 1 >= 4:
                peak_events.append(
                    PeakEventCache(
                        observation=obs,
                        anchor_idx=int(anchor_idx),
                        kind=int(kind),
                        lo=int(lo),
                        hi=int(hi),
                    )
                )

        if lag_end_idx - lag_start_idx >= win_steps:
            for lo in range(lag_start_idx, lag_end_idx - win_steps + 1, stride_steps):
                hi = lo + win_steps
                t = target_sig[lo:hi]
                t_norm = (t - t.mean()) / t.std().clamp_min(eps)
                valid_lags: list[int] = []
                for lag in range(-max_lag_steps, max_lag_steps + 1):
                    if lag < 0:
                        n = int(t_norm[-lag:].numel())
                    elif lag > 0:
                        n = int(t_norm[:-lag].numel())
                    else:
                        n = int(t_norm.numel())
                    if n >= 8:
                        valid_lags.append(int(lag))
                if valid_lags:
                    lag_windows.append(
                        LagWindowCache(
                            observation=obs,
                            lo=int(lo),
                            hi=int(hi),
                            valid_lags=tuple(valid_lags),
                            target_norm=t_norm.detach().clone(),
                        )
                    )

    return PeakLagAlignmentCache(
        dof_indices=dof_indices.detach().clone(),
        observations=obs_list,
        last_k=int(last_k),
        dt=float(dt),
        peak_window_seconds=float(peak_window_seconds),
        peak_temperature=float(peak_temperature),
        lag_window_seconds=float(lag_window_seconds),
        max_lag_seconds=float(max_lag_seconds),
        lag_temperature=float(lag_temperature),
        peak_events=tuple(peak_events),
        lag_windows=tuple(lag_windows),
    )


def peak_and_lag_alignment_loss_from_cache(
    pred: torch.Tensor,
    cache: PeakLagAlignmentCache,
    *,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    pred_btd = _as_btd(pred)
    if int(pred_btd.shape[0]) != 1:
        raise ValueError(
            "Cached peak/lag alignment currently expects one predicted case at a time; "
            f"got batch={pred_btd.shape[0]}."
        )

    idx = cache.dof_indices.to(device=pred_btd.device, dtype=torch.long)
    pred_signals: dict[str, torch.Tensor] = {}
    for obs in cache.observations:
        pred_signals[obs] = direction_observation_signal(
            pred_btd,
            dof_indices=idx,
            observation=obs,
            last_k=cache.last_k,
        )[0]

    dtype = pred_btd.dtype
    device = pred_btd.device
    tau_peak = max(float(cache.peak_temperature), 1.0e-8)
    peak_window_seconds_safe = max(float(cache.peak_window_seconds), float(cache.dt))
    tau_lag = max(float(cache.lag_temperature), 1.0e-8)
    max_lag_steps = max(1, int(round(float(cache.max_lag_seconds) / float(cache.dt))))

    obs_peak_losses: list[torch.Tensor] = []
    obs_lag_losses: list[torch.Tensor] = []
    obs_peak_errs: list[torch.Tensor] = []
    obs_lag_errs: list[torch.Tensor] = []
    obs_peak_counts: list[torch.Tensor] = []
    obs_lag_counts: list[torch.Tensor] = []

    for obs in cache.observations:
        peak_losses: list[torch.Tensor] = []
        peak_abs_errors: list[torch.Tensor] = []
        obs_events = [e for e in cache.peak_events if e.observation == obs]
        for event in obs_events:
            pred_win = pred_signals[event.observation][event.lo : event.hi + 1]
            if pred_win.numel() < 4:
                continue
            score = pred_win if int(event.kind) > 0 else -pred_win
            score = (score - score.mean()) / score.std().clamp_min(eps)
            weights = torch.softmax(score / tau_peak, dim=0)
            times = torch.arange(event.lo, event.hi + 1, device=device, dtype=dtype) * float(cache.dt)
            pred_time = torch.sum(weights * times)
            teacher_time = torch.as_tensor(event.anchor_idx * float(cache.dt), device=device, dtype=dtype)
            err = pred_time - teacher_time
            peak_losses.append((err / peak_window_seconds_safe) ** 2)
            peak_abs_errors.append(torch.abs(err))

        if peak_losses:
            obs_peak_losses.append(torch.stack(peak_losses).mean())
            obs_peak_errs.append(torch.stack(peak_abs_errors).mean().detach())
        else:
            zero = _zero_like_signal_loss(pred_btd)
            obs_peak_losses.append(zero)
            obs_peak_errs.append(zero.detach())
        # Match the historical metric in peak_time_alignment_loss_from_signal,
        # where n_events is incremented twice per anchor. This affects logging
        # only, not the differentiable loss.
        obs_peak_counts.append(torch.as_tensor(float(len(obs_events)), device=device, dtype=dtype))

        lag_losses: list[torch.Tensor] = []
        lag_abs_errors: list[torch.Tensor] = []
        obs_windows = [w for w in cache.lag_windows if w.observation == obs]
        for window in obs_windows:
            p = pred_signals[window.observation][window.lo : window.hi]
            if p.numel() < 8:
                continue
            p_norm = (p - p.mean()) / p.std().clamp_min(eps)
            t_norm = window.target_norm.to(device=device, dtype=dtype)

            corrs: list[torch.Tensor] = []
            valid_lags: list[int] = []
            for lag in window.valid_lags:
                if lag < 0:
                    p_l = p_norm[:lag]
                    t_l = t_norm[-lag:]
                elif lag > 0:
                    p_l = p_norm[lag:]
                    t_l = t_norm[:-lag]
                else:
                    p_l = p_norm
                    t_l = t_norm
                if p_l.numel() < 8:
                    continue
                corrs.append(torch.mean(p_l * t_l))
                valid_lags.append(int(lag))

            if not corrs:
                continue
            corr_tensor = torch.stack(corrs)
            lag_tensor = torch.tensor(valid_lags, device=device, dtype=dtype)
            weights = torch.softmax(corr_tensor / tau_lag, dim=0)
            soft_lag_steps = torch.sum(weights * lag_tensor)
            lag_losses.append((soft_lag_steps / float(max_lag_steps)) ** 2)
            lag_abs_errors.append(torch.abs(soft_lag_steps) * float(cache.dt))

        if lag_losses:
            obs_lag_losses.append(torch.stack(lag_losses).mean())
            obs_lag_errs.append(torch.stack(lag_abs_errors).mean().detach())
        else:
            zero = _zero_like_signal_loss(pred_btd)
            obs_lag_losses.append(zero)
            obs_lag_errs.append(zero.detach())
        obs_lag_counts.append(torch.as_tensor(float(len(obs_windows)), device=device, dtype=dtype))

    return {
        "peak_time_loss": torch.stack(obs_peak_losses).mean(),
        "lag_loss": torch.stack(obs_lag_losses).mean(),
        "mean_abs_peak_time_error_s": torch.stack(obs_peak_errs).mean().detach(),
        "mean_abs_lag_s": torch.stack(obs_lag_errs).mean().detach(),
        "n_peak_events": torch.stack(obs_peak_counts).mean().detach(),
        "n_lag_windows": torch.stack(obs_lag_counts).mean().detach(),
    }
