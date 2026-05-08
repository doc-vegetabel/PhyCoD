import numpy as np
import yaml
from pathlib import Path


def extract_tip_flap_signal(u_tch: np.ndarray) -> np.ndarray:
    """
    从全叶片平动响应中提取叶尖 flapwise 通道。
    约定当前数据排布为:
        [node1_x, node1_y, node1_z, node2_x, node2_y, node2_z, ...]
    且你当前绘图代码中 tip_flap_idx = -3，
    因此这里沿用最后一个节点的第一个平动分量作为 tip flap signal。
    """
    if u_tch.ndim != 2:
        raise ValueError(f"u_tch 应为二维数组 [T, n_dofs]，当前 shape={u_tch.shape}")
    if u_tch.shape[1] < 3:
        raise ValueError("u_tch 的自由度维数不足，无法提取 tip flap 通道。")

    return u_tch[:, -3].copy()


def estimate_release_index_from_signal(signal: np.ndarray, dt: float, min_search_time: float = 0.2) -> int:
    """
    自动估计撤载后的自由衰减起点。
    对 pluck 工况，前一段持续受力，撤载后会出现明显转折。
    这里采用最简单稳妥的方法：寻找前段时间里 |dx/dt| 最大的位置附近作为 release 点。
    """
    if len(signal) < 10:
        raise ValueError("信号长度太短，无法识别撤载点。")

    grad = np.gradient(signal, dt)
    start_idx = max(1, int(min_search_time / dt))

    # 只在前 30% 时间里找撤载点，避免后面自由振动干扰
    end_idx = max(start_idx + 5, int(0.3 * len(signal)))
    local_grad = np.abs(grad[start_idx:end_idx])

    rel_idx = np.argmax(local_grad)
    release_idx = start_idx + rel_idx

    return int(release_idx)


def extract_free_decay_segment(time_array: np.ndarray,
                               signal: np.ndarray,
                               release_idx: int,
                               trim_head: int = 2):
    """
    取撤载后的自由衰减段。
    trim_head 用于略微跳过撤载瞬间的尖峰数值扰动。
    """
    start_idx = min(len(signal) - 1, release_idx + trim_head)
    t_free = time_array[start_idx:] - time_array[start_idx]
    x_free = signal[start_idx:].copy()

    # 去均值，减少频域直流偏置
    x_free = x_free - np.mean(x_free)

    return t_free, x_free, start_idx


def estimate_frequency_fft(signal: np.ndarray, dt: float, fmin: float = 0.05, fmax: float = 5.0) -> float:
    """
    用 FFT 估计主频。
    这是第一版最稳的做法，不引入 scipy。
    """
    n = len(signal)
    if n < 8:
        raise ValueError("自由衰减段太短，无法做 FFT 频率估计。")

    window = np.hanning(n)
    sig = signal * window

    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=dt)
    amps = np.abs(fft_vals)

    valid = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(valid):
        raise ValueError(f"在频率范围 [{fmin}, {fmax}] Hz 内未找到有效频率点。")

    freqs_valid = freqs[valid]
    amps_valid = amps[valid]

    peak_idx = np.argmax(amps_valid)
    return float(freqs_valid[peak_idx])


def _find_local_peaks(signal: np.ndarray, min_distance: int = 5):
    """
    不依赖 scipy 的简易峰值检测：
    找局部极大值，并设置最小峰间距。
    """
    peaks = []
    last_peak = -min_distance

    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            if i - last_peak >= min_distance:
                peaks.append(i)
                last_peak = i

    return np.array(peaks, dtype=int)


def estimate_damping_logdec(signal: np.ndarray,
                            dt: float,
                            freq_hz: float,
                            min_peaks: int = 4):
    """
    用对数递减法估计阻尼比。
    做法：
    1. 找自由衰减段的正峰值
    2. 取若干峰值做 log decrement
    3. 用平均 δ 估计 zeta
    """
    if freq_hz <= 0:
        raise ValueError("freq_hz 必须为正数。")

    approx_period = 1.0 / freq_hz
    min_distance = max(3, int(0.5 * approx_period / dt))

    peaks = _find_local_peaks(signal, min_distance=min_distance)

    # 只保留正峰值且幅值明显的峰
    peak_vals = signal[peaks]
    valid_mask = peak_vals > 0.05 * np.max(np.abs(signal))
    peaks = peaks[valid_mask]
    peak_vals = peak_vals[valid_mask]

    if len(peak_vals) < min_peaks:
        raise ValueError(
            f"可用峰值数不足，当前仅检测到 {len(peak_vals)} 个峰，无法稳定估计阻尼。"
        )

    # 用首峰到后续峰做多组 log decrement
    deltas = []
    a0 = peak_vals[0]
    for k in range(1, len(peak_vals)):
        ak = peak_vals[k]
        if ak <= 0:
            continue
        delta_k = (1.0 / k) * np.log(a0 / ak)
        if np.isfinite(delta_k) and delta_k > 0:
            deltas.append(delta_k)

    if len(deltas) == 0:
        raise ValueError("未能得到有效的 log decrement。")

    delta = float(np.mean(deltas))
    zeta = delta / np.sqrt((2.0 * np.pi) ** 2 + delta ** 2)

    return float(zeta), int(len(peak_vals))


def identify_teacher_modal_properties(time_array: np.ndarray,
                                      u_tch: np.ndarray,
                                      dt: float,
                                      release_idx: int = None):
    """
    第一版 teacher 模态识别主函数：
    - 默认使用 tip flap 通道
    - 自动识别撤载点
    - 提取自由衰减段
    - FFT 估计主频
    - 对数递减估计阻尼比
    """
    signal = extract_tip_flap_signal(u_tch)

    if release_idx is None:
        release_idx = estimate_release_index_from_signal(signal, dt)

    t_free, x_free, free_start_idx = extract_free_decay_segment(time_array, signal, release_idx)

    freq_hz = estimate_frequency_fft(x_free, dt)
    omega_rad_s = 2.0 * np.pi * freq_hz
    zeta, n_peaks_used = estimate_damping_logdec(x_free, dt, freq_hz)

    result = {
        "channel_name": "tip_flap",
        "release_index": int(release_idx),
        "free_decay_start_index": int(free_start_idx),
        "release_time": float(time_array[release_idx]),
        "freq_hz": float(freq_hz),
        "omega_rad_s": float(omega_rad_s),
        "zeta": float(zeta),
        "n_peaks_used": int(n_peaks_used),
    }
    return result


def save_modal_id_yaml(result: dict, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, sort_keys=False, allow_unicode=True)

    print(f"✅ Teacher 模态识别结果已保存到: {save_path}")


def load_release_index_from_case_yaml(case_name: str, project_root, dt: float) -> int:
    """
    从工况 yaml 中读取 pluck_duration，并转换成 release_idx。
    """
    yaml_path = Path(project_root) / f"cases/structure/{case_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到工况 yaml 文件: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    load_profile = config.get("load_profile", None)
    pluck_duration = config.get("pluck_duration", None)

    if load_profile != "pluck":
        raise ValueError(f"工况 [{case_name}] 的 load_profile 不是 pluck，而是 {load_profile}")

    if pluck_duration is None:
        raise ValueError(f"工况 [{case_name}] 的 yaml 中没有 pluck_duration 字段。")

    release_idx = int(round(float(pluck_duration) / dt))
    return max(1, release_idx)

def identify_teacher_modal_properties_from_case(case_name: str,
                                                project_root,
                                                time_array: np.ndarray,
                                                u_tch: np.ndarray,
                                                dt: float):
    """
    从 case yaml 自动读取 release_idx，再进行 teacher 模态识别。
    """
    release_idx = load_release_index_from_case_yaml(case_name, project_root, dt)
    result = identify_teacher_modal_properties(
        time_array=time_array,
        u_tch=u_tch,
        dt=dt,
        release_idx=release_idx
    )
    result["case_name"] = case_name
    return result



def load_modal_id_yaml(load_path):
    load_path = Path(load_path)
    with open(load_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data