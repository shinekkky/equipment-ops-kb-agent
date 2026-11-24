"""四层异常检测：阈值规则 / 滚动 Z-score / 趋势劣化 / 多变量联合。

为什么是四层，而不是一个"异常检测模型"
------------------------------------
四层不是叠床架屋。每一层负责的形态，恰好是其余三层必然漏掉的，
scripts/gen_sensor_data.py 为每一层都注入了专属样本，可直接验证：

  层            抓什么                              专属样本
  ------------  ----------------------------------  --------------------------------
  阈值规则      已经越线的显性告警                    M-201 振动越过 7.1 mm/s
  滚动 Z-score  越线之前、甚至永远不越线的尖峰        P-103 冲高 +7.5 ℃ 但未达 75 ℃
  趋势劣化      缓慢爬升，越线只是时间问题            P-101 轴承温度每天 +1.6 ℃
  多变量联合    单指标都正常、但几个指标同步偏移      M-202 温度/振动/电流各偏 ~1.7σ

只做阈值规则，后三类隐患全部漏掉——而现场出事的往往正是"还没报警"的设备。
只做阈值 + Z-score，缓慢劣化仍会漏，因为爬升的每一天单独看都很正常。

两个工程上必须做的处理
--------------------
1. **窗口未填满不许判**。滚动中位数/MAD 在小样本上会严重低估尺度：实测窗口
   只有 3 个点时 σ 估成 0.08（真值 1.60），z 值随之虚高，序列开头会冒出一片
   假阳性。故统计层一律要求窗口内至少 Z_MIN_WINDOW_HOURS 个点才出结论。

2. **段合并要容忍短空洞**。异常几乎不会逐小时连续命中：M-202 的漂移尾部
   24 小时里模型判出 22 个点，被 2 个漏点切成 3 段，每段都不足最小持续时长，
   整条隐患就此消失。故检测出的段先按 _merge_gaps 合并间隔极短的相邻段。

关于"基线"
--------
Z-score、趋势、多变量三层都需要一个"正常"的参照。本模块统一取序列开头的
BASELINE_DAYS 天作为基线期。这是个**明确的假设**：假定投运初期工况正常。
现场部署时应改为用检修后的稳定期数据，或人工指定基线窗口。
所有检测结果都记录 baseline 值，报告里一并给出，便于人工复核这个假设是否成立。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .thresholds import (
    ALARM,
    ATTENTION,
    NORMAL,
    SEVERITY_ORDER,
    Band,
    band_for,
    bands_for,
    feature_labels,
)

# ---------------------------------------------------------------- 参数

BASELINE_DAYS = 10         # 基线期长度，三层统计检测共用

Z_WINDOW_HOURS = 168       # 滚动 Z-score 的参考窗（7 天）
Z_MIN_WINDOW_HOURS = 72    # 窗口至少填入 3 天才出结论，避免小样本尺度失真
Z_THRESHOLD = 3.5          # |z| 超过此值判为异常点

TREND_WINDOW_HOURS = 120   # 趋势拟合窗（5 天）
TREND_MIN_SIGMA_PER_DAY = 0.25  # 每天变化量达 0.25σ 才算劣化，低于此值淹没在噪声里
TREND_MIN_NET_SIGMA = 1.0  # 窗口两端净变化至少 1σ，用以排除"已经过去的一次尖峰"
TREND_MIN_HOURS = 48       # 趋势至少持续 2 天，孤立的高斜率点只是噪声抖动

MV_SMOOTH_HOURS = 24       # 多变量检测先做 24 h 滚动平均
MV_CONTAMINATION = 0.05    # IsolationForest 预期异常比例
MV_MIN_HOURS = 24          # 多变量异常至少持续 24 h，否则视为单点抖动
MV_MIN_JOINT_Z = 2.0       # 联合偏离至少 2σ，避免把微小漂移报成隐患

EVENT_MAX_GAP_HOURS = 12   # 段合并时允许的最大空洞

LAYER_LABELS = {
    "threshold": "阈值规则",
    "zscore": "滚动 Z-score",
    "trend": "趋势劣化",
    "multivariate": "多变量联合",
}

# 各指标对应的"症状说法"，用于拼检索查询——检索要的是运维人员的口语，
# 不是列名。问"振动_mm_s 超过 7.1"召回率远低于问"振动超标怎么处理"。
SYMPTOMS = {
    "温度_℃": ("温度过高", "温度持续上升"),
    "压力_MPa": ("出口压力偏低 汽蚀", "出口压力持续下降"),
    "振动_mm_s": ("振动超标", "振动持续上升"),
    "电流_A": ("电机过载 电流异常", "电流持续上升"),
    "CPU利用率_%": ("CPU 利用率过高", "CPU 利用率持续上升"),
    "内存利用率_%": ("内存利用率过高", "内存利用率持续上升"),
    "端口丢包率_%": ("端口丢包 错包", "端口丢包率持续上升"),
}


# ---------------------------------------------------------------- 数据结构


@dataclass
class Anomaly:
    """一条原始检测发现（未经跨层合并）。"""

    device_id: str
    device_name: str
    device_type: str
    model: str
    location: str
    metric: str
    metric_label: str
    unit: str
    layer: str
    severity: str
    start: pd.Timestamp
    end: pd.Timestamp
    peak: float
    baseline: float
    detail: str
    boundary: float | None = None
    # 多变量层的归因：[(中文标签, 基线期 σ 倍数), ...]
    attribution: list[tuple[str, float]] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    @property
    def layer_label(self) -> str:
        return LAYER_LABELS.get(self.layer, self.layer)


# ---------------------------------------------------------------- 通用工具


def _mask_to_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """把布尔掩码切成连续段，返回 [(起始下标, 结束下标)]，两端均含。"""
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge_gaps(runs: list[tuple[int, int]], max_gap: int = EVENT_MAX_GAP_HOURS) -> list[tuple[int, int]]:
    """把间隔不超过 max_gap 的相邻段并成一段。

    没有这一步，只要异常时段里漏判一两个小时，一段持续数天的隐患就会被切成
    若干短段，而每段都过不了"最小持续时长"这道关——隐患就这样凭空消失了。
    """
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _baseline_slice(n: int) -> slice:
    return slice(0, min(n, BASELINE_DAYS * 24))


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    """用中位数与 1.4826×MAD 估计中心与尺度。

    相比均值/标准差，这两个统计量对异常段本身不敏感——否则一段持续劣化
    会把均值抬高、标准差撑大，反而把自己"洗白"成正常。
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = 1.4826 * mad
    if scale <= 1e-9:  # 信号几乎恒定（如停机状态），退化为标准差
        scale = float(np.std(finite))
    return center, scale


def _fmt(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%m-%d %H:%M")


def _num(value: float) -> str:
    """按量级设小数位，并去掉无意义的末尾零。

    出口压力这类量值只有 0.3~0.7，统一保留两位会把有效信息抹平；而振动 7.1
    保留三位又会写成 "7.100mm/s"，看着像测量精度比实际高。
    """
    magnitude = abs(value)
    if magnitude >= 100:
        text = f"{value:.1f}"
    elif magnitude >= 10:
        text = f"{value:.2f}"
    elif magnitude >= 1:
        text = f"{value:.2f}"
    else:
        text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _num_decimals(value: float) -> int:
    """_num 在相应量级用的小数位，供"判定精度与显示精度保持一致"使用。"""
    magnitude = abs(value)
    if magnitude >= 100:
        return 1
    if magnitude >= 1:
        return 2
    return 3


# ---------------------------------------------------------------- 第一层：阈值规则


def detect_threshold(df: pd.DataFrame, band: Band, device: dict) -> list[Anomaly]:
    """逐点套用手册阈值，把连续越限的时段合并成一条发现。"""
    values = df[band.metric].to_numpy(dtype=float)
    times = df["时间"]

    severities = np.array([band.evaluate(v) for v in values], dtype=object)
    mask = severities != NORMAL
    if not mask.any():
        return []

    base_center, _ = _robust_center_scale(values[_baseline_slice(len(values))])
    higher_is_worse = band.direction == "high"
    out: list[Anomaly] = []

    # 越限往往断续发生：温度爬升过程中会在预警线附近反复穿越。逐段列出会得到
    # 十几条"连续 1 小时"的证据，读者只会看到一堆碎片。合并成一段、改为报告
    # "累计 N 小时、峰值多少"，才对应运维眼中的一件事。
    for start_i, end_i in _merge_gaps(_mask_to_runs(mask)):
        seg_sev = [s for s in severities[start_i:end_i + 1] if s != NORMAL]
        if not seg_sev:
            continue
        severity = max(seg_sev, key=lambda s: SEVERITY_ORDER[s])
        seg_vals = values[start_i:end_i + 1]
        peak = float(np.nanmax(seg_vals) if higher_is_worse else np.nanmin(seg_vals))
        boundary = band.boundary_for(severity)

        flagged = int(np.count_nonzero(mask[start_i:end_i + 1]))
        span = end_i - start_i + 1
        extent = (
            f"连续 {span} 小时" if flagged == span
            else f"累计 {flagged} 小时（跨 {span} 小时）"
        )
        out.append(Anomaly(
            device_id=device["设备ID"], device_name=device["设备名称"],
            device_type=device["设备类型"], model=device["设备型号"],
            location=device["安装位置"],
            metric=band.metric, metric_label=band.label, unit=band.unit,
            layer="threshold", severity=severity,
            start=times.iloc[start_i], end=times.iloc[end_i],
            peak=peak, baseline=base_center, boundary=boundary,
            detail=(
                f"{band.label}在 {_fmt(times.iloc[start_i])} 至 {_fmt(times.iloc[end_i])} "
                f"{extent}{'高于' if higher_is_worse else '低于'}"
                f"{_num(boundary)}{band.unit}（{severity}级），峰值 {_num(peak)}{band.unit}。"
                f"判据：{band.source}"
            ),
        ))
    return out


# ---------------------------------------------------------------- 第二层：滚动 Z-score


def _rolling_center_scale(
    values: np.ndarray, window: int, min_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """滚动中位数与 1.4826×MAD。窗口内点数不足 min_points 时返回 NaN。"""
    n = len(values)
    centers = np.full(n, np.nan)
    scales = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        if i - lo + 1 < min_points:
            continue
        centers[i], scales[i] = _robust_center_scale(values[lo:i + 1])
    return centers, scales


def detect_zscore(df: pd.DataFrame, band: Band, device: dict) -> list[Anomaly]:
    """相对滚动基线的偏离。抓的是"没越线但明显不对劲"的短时尖峰。"""
    values = df[band.metric].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return []

    centers, scales = _rolling_center_scale(values, Z_WINDOW_HOURS, Z_MIN_WINDOW_HOURS)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (values - centers) / scales

    # 只取"危险方向"的偏离。手册的阈值是单向的（温度越高越危险、出口压力越低越
    # 危险），反方向的离群值意味着传感器跳变或采集丢点，属于数据质量问题，
    # 报成"设备隐患"会把报告带偏。故 direction="low" 的指标取 -z。
    signed = z if band.direction == "high" else -z
    mask = np.isfinite(signed) & (signed > Z_THRESHOLD)
    if not mask.any():
        return []

    times = df["时间"]
    higher_is_worse = band.direction == "high"
    out: list[Anomaly] = []
    for start_i, end_i in _merge_gaps(_mask_to_runs(mask)):
        seg_z = signed[start_i:end_i + 1]
        seg_z = seg_z[np.isfinite(seg_z)]
        if seg_z.size == 0:
            continue
        peak_z = float(seg_z[np.argmax(seg_z)])
        seg_vals = values[start_i:end_i + 1]
        peak = float(np.nanmax(seg_vals) if higher_is_worse else np.nanmin(seg_vals))
        base = float(np.nanmedian(centers[start_i:end_i + 1]))

        # 这一层若同时越了手册阈值，说明阈值层也会报，合并时归为同一问题；
        # 若没越线，就是阈值层漏报的隐患——这正是本层存在的理由。
        warn_line = band.warn if band.warn is not None else band.alarm
        # 判定用与显示相同的精度：否则会出现"未越预警线（0.38MPa），峰值 0.38MPa"
        # 这种自相矛盾的句子——实际是 0.3801 与 0.3800 之差，被四舍五入抹平了。
        suffix = (
            "该时段同时越过手册阈值，属阈值层可见的显性异常。"
            if band.evaluate(round(peak, _num_decimals(peak))) != NORMAL else
            f"该时段未触发阈值规则层的越限判定（预警线 {_num(warn_line)}{band.unit}），"
            "属统计层独立捕获的早期异常。"
        )
        out.append(Anomaly(
            device_id=device["设备ID"], device_name=device["设备名称"],
            device_type=device["设备类型"], model=device["设备型号"],
            location=device["安装位置"],
            metric=band.metric, metric_label=band.label, unit=band.unit,
            layer="zscore", severity=ATTENTION,
            start=times.iloc[start_i], end=times.iloc[end_i],
            peak=peak, baseline=base,
            boundary=band.alarm,
            detail=(
                f"{band.label}在 {_fmt(times.iloc[start_i])} 出现偏离滚动基线 "
                f"{abs(peak_z):.1f}σ 的异常波动（滚动基线 {_num(base)}{band.unit}，"
                f"峰值 {_num(peak)}{band.unit}）。{suffix}"
            ),
        ))
    return out


# ---------------------------------------------------------------- 第三层：趋势劣化


def _rolling_trend(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """尾部窗口的一元线性回归。

    返回 (每天斜率, 窗口两端净变化)。两者都要用：
    斜率说明"正在变坏"，净变化说明"确实积累下来了"。
    只看斜率会把一次已经过去、数值早已回落的尖峰误判成趋势。
    """
    n = len(values)
    slope = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    if n < window:
        return slope, delta

    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    sxx = float((x_centered ** 2).sum())

    for i in range(window - 1, n):
        y = values[i - window + 1: i + 1]
        if not np.isfinite(y).all():
            continue
        slope[i] = float((x_centered * (y - y.mean())).sum() / sxx * 24.0)  # → 每天
        delta[i] = float(y[-1] - y[0])
    return slope, delta


def detect_trend(df: pd.DataFrame, band: Band, device: dict) -> list[Anomaly]:
    """拟合尾部窗口的斜率，捕捉"每天涨一点"的缓慢劣化。

    阈值层和 Z-score 层都只能看到"已经偏离"，对每天 +0.5℃ 这类爬升无能为力：
    单看每一天都很正常，等它越过报警线时往往只剩几天的处置窗口。
    """
    values = df[band.metric].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return []

    base_center, base_scale = _robust_center_scale(values[_baseline_slice(len(values))])
    if base_scale <= 1e-9:
        return []

    slopes, deltas = _rolling_trend(values, TREND_WINDOW_HOURS)
    sign = 1.0 if band.direction == "high" else -1.0  # 只有朝危险方向变化才算劣化
    rate = sign * slopes / base_scale               # σ / 天
    net = sign * deltas / base_scale                # σ

    mask = (
        np.isfinite(rate) & np.isfinite(net)
        & (rate >= TREND_MIN_SIGMA_PER_DAY)
        & (net >= TREND_MIN_NET_SIGMA)
    )
    if not mask.any():
        return []

    times = df["时间"]
    higher_is_worse = band.direction == "high"
    out: list[Anomaly] = []
    for start_i, end_i in _merge_gaps(_mask_to_runs(mask)):
        if end_i - start_i + 1 < TREND_MIN_HOURS:
            continue  # 孤立的高斜率点只是噪声抖动，够不上"趋势"
        seg = rate[start_i:end_i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        # 用段内速率的中位数而非峰值，避免把一次抖动说成长期趋势
        rep_rate = float(np.median(seg)) * base_scale
        tail_lo = max(0, end_i - 23)
        seg_vals = values[tail_lo:end_i + 1]
        peak = float(np.nanmax(seg_vals) if higher_is_worse else np.nanmin(seg_vals))
        eta = _eta_days(values[end_i], rep_rate, band)

        out.append(Anomaly(
            device_id=device["设备ID"], device_name=device["设备名称"],
            device_type=device["设备类型"], model=device["设备型号"],
            location=device["安装位置"],
            metric=band.metric, metric_label=band.label, unit=band.unit,
            layer="trend", severity=ATTENTION,
            start=times.iloc[start_i], end=times.iloc[end_i],
            peak=peak, baseline=base_center,
            boundary=band.warn if band.warn is not None else band.alarm,
            detail=(
                f"{band.label}自 {_fmt(times.iloc[start_i])} 起呈持续"
                f"{'上升' if higher_is_worse else '下降'}趋势，"
                f"速率约 {_num(abs(rep_rate))}{band.unit}/天（基线 {_num(base_center)}{band.unit}，"
                f"波动 σ={_num(base_scale)}，约 {abs(rep_rate) / base_scale:.2f}σ/天）。"
                + (f"按此外推，约 {eta:.0f} 天后将触及报警线。" if eta else
                   "当前尚未越限，但斜率已显著偏离正常波动。")
            ),
        ))
    return out


def _eta_days(current: float, rate_per_day: float, band: Band) -> float | None:
    """按当前速率外推到报警线还需多少天；已越线或方向不对则返回 None。"""
    target = band.alarm
    if target is None or rate_per_day == 0:
        return None
    gap = (target - current) if band.direction == "high" else (current - target)
    if gap <= 0 or rate_per_day <= 0:
        return None
    days = gap / rate_per_day
    return days if days <= 365 else None


# ---------------------------------------------------------------- 第四层：多变量联合


def detect_multivariate(df: pd.DataFrame, features: list[str], device: dict) -> list[Anomaly]:
    """多指标联合偏离检测：单看每个指标都不越限，合起来已经不对了。

    做法是"黑箱打分 + 白箱归因"：
      * IsolationForest 负责判断"这段窗口不正常"，它不解释原因；
      * 各指标相对基线期的 σ 倍数负责解释"因为哪几个参数同步偏了"。
    只有打分没有归因，运维不知道该查什么；只有归因没有打分，
    就得为每个指标手写阈值组合规则——那正是多层要做的事。

    为什么先做滚动平均再喂给模型
    ---------------------------
    直接把逐点数值喂给 IsolationForest，它找到的是**全局最孤立的点**，
    也就是各指标各自的噪声极值，而不是系统性漂移。实测 M-202：逐点时
    36 个命中点散成 33 段，没有一段够长；先做 24 h 滚动平均后，噪声被压掉，
    漂移尾部 24 小时里 22 个小时被判为异常，隐患才浮出来。
    滚动平均在这里不是平滑装饰，它是让模型看得见"系统性偏移"的前提。

    刻意排除已越限的时段：那些属于阈值层的职责，重复报会让报告显得虚。
    """
    from sklearn.ensemble import IsolationForest

    cols = [c for c in features if c in df.columns and df[c].notna().any()]
    if len(cols) < 2:
        return []

    raw = df[cols].to_numpy(dtype=float)
    base = raw[_baseline_slice(len(raw))]

    centers = np.array([np.nanmedian(base[:, j]) for j in range(len(cols))])
    scales = np.array([_robust_center_scale(base[:, j])[1] for j in range(len(cols))])
    scales[scales <= 1e-9] = 1.0

    with np.errstate(invalid="ignore"):
        z = (raw - centers) / scales
    # 行内含 NaN 时（如网络设备没有压力/振动量）用 0 填充，
    # 等于"该维度无信息"，不会把缺失误判成偏离。
    z = np.nan_to_num(z, nan=0.0)

    smoothed = pd.DataFrame(z).rolling(MV_SMOOTH_HOURS, min_periods=MV_SMOOTH_HOURS).mean().to_numpy()
    valid = np.isfinite(smoothed).all(axis=1)
    if valid.sum() < MV_MIN_HOURS:
        return []

    model = IsolationForest(
        n_estimators=300, contamination=MV_CONTAMINATION, random_state=42
    )
    flags = np.zeros(len(smoothed), dtype=bool)
    flags[valid] = model.fit_predict(smoothed[valid]) == -1

    abs_z = np.nan_to_num(np.abs(smoothed), nan=0.0)
    joint_z = np.sqrt(np.nansum(smoothed ** 2, axis=1))
    joint_z = np.nan_to_num(joint_z, nan=0.0)

    # 三重条件缺一不可：模型判异常、单指标都未越统计线、联合偏离够大
    mask = flags & (abs_z.max(axis=1) < Z_THRESHOLD) & (joint_z >= MV_MIN_JOINT_Z)
    if not mask.any():
        return []

    times = df["时间"]
    labels = feature_labels(cols)
    out: list[Anomaly] = []
    for start_i, end_i in _merge_gaps(_mask_to_runs(mask)):
        if end_i - start_i + 1 < MV_MIN_HOURS:
            continue  # 单点抖动不算隐患
        contrib = np.nanmean(z[start_i:end_i + 1], axis=0)
        attribution = sorted(
            ((labels[j], float(contrib[j])) for j in range(len(cols))),
            key=lambda kv: -abs(kv[1]),
        )
        top = attribution[0]
        desc = "、".join(f"{name} {val:+.2f}σ" for name, val in attribution)
        out.append(Anomaly(
            device_id=device["设备ID"], device_name=device["设备名称"],
            device_type=device["设备类型"], model=device["设备型号"],
            location=device["安装位置"],
            metric="+".join(cols), metric_label="多参数联合", unit="",
            layer="multivariate", severity=ATTENTION,
            start=times.iloc[start_i], end=times.iloc[end_i],
            peak=float(np.max(joint_z[start_i:end_i + 1])),
            baseline=0.0,
            boundary=None,
            attribution=attribution,
            detail=(
                f"{'、'.join(labels)}在 {_fmt(times.iloc[start_i])} 之后同步偏离基线期水平"
                f"（{desc}），而任一指标单独看均未越报警线。"
                f"以 {top[0]} 偏离最显著。多参数同步偏移指向系统性劣化，"
                "单指标阈值规则无法识别。"
            ),
        ))
    return out


# ---------------------------------------------------------------- 检索提示


def retrieval_hint(anomaly: Anomaly) -> tuple[str, str]:
    """给一条发现配检索查询，返回 (查询语句, 建议优先命中的手册名)。

    查询用运维人员的口语拼，而不是拿列名拼：手册里写的是"轴承温度过高"，
    没人会搜"温度_℃ 大于 85"。这条查询会直接送进 KnowledgeIndex。
    """
    from .thresholds import MANUAL_FOR_TYPE

    doc = MANUAL_FOR_TYPE.get(anomaly.device_type, "")
    # 一律带上 metric_label：手册里写的就是"轴承温度""振动烈度""出口压力"
    # 这些词，而不是"温度_℃"。用列名拼查询等于自己丢掉最关键的锚点。
    head = f"{anomaly.device_type} {anomaly.metric_label}"
    rising, falling = SYMPTOMS.get(anomaly.metric, ("异常", "持续异常"))

    if anomaly.layer == "threshold":
        query = (
            f"{head}超过 {_num(anomaly.boundary)}{anomaly.unit} {rising} 处置"
            if anomaly.boundary is not None else f"{head} {rising} 处置"
        )
    elif anomaly.layer == "trend":
        query = f"{head} {falling} 劣化趋势 原因 处置"
    elif anomaly.layer == "multivariate":
        query = f"{anomaly.device_type} 振动 温度 电流 同步上升 早期劣化 状态检修"
    else:  # zscore
        query = f"{head} 异常波动 {rising} 原因 处置"

    return query, doc


# ---------------------------------------------------------------- 单设备总入口


DEFAULT_LAYERS = frozenset({"threshold", "zscore", "trend", "multivariate"})


def detect_device(
    df: pd.DataFrame,
    device: dict,
    layers: set[str] | frozenset[str] | None = None,
) -> list[Anomaly]:
    """对一台设备运行指定检测层，返回全部原始发现。

    layers=None 表示跑满四层；评测消融时传入单层集合，复用同一套检测代码，
    避免"评测副本"和线上逻辑各写一份、最后对不上。
    """
    from .thresholds import MULTIVARIATE_FEATURES

    active = DEFAULT_LAYERS if layers is None else frozenset(layers)
    unknown = active - DEFAULT_LAYERS
    if unknown:
        raise ValueError(f"未知检测层：{', '.join(sorted(unknown))}")

    device_type, model = device["设备类型"], device["设备型号"]
    found: list[Anomaly] = []

    for band in bands_for(device_type, model):
        if band.metric not in df.columns or not df[band.metric].notna().any():
            continue
        if "threshold" in active:
            found.extend(detect_threshold(df, band, device))
        if "zscore" in active:
            found.extend(detect_zscore(df, band, device))
        if "trend" in active:
            found.extend(detect_trend(df, band, device))

    features = MULTIVARIATE_FEATURES.get(device_type, [])
    if features and "multivariate" in active:
        found.extend(detect_multivariate(df, features, device))

    return found
