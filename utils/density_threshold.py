"""Pure helpers for event-density-aware Challenge 2 threshold analysis."""

from dataclasses import dataclass

from utils.challenge_eval import ChallengeMetrics, challenge_score


def _as_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', '1', 'yes', 'on'}:
            return True
        if normalized in {'false', '0', 'no', 'off'}:
            return False
        raise ValueError('Expected a boolean value, got {!r}.'.format(value))
    return bool(value)


@dataclass(frozen=True)
class DensityAdaptiveThresholdConfig:
    """Optional inference policy for density-dependent decision thresholds."""

    enabled: bool = False
    event_count_cutoff: int = 100000
    low_density_threshold: float = 0.70
    high_density_threshold: float = 0.92
    # Optional M169 audit policy.  It is disabled by default and preserves
    # the original two-density behavior unless explicitly enabled.
    polarity_domain_enabled: bool = False
    middle_event_count_cutoff: int = 200000
    middle_density_threshold: float = 0.728
    high_polarity_minority_cutoff: float = 0.20
    high_imbalanced_threshold: float = 0.722
    high_balanced_threshold: float = 0.724

    def __post_init__(self):
        select_density_threshold(
            0,
            self.event_count_cutoff,
            self.low_density_threshold,
            self.high_density_threshold,
        )

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, 'p6_density_threshold_enabled', False)),
            event_count_cutoff=int(getattr(cfg, 'p6_event_count_cutoff', 100000)),
            low_density_threshold=float(
                getattr(cfg, 'p6_low_density_threshold', 0.70)
            ),
            high_density_threshold=float(
                getattr(cfg, 'p6_high_density_threshold', 0.92)
            ),
            polarity_domain_enabled=_as_bool(
                getattr(cfg, 'p6_polarity_domain_enabled', False)
            ),
            middle_event_count_cutoff=int(
                getattr(cfg, 'p6_middle_event_count_cutoff', 200000)
            ),
            middle_density_threshold=float(
                getattr(cfg, 'p6_middle_density_threshold', 0.728)
            ),
            high_polarity_minority_cutoff=float(
                getattr(cfg, 'p6_high_polarity_minority_cutoff', 0.20)
            ),
            high_imbalanced_threshold=float(
                getattr(cfg, 'p6_high_imbalanced_threshold', 0.722)
            ),
            high_balanced_threshold=float(
                getattr(cfg, 'p6_high_balanced_threshold', 0.724)
            ),
        )

    def threshold_for_event_count(self, event_count, fallback_threshold):
        """Return the active per-video threshold or the original static value."""
        if not self.enabled:
            return float(fallback_threshold)
        return select_density_threshold(
            event_count,
            self.event_count_cutoff,
            self.low_density_threshold,
            self.high_density_threshold,
        )

    def threshold_for_sample(self, event_count, event_features, fallback_threshold):
        """Return a density/polarity-domain threshold for one full video.

        The polarity branch uses only the observed event stream.  The
        dataset exposes polarity as column 3 of ``evs_norm`` (``[x, y, t, p]``);
        ``ev_loc`` intentionally contains only ``[x, y, t]`` and must not be
        passed here.  Invalid or empty polarity data falls back to the
        ordinary event-count policy.
        """
        if not self.enabled or not self.polarity_domain_enabled:
            return self.threshold_for_event_count(event_count, fallback_threshold)
        count = int(event_count)
        if count <= int(self.event_count_cutoff):
            return float(self.low_density_threshold)
        if count <= int(self.middle_event_count_cutoff):
            return float(self.middle_density_threshold)
        try:
            if hasattr(event_features, 'detach'):
                polarity = event_features.detach().cpu().numpy()[:, 3]
            else:
                polarity = __import__('numpy').asarray(event_features)[:, 3]
            if polarity.size == 0:
                raise ValueError('empty polarity')
            positive = float((polarity > 0.5).mean())
            minority = min(positive, 1.0 - positive)
        except (IndexError, TypeError, ValueError):
            return self.threshold_for_event_count(event_count, fallback_threshold)
        if minority < float(self.high_polarity_minority_cutoff):
            return float(self.high_imbalanced_threshold)
        return float(self.high_balanced_threshold)

    def describe(self, fallback_threshold):
        if not self.enabled:
            return 'static ({:.3f})'.format(float(fallback_threshold))
        if self.polarity_domain_enabled:
            return (
                'density/polarity-domain (<= {} -> {:.3f}, <= {} -> {:.3f}, '
                'high imbalanced/balanced -> {:.3f}/{:.3f})'
            ).format(
                self.event_count_cutoff,
                self.low_density_threshold,
                self.middle_event_count_cutoff,
                self.middle_density_threshold,
                self.high_imbalanced_threshold,
                self.high_balanced_threshold,
            )
        return (
            'density-adaptive (event_count > {} -> {:.3f}, otherwise {:.3f})'
        ).format(
            self.event_count_cutoff,
            self.high_density_threshold,
            self.low_density_threshold,
        )


@dataclass(frozen=True)
class ChallengeCountTotals:
    """Sufficient statistics needed to reproduce the Challenge 2 metrics."""

    true_positive_events: int
    false_positive_events: int
    positive_events: int
    detected_target_frames: int
    target_frames: int
    false_components: int
    frame_count: int

    def __post_init__(self):
        values = (
            self.true_positive_events,
            self.false_positive_events,
            self.positive_events,
            self.detected_target_frames,
            self.target_frames,
            self.false_components,
            self.frame_count,
        )
        if any(int(value) < 0 for value in values):
            raise ValueError('Challenge counts must be non-negative.')
        if self.true_positive_events > self.positive_events:
            raise ValueError('true_positive_events cannot exceed positive_events.')
        if self.detected_target_frames > self.target_frames:
            raise ValueError('detected_target_frames cannot exceed target_frames.')


def select_density_threshold(
    event_count,
    event_count_cutoff,
    low_density_threshold,
    high_density_threshold,
):
    """Choose the threshold from observable per-video event density."""
    event_count = int(event_count)
    event_count_cutoff = int(event_count_cutoff)
    low_density_threshold = float(low_density_threshold)
    high_density_threshold = float(high_density_threshold)
    if event_count < 0 or event_count_cutoff < 0:
        raise ValueError('event counts must be non-negative.')
    if not 0.0 < low_density_threshold < 1.0:
        raise ValueError('low_density_threshold must be in (0, 1).')
    if not 0.0 < high_density_threshold < 1.0:
        raise ValueError('high_density_threshold must be in (0, 1).')
    if event_count > event_count_cutoff:
        return high_density_threshold
    return low_density_threshold


def aggregate_challenge_counts(counts, width=346, height=260):
    """Compute global Challenge 2 metrics from independently scored videos."""
    counts = tuple(counts)
    if not counts:
        raise ValueError('At least one video count record is required.')
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError('width and height must be positive.')

    true_positive_events = sum(item.true_positive_events for item in counts)
    false_positive_events = sum(item.false_positive_events for item in counts)
    positive_events = sum(item.positive_events for item in counts)
    detected_target_frames = sum(item.detected_target_frames for item in counts)
    target_frames = sum(item.target_frames for item in counts)
    false_components = sum(item.false_components for item in counts)
    frame_count = sum(item.frame_count for item in counts)
    if positive_events == 0 or target_frames == 0 or frame_count == 0:
        raise ValueError('Aggregated validation counts must contain positives, targets, and frames.')

    iou = true_positive_events / (positive_events + false_positive_events)
    acc = true_positive_events / positive_events
    pd = detected_target_frames / target_frames
    fa = false_components / (frame_count * width * height)
    score_fa, score = challenge_score(iou, acc, pd, fa)
    return ChallengeMetrics(
        iou=iou,
        acc=acc,
        pd=pd,
        fa=fa,
        score_fa=score_fa,
        score=score,
    )
