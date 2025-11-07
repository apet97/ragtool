"""Metrics tracking and export for Clockify RAG system.

Priority #13: Export KPI metrics (ROI 5/10)

This module provides:
- Real-time metric collection (latency, cache hits, retrieval quality)
- Multiple export formats (JSON, Prometheus, CSV)
- Aggregation and reporting
- Thread-safe metric updates
"""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of a single metric value at a point in time."""
    timestamp: float
    value: float
    labels: Dict[str, str]


@dataclass
class AggregatedMetrics:
    """Aggregated statistics for a metric."""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    p50: float
    p95: float
    p99: float


class MetricsCollector:
    """Thread-safe metrics collector for tracking KPIs.

    Supports:
    - Counters (monotonically increasing values)
    - Gauges (point-in-time values)
    - Histograms (distribution of values)
    - Timers (duration measurements)
    """

    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector.

        Args:
            max_history: Maximum number of historical values to retain per metric
        """
        self._lock = threading.RLock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=max_history))
        self._labels: Dict[str, Dict[str, str]] = {}
        self._start_time = time.time()

    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Amount to increment (default: 1.0)
            labels: Optional metric labels
        """
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            if labels:
                self._labels[key] = labels

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric to a specific value.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional metric labels
        """
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            if labels:
                self._labels[key] = labels

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram observation.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional metric labels
        """
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            if labels:
                self._labels[key] = labels

    def time_operation(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.

        Args:
            name: Metric name
            labels: Optional metric labels

        Example:
            with metrics.time_operation("query_latency"):
                # ... operation ...
        """
        return TimerContext(self, name, labels)

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value.

        Args:
            name: Metric name
            labels: Optional metric labels

        Returns:
            Current counter value
        """
        with self._lock:
            key = self._make_key(name, labels)
            return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value.

        Args:
            name: Metric name
            labels: Optional metric labels

        Returns:
            Current gauge value or None if not set
        """
        with self._lock:
            key = self._make_key(name, labels)
            return self._gauges.get(key)

    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[AggregatedMetrics]:
        """Get aggregated statistics for a histogram.

        Args:
            name: Metric name
            labels: Optional metric labels

        Returns:
            Aggregated statistics or None if no data
        """
        with self._lock:
            key = self._make_key(name, labels)
            values = list(self._histograms.get(key, []))

            if not values:
                return None

            sorted_values = sorted(values)
            n = len(sorted_values)

            return AggregatedMetrics(
                count=n,
                sum=sum(values),
                min=sorted_values[0],
                max=sorted_values[-1],
                mean=sum(values) / n,
                p50=sorted_values[int(n * 0.5)],
                p95=sorted_values[int(n * 0.95)],
                p99=sorted_values[int(n * 0.99)]
            )

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._labels.clear()
            self._start_time = time.time()

    def export_json(self, include_histograms: bool = True) -> str:
        """Export all metrics as JSON.

        Args:
            include_histograms: Include histogram data (can be large)

        Returns:
            JSON string with all metrics
        """
        with self._lock:
            data = {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self._start_time,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histogram_stats": {}
            }

            # Add aggregated histogram stats
            for key in self._histograms.keys():
                name, labels = self._parse_key(key)
                stats = self.get_histogram_stats(name, labels)
                if stats:
                    data["histogram_stats"][key] = asdict(stats)

            # Optionally include raw histogram data
            if include_histograms:
                data["histogram_raw"] = {
                    key: list(values) for key, values in self._histograms.items()
                }

            return json.dumps(data, indent=2)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics
        """
        lines = []

        with self._lock:
            # Counters
            for key, value in self._counters.items():
                name, labels = self._parse_key(key)
                labels_str = self._format_prometheus_labels(labels)
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels_str} {value}")

            # Gauges
            for key, value in self._gauges.items():
                name, labels = self._parse_key(key)
                labels_str = self._format_prometheus_labels(labels)
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels_str} {value}")

            # Histograms (as summaries)
            processed = set()
            for key in self._histograms.keys():
                name, labels = self._parse_key(key)
                if name in processed:
                    continue
                processed.add(name)

                stats = self.get_histogram_stats(name, labels)
                if not stats:
                    continue

                labels_str = self._format_prometheus_labels(labels)
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count{labels_str} {stats.count}")
                lines.append(f"{name}_sum{labels_str} {stats.sum}")
                lines.append(f"{name}{{quantile=\"0.5\"{self._add_labels(labels)}}} {stats.p50}")
                lines.append(f"{name}{{quantile=\"0.95\"{self._add_labels(labels)}}} {stats.p95}")
                lines.append(f"{name}{{quantile=\"0.99\"{self._add_labels(labels)}}} {stats.p99}")

        return "\n".join(lines)

    def export_csv(self) -> str:
        """Export metrics as CSV.

        Returns:
            CSV-formatted metrics
        """
        lines = ["metric_type,metric_name,labels,value"]

        with self._lock:
            # Counters
            for key, value in self._counters.items():
                name, labels = self._parse_key(key)
                labels_str = json.dumps(labels) if labels else "{}"
                lines.append(f"counter,{name},{labels_str},{value}")

            # Gauges
            for key, value in self._gauges.items():
                name, labels = self._parse_key(key)
                labels_str = json.dumps(labels) if labels else "{}"
                lines.append(f"gauge,{name},{labels_str},{value}")

            # Histogram stats
            for key in self._histograms.keys():
                name, labels = self._parse_key(key)
                stats = self.get_histogram_stats(name, labels)
                if not stats:
                    continue

                labels_str = json.dumps(labels) if labels else "{}"
                lines.append(f"histogram_count,{name},{labels_str},{stats.count}")
                lines.append(f"histogram_mean,{name},{labels_str},{stats.mean}")
                lines.append(f"histogram_p50,{name},{labels_str},{stats.p50}")
                lines.append(f"histogram_p95,{name},{labels_str},{stats.p95}")
                lines.append(f"histogram_p99,{name},{labels_str},{stats.p99}")

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics.

        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            summary = {
                "uptime_seconds": time.time() - self._start_time,
                "total_counters": len(self._counters),
                "total_gauges": len(self._gauges),
                "total_histograms": len(self._histograms),
                "key_metrics": {}
            }

            # Add key metrics if available
            key_metrics = [
                "queries_total",
                "cache_hits",
                "cache_misses",
                "retrieval_latency_ms",
                "llm_latency_ms",
                "errors_total"
            ]

            for metric in key_metrics:
                # Check counter
                if metric in self._counters:
                    summary["key_metrics"][metric] = self._counters[metric]
                # Check histogram
                elif metric in self._histograms:
                    stats = self.get_histogram_stats(metric)
                    if stats:
                        summary["key_metrics"][metric] = {
                            "count": stats.count,
                            "mean": stats.mean,
                            "p95": stats.p95
                        }

            return summary

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _parse_key(self, key: str) -> tuple:
        """Parse a metric key into name and labels."""
        if "{" not in key:
            return key, {}

        name, labels_str = key.split("{", 1)
        labels_str = labels_str.rstrip("}")

        labels = {}
        if labels_str:
            for pair in labels_str.split(","):
                k, v = pair.split("=", 1)
                labels[k] = v

        return name, labels

    def _format_prometheus_labels(self, labels: Optional[Dict[str, str]]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_pairs) + "}"

    def _add_labels(self, labels: Optional[Dict[str, str]]) -> str:
        """Add labels to Prometheus metric (with leading comma)."""
        if not labels:
            return ""
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "," + ",".join(label_pairs)


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.collector.observe_histogram(self.name, elapsed_ms, self.labels)
        return False


# Global metrics collector instance
_global_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Global MetricsCollector instance
    """
    return _global_metrics


def increment_counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
    """Increment a global counter metric (convenience function)."""
    _global_metrics.increment_counter(name, value, labels)


def set_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Set a global gauge metric (convenience function)."""
    _global_metrics.set_gauge(name, value, labels)


def observe_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record a global histogram observation (convenience function)."""
    _global_metrics.observe_histogram(name, value, labels)


def time_operation(name: str, labels: Optional[Dict[str, str]] = None):
    """Time an operation using global metrics (convenience function)."""
    return _global_metrics.time_operation(name, labels)


# Standard KPI metric names
class MetricNames:
    """Standard metric names for consistency."""

    # Counters
    QUERIES_TOTAL = "queries_total"
    CACHE_HITS = "cache_hits"
    CACHE_MISSES = "cache_misses"
    ERRORS_TOTAL = "errors_total"
    REFUSALS_TOTAL = "refusals_total"

    # Histograms (latency in milliseconds)
    QUERY_LATENCY = "query_latency_ms"
    RETRIEVAL_LATENCY = "retrieval_latency_ms"
    EMBEDDING_LATENCY = "embedding_latency_ms"
    RERANK_LATENCY = "rerank_latency_ms"
    LLM_LATENCY = "llm_latency_ms"

    # Gauges
    CACHE_SIZE = "cache_size_entries"
    INDEX_SIZE = "index_size_chunks"
    ACTIVE_QUERIES = "active_queries"


__all__ = [
    "MetricsCollector",
    "MetricSnapshot",
    "AggregatedMetrics",
    "TimerContext",
    "get_metrics",
    "increment_counter",
    "set_gauge",
    "observe_histogram",
    "time_operation",
    "MetricNames",
]
