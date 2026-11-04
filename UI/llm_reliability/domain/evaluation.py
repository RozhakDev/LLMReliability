from dataclasses import dataclass
from typing import Optional, Dict, Any

from llm_reliability.core.constants import (
    PROMPT_FILTER_ALL,
    DOMAIN_ALL,
    SYMMETRY_FILTER_ALL,
    GT_FILTER_ALL
)

@dataclass(frozen=True)
class DataPaths:
    """Kontrak path artefak evaluasi dan dataset di Google Drive."""
    results_dir: Optional[str] = None
    data_dir: Optional[str] = None
    metrics_csv: Optional[str] = None
    metrics_parquet: Optional[str] = None
    detailed_parquet: Optional[str] = None
    domain_csv: Optional[str] = None
    indopref_parquet: Optional[str] = None
    plot_accuracy: Optional[str] = None
    plot_consistency: Optional[str] = None
    plot_parsing_domain: Optional[str] = None
    plot_heatmaps: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "results_dir": self.results_dir,
            "data_dir": self.data_dir,
            "metrics_csv": self.metrics_csv,
            "metrics_parquet": self.metrics_parquet,
            "detailed_parquet": self.detailed_parquet,
            "domain_csv": self.domain_csv,
            "indopref_parquet": self.indopref_parquet,
            "plot_accuracy": self.plot_accuracy,
            "plot_consistency": self.plot_consistency,
            "plot_parsing_domain": self.plot_parsing_domain,
            "plot_heatmaps": self.plot_heatmaps,
        }

@dataclass(frozen=True)
class FilterCriteria:
    """Kriteria penyaringan data evaluasi pada sidebar."""
    prompt_id: str = PROMPT_FILTER_ALL
    domain: str = DOMAIN_ALL
    symmetry_filter: str = SYMMETRY_FILTER_ALL
    gt_filter: str = GT_FILTER_ALL

@dataclass(frozen=True)
class McNemarMetrics:
    """Hasil kalkulasi uji signifikansi statistik McNemar."""
    statistic: float
    pvalue: float
    b: int
    c: int
    total_flips: int
    is_significant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "b": self.b,
            "c": self.c,
            "total_flips": self.total_flips,
            "is_significant": self.is_significant
        }

@dataclass(frozen=True)
class DualInferenceResult:
    """Hasil komparasi inferensi simetri ganda (AB dan BA)."""
    parsed_AB: str
    raw_output_AB: str
    logical_decision_AB: int
    runtime_AB: float
    parsed_BA: str
    raw_output_BA: str
    logical_decision_BA: int
    runtime_BA: float
    is_consistent: bool
    is_flip: bool
    mode: str
    ram_usage_mb: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    device_name: str = "CPU"

    def __getitem__(self, item: str) -> Any:
        """Mendukung akses berbasis index string untuk backward compatibility."""
        return getattr(self, item)
