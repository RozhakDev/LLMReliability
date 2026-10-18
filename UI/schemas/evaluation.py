from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class DataPaths:
    """Kontrak path artefak evaluasi dan dataset di Google Drive."""
    results_dir: Optional[str]
    data_dir: Optional[str]
    metrics_csv: Optional[str]
    metrics_parquet: Optional[str]
    detailed_parquet: Optional[str]
    domain_csv: Optional[str]
    indopref_parquet: Optional[str]
    plot_accuracy: Optional[str]
    plot_consistency: Optional[str]
    plot_parsing_domain: Optional[str]
    plot_heatmaps: Optional[str]

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
    prompt_id: str = "Semua (Overall)"
    domain: str = "Semua Domain"
    symmetry_filter: str = "Semua Kasus"
    gt_filter: str = "Semua Hasil"

@dataclass(frozen=True)
class McNemarMetrics:
    """Hasil kalkulasi uji signifikansi statistik McNemar."""
    statistic: float
    pvalue: float
    b: int
    c: int
    total_flips: int
    is_significant: bool

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

    def __getitem__(self, item: str) -> Any:
        """Mendukung akses berbasis index string untuk backward compatibility."""
        return getattr(self, item)
