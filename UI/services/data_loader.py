import os
import math
from typing import Dict, Any, Optional, List
import pandas as pd
import streamlit as st

from schemas import DataPaths, FilterCriteria, McNemarMetrics

PROJECT_ROOT: str = os.environ.get("PROJECT_ROOT", "/content/drive/MyDrive/LLMReliability")
RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data", "processed")

def _check_file(path: str) -> Optional[str]:
    """Memeriksa keberadaan file di Google Drive."""
    return os.path.abspath(path) if os.path.exists(path) and os.path.isfile(path) else None

class EvaluationRepository:
    """Repository data evaluasi reliabilitas dan bias posisi murni Google Drive."""

    @staticmethod
    def get_paths() -> DataPaths:
        """Memetakan seluruh path artefak hasil evaluasi secara eksklusif dari Google Drive."""
        res_dir = os.path.abspath(RESULTS_DIR) if os.path.exists(RESULTS_DIR) else None
        dt_dir = os.path.abspath(DATA_DIR) if os.path.exists(DATA_DIR) else None

        return DataPaths(
            results_dir=res_dir,
            data_dir=dt_dir,
            metrics_csv=_check_file(os.path.join(RESULTS_DIR, "evaluation_metrics_summary.csv")),
            metrics_parquet=_check_file(os.path.join(RESULTS_DIR, "evaluation_metrics_summary.parquet")),
            detailed_parquet=_check_file(os.path.join(RESULTS_DIR, "evaluation_results_detailed.parquet")),
            domain_csv=_check_file(os.path.join(RESULTS_DIR, "domain_accuracy_summary.csv")),
            indopref_parquet=_check_file(os.path.join(DATA_DIR, "indopref_eval_stratified.parquet")),
            plot_accuracy=_check_file(os.path.join(RESULTS_DIR, "plot_accuracy_and_position_bias.png")),
            plot_consistency=_check_file(os.path.join(RESULTS_DIR, "plot_consistency_and_kappa.png")),
            plot_parsing_domain=_check_file(os.path.join(RESULTS_DIR, "plot_parsing_and_domain_acc.png")),
            plot_heatmaps=_check_file(os.path.join(RESULTS_DIR, "plot_positional_heatmaps.png")),
        )

    @classmethod
    @st.cache_data(show_spinner=False)
    def load_metrics_summary(cls) -> pd.DataFrame:
        """Memuat tabel ringkasan metrik evaluasi P1, P2, P3 dari Google Drive."""
        paths = cls.get_paths()
        if paths.metrics_csv:
            return pd.read_csv(paths.metrics_csv)
        if paths.metrics_parquet:
            return pd.read_parquet(paths.metrics_parquet)

        return pd.DataFrame([
            {"Prompt": "P1", "Accuracy (%)": 72.54, "Consistency (%)": 73.99, "Flip Rate (%)": 26.01, "Cohen Kappa": 0.451, "Kappa 95% CI": "[0.393, 0.504]"},
            {"Prompt": "P2", "Accuracy (%)": 70.23, "Consistency (%)": 69.36, "Flip Rate (%)": 30.64, "Cohen Kappa": 0.405, "Kappa 95% CI": "[0.349, 0.465]"},
            {"Prompt": "P3", "Accuracy (%)": 70.33, "Consistency (%)": 63.78, "Flip Rate (%)": 36.22, "Cohen Kappa": 0.407, "Kappa 95% CI": "[0.355, 0.464]"}
        ])

    @classmethod
    @st.cache_data(show_spinner=False)
    def load_domain_accuracy(cls) -> pd.DataFrame:
        """Memuat tabel akurasi per domain topik dari Google Drive."""
        paths = cls.get_paths()
        if paths.domain_csv:
            df = pd.read_csv(paths.domain_csv)
            if 'is_correct' in df.columns:
                df['Akurasi (%)'] = df['is_correct'].round(2)
            return df

        return pd.DataFrame([
            {"domain": "analysis", "Akurasi (%)": 86.33},
            {"domain": "open question", "Akurasi (%)": 83.67},
            {"domain": "creative writing", "Akurasi (%)": 80.00},
            {"domain": "summarization", "Akurasi (%)": 72.67},
            {"domain": "brainstorming", "Akurasi (%)": 70.00},
            {"domain": "logic", "Akurasi (%)": 69.50},
            {"domain": "coding", "Akurasi (%)": 68.33},
            {"domain": "safety", "Akurasi (%)": 62.32},
            {"domain": "math", "Akurasi (%)": 61.56},
            {"domain": "translation", "Akurasi (%)": 58.04}
        ])

    @classmethod
    @st.cache_data(show_spinner=False)
    def load_paired_evaluations(cls) -> pd.DataFrame:
        """
        Memuat dan menggabungkan hasil evaluasi posisi simetris (AB dan BA)
        dengan dataset IndoPref dari Google Drive.
        """
        paths = cls.get_paths()
        if not paths.detailed_parquet or not paths.indopref_parquet:
            st.error(
                f"[DATA_LOADER] File evaluasi lengkap tidak ditemukan di Google Drive (`{PROJECT_ROOT}`). "
                "Pastikan Anda telah menjalankan notebook deployment dan Google Drive telah dimount."
            )
            return pd.DataFrame()

        df_detailed = pd.read_parquet(paths.detailed_parquet)
        df_indopref = pd.read_parquet(paths.indopref_parquet)

        df_ab = df_detailed[df_detailed['position'] == 'AB'].copy()
        df_ba = df_detailed[df_detailed['position'] == 'BA'].copy()

        merged_pos = pd.merge(
            df_ab,
            df_ba,
            on=['sample_id', 'prompt_id', 'domain'],
            suffixes=('_AB', '_BA')
        )

        merged_pos['is_consistent'] = merged_pos['logical_decision_AB'] == merged_pos['logical_decision_BA']
        merged_pos['is_flip'] = ~merged_pos['is_consistent']
        merged_pos['ground_truth'] = merged_pos['logical_human_AB']

        merged_pos['is_correct_AB'] = merged_pos['logical_decision_AB'] == merged_pos['ground_truth']
        merged_pos['is_correct_BA'] = merged_pos['logical_decision_BA'] == merged_pos['ground_truth']
        merged_pos['is_robust_correct'] = merged_pos['is_correct_AB'] & merged_pos['is_correct_BA']

        stratified_cols = ['original_id', 'input', 'response_1', 'response_2', 'chosen_model', 'rejected_model']
        available_cols = [c for c in stratified_cols if c in df_indopref.columns]

        full_df = pd.merge(
            merged_pos,
            df_indopref[available_cols],
            left_on='sample_id',
            right_on='original_id',
            how='inner'
        )

        strategy_map = {
            'P1': 'P1 (Baseline Direct)',
            'P2': 'P2 (Multi-Criteria)',
            'P3': 'P3 (Debiasing RCAF)'
        }
        full_df['strategy_name'] = full_df['prompt_id'].map(strategy_map)
        return full_df

    @classmethod
    def get_available_domains(cls) -> List[str]:
        """Mendapatkan daftar unik domain topik yang tersedia."""
        df = cls.load_paired_evaluations()
        if not df.empty and 'domain' in df.columns:
            return sorted(df['domain'].unique().tolist())
        return []

    @classmethod
    def filter_evaluations(cls, criteria: FilterCriteria) -> pd.DataFrame:
        """Menyaring dataset pasangan evaluasi berdasarkan kriteria filter (SRP)."""
        df = cls.load_paired_evaluations()
        if df.empty:
            return df

        filtered = df.copy()
        if criteria.prompt_id != "Semua (Overall)":
            p_code = criteria.prompt_id.split(" ")[0]
            filtered = filtered[filtered['prompt_id'] == p_code]

        if criteria.domain != "Semua Domain":
            filtered = filtered[filtered['domain'] == criteria.domain]

        if criteria.symmetry_filter == "Hanya Kasus Flip (Bias Posisi)":
            filtered = filtered[filtered['is_flip'] == True]
        elif criteria.symmetry_filter == "Hanya Kasus Konsisten":
            filtered = filtered[filtered['is_consistent'] == True]

        if criteria.gt_filter == "Sesuai Ground Truth (Kedua Posisi)":
            filtered = filtered[filtered['is_robust_correct'] == True]
        elif criteria.gt_filter == "Tidak Sesuai Ground Truth":
            filtered = filtered[filtered['is_robust_correct'] == False]

        return filtered

    @classmethod
    @st.cache_data(show_spinner=False)
    def get_mcnemar_data(cls) -> Dict[str, McNemarMetrics]:
        """Menghitung metrik uji signifikansi McNemar berorientasi objek Typed DataClass."""
        df_paired = cls.load_paired_evaluations()
        if df_paired.empty:
            return {}

        mcnemar_dict: Dict[str, McNemarMetrics] = {}
        for p_id in ['ALL', 'P1', 'P2', 'P3']:
            sub = df_paired if p_id == 'ALL' else df_paired[df_paired['prompt_id'] == p_id]
            b = int(((sub['logical_decision_AB'] == 1) & (sub['logical_decision_BA'] == 2)).sum())
            c = int(((sub['logical_decision_AB'] == 2) & (sub['logical_decision_BA'] == 1)).sum())
            total_flips = b + c

            if total_flips > 0:
                stat = ((abs(b - c) - 1.0) ** 2) / float(total_flips)
                pvalue = math.erfc(math.sqrt(stat) / math.sqrt(2.0))
            else:
                stat = 0.0
                pvalue = 1.0

            mcnemar_dict[p_id] = McNemarMetrics(
                statistic=round(stat, 4),
                pvalue=pvalue,
                b=b,
                c=c,
                total_flips=total_flips,
                is_significant=(pvalue < 0.05)
            )
        return mcnemar_dict

    @classmethod
    @st.cache_data(show_spinner=False)
    def get_prompt_stability_matrix(cls) -> pd.DataFrame:
        """Menghitung tingkat ketidakstabilan keputusan antar-strategi prompt."""
        df_paired = cls.load_paired_evaluations()
        if df_paired.empty:
            return pd.DataFrame()

        piv = df_paired[df_paired['position_AB'] == 'AB'].pivot_table(
            index='sample_id',
            columns='prompt_id',
            values='logical_decision_AB'
        ).dropna()

        if piv.empty:
            return pd.DataFrame()

        p1_p2 = float((piv['P1'] != piv['P2']).mean() * 100)
        p1_p3 = float((piv['P1'] != piv['P3']).mean() * 100)
        p2_p3 = float((piv['P2'] != piv['P3']).mean() * 100)

        return pd.DataFrame({
            "Pasangan Strategi": ["P1 vs P2", "P1 vs P3", "P2 vs P3"],
            "Tingkat Perubahan Keputusan (%)": [round(p1_p2, 2), round(p1_p3, 2), round(p2_p3, 2)],
            "Tingkat Keselarasan (%)": [round(100 - p1_p2, 2), round(100 - p1_p3, 2), round(100 - p2_p3, 2)],
        })

def get_data_paths() -> Dict[str, Optional[str]]:
    return EvaluationRepository.get_paths().to_dict()

def load_metrics_summary() -> pd.DataFrame:
    return EvaluationRepository.load_metrics_summary()

def load_domain_accuracy() -> pd.DataFrame:
    return EvaluationRepository.load_domain_accuracy()

def load_paired_evaluations() -> pd.DataFrame:
    return EvaluationRepository.load_paired_evaluations()

def get_mcnemar_data() -> Dict[str, Any]:
    raw_dict = EvaluationRepository.get_mcnemar_data()
    return {k: v.to_dict() for k, v in raw_dict.items()}

def get_prompt_stability_matrix() -> pd.DataFrame:
    return EvaluationRepository.get_prompt_stability_matrix()
