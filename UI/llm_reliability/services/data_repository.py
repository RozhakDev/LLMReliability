import os
import math
from typing import Dict, Any, Optional, List
import pandas as pd
import streamlit as st

from llm_reliability.domain import DataPaths, FilterCriteria, McNemarMetrics
from llm_reliability.core.runtime import detect_environment
from llm_reliability.core.settings import get_settings
from llm_reliability.core.constants import (
    DOMAIN_ALL,
    SYMMETRY_FILTER_FLIP,
    SYMMETRY_FILTER_CONSISTENT,
    GT_FILTER_CORRECT,
    GT_FILTER_INCORRECT,
    PROMPT_FILTER_ALL,
)

def _check_file(*paths: str) -> Optional[str]:
    """Memeriksa keberadaan file di berbagai kandidat path (fallback multi-lokasi)."""
    for p in paths:
        if p and os.path.exists(p) and os.path.isfile(p):
            return os.path.abspath(p)
    return None

class EvaluationRepository:
    """Repository data evaluasi reliabilitas dan bias posisi (Mendukung Colab Google Drive & UI/data lokal)."""

    @staticmethod
    def get_paths() -> DataPaths:
        """Memetakan seluruh path artefak hasil evaluasi secara dinamis sesuai profil lingkungan."""
        config = detect_environment()
        root = config.data_root

        cand_results = [
            os.path.join(root, "results"),
            root
        ]
        cand_processed = [
            os.path.join(root, "processed"),
            os.path.join(root, "data", "processed"),
            root
        ]

        res_dir = next((os.path.abspath(d) for d in cand_results if os.path.exists(d)), None)
        dt_dir = next((os.path.abspath(d) for d in cand_processed if os.path.exists(d)), None)

        def find_file(filename: str, preferred_dir: Optional[str]) -> Optional[str]:
            candidates = []
            if preferred_dir:
                candidates.append(os.path.join(preferred_dir, filename))
            candidates.append(os.path.join(root, filename))
            candidates.append(os.path.join(root, "results", filename))
            candidates.append(os.path.join(root, "processed", filename))
            return _check_file(*candidates)

        return DataPaths(
            results_dir=res_dir,
            data_dir=dt_dir,
            metrics_csv=find_file("evaluation_metrics_summary.csv", res_dir),
            metrics_parquet=find_file("evaluation_metrics_summary.parquet", res_dir),
            detailed_parquet=find_file("evaluation_results_detailed.parquet", res_dir),
            domain_csv=find_file("domain_accuracy_summary.csv", res_dir),
            indopref_parquet=find_file("indopref_eval_stratified.parquet", dt_dir),
            plot_accuracy=find_file("plot_accuracy_and_position_bias.png", res_dir),
            plot_consistency=find_file("plot_consistency_and_kappa.png", res_dir),
            plot_parsing_domain=find_file("plot_parsing_and_domain_acc.png", res_dir),
            plot_heatmaps=find_file("plot_positional_heatmaps.png", res_dir),
        )

    @classmethod
    @st.cache_data(show_spinner=False)
    def load_metrics_summary(cls) -> pd.DataFrame:
        """Memuat tabel ringkasan metrik evaluasi P1, P2, P3 dari hasil eksperimen riil."""
        paths = cls.get_paths()
        if paths.metrics_csv:
            return pd.read_csv(paths.metrics_csv)
        if paths.metrics_parquet:
            return pd.read_parquet(paths.metrics_parquet)

        return pd.DataFrame()

    @classmethod
    @st.cache_data(show_spinner=False)
    def load_domain_accuracy(cls) -> pd.DataFrame:
        """Memuat tabel akurasi per domain topik dari hasil eksperimen riil."""
        paths = cls.get_paths()
        if paths.domain_csv:
            df = pd.read_csv(paths.domain_csv)
            if 'is_correct' in df.columns:
                df['Akurasi (%)'] = df['is_correct'].round(2)
            return df

        df_paired = cls.load_paired_evaluations()
        if not df_paired.empty and 'domain' in df_paired.columns and 'is_robust_correct' in df_paired.columns:
            domain_acc = df_paired.groupby('domain')['is_robust_correct'].mean() * 100
            df_calc = domain_acc.reset_index()
            df_calc.columns = ['domain', 'Akurasi (%)']
            df_calc['Akurasi (%)'] = df_calc['Akurasi (%)'].round(2)
            return df_calc.sort_values(by='Akurasi (%)', ascending=False)

        return pd.DataFrame()

    @classmethod
    @st.cache_data(show_spinner=False)
    def load_paired_evaluations(cls) -> pd.DataFrame:
        """
        Memuat dan menggabungkan hasil evaluasi posisi simetris (AB dan BA)
        dengan dataset IndoPref.
        """
        paths = cls.get_paths()
        if not paths.detailed_parquet or not paths.indopref_parquet:
            config = detect_environment()
            st.error(
                f"[DATA_LOADER] File evaluasi lengkap tidak ditemukan pada direktori `{config.data_root}`. "
                "Pastikan file 'evaluation_results_detailed.parquet' dan 'indopref_eval_stratified.parquet' tersedia di UI/data/ atau Google Drive."
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
            'P1': 'P1 (Direct)',
            'P2': 'P2 (Multi-Criteria)',
            'P3': 'P3 (Mitigasi Bias)'
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
        if criteria.prompt_id != PROMPT_FILTER_ALL:
            p_code = criteria.prompt_id.split(" ")[0]
            filtered = filtered[filtered['prompt_id'] == p_code]

        if criteria.domain != DOMAIN_ALL:
            filtered = filtered[filtered['domain'] == criteria.domain]

        if criteria.symmetry_filter == SYMMETRY_FILTER_FLIP:
            filtered = filtered[filtered['is_flip'] == True]
        elif criteria.symmetry_filter == SYMMETRY_FILTER_CONSISTENT:
            filtered = filtered[filtered['is_consistent'] == True]

        if criteria.gt_filter == GT_FILTER_CORRECT:
            filtered = filtered[filtered['is_robust_correct'] == True]
        elif criteria.gt_filter == GT_FILTER_INCORRECT:
            filtered = filtered[filtered['is_robust_correct'] == False]

        return filtered

    @classmethod
    @st.cache_data(show_spinner=False)
    def get_dataset_statistics(cls) -> Dict[str, int]:
        """Menghitung metrik agregat volume sampel dan evaluasi empiris secara dinamis."""
        df_paired = cls.load_paired_evaluations()
        if df_paired.empty:
            return {
                "sample_count": 0,
                "paired_count": 0,
                "total_evaluations": 0
            }
        unique_samples = len(df_paired['sample_id'].unique())
        paired_count = len(df_paired)
        return {
            "sample_count": unique_samples,
            "paired_count": paired_count,
            "total_evaluations": paired_count * 2
        }

    @classmethod
    @st.cache_data(show_spinner=False)
    def get_mcnemar_data(cls) -> Dict[str, McNemarMetrics]:
        """Menghitung metrik uji signifikansi McNemar berorientasi objek Typed DataClass."""
        df_paired = cls.load_paired_evaluations()
        if df_paired.empty:
            return {}

        s = get_settings()
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
                is_significant=(pvalue < s.statistical_alpha)
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
