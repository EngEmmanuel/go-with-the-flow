from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

# ---------- helpers -----------------------------------------------------------

def windows_count(L: int, window_size: int = 10, max_overlap: float = 0.5) -> int:
    """
    Max number of windows of length 'window_size' from a clip of length L,
    with at most 'max_overlap' fractional overlap between consecutive windows.
    For max_overlap=0.5 and window_size=10 -> stride = 5.
    """
    if pd.isna(L):
        return 0
    L = int(L)
    if L < window_size:
        return 0
    stride = max(1, int(np.ceil(window_size * (1 - max_overlap))))  # e.g., 5
    return 1 + (L - window_size) // stride


def ascii_hist_from_counts(edges: np.ndarray, counts: np.ndarray, width: int = 40, symbol: str = "#") -> str:
    maxc = int(counts.max()) if len(counts) else 0
    lines = []
    for i in range(len(edges) - 1):
        bar_len = int(round(width * (counts[i] / maxc))) if maxc > 0 else 0
        lines.append(f"{edges[i]:6.2f} – {edges[i+1]:6.2f} | {int(counts[i]):5d} {symbol * bar_len}")
    return "\n".join(lines)


def basic_stats(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "range": np.nan, "n": 0}
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "range": float(x.max() - x.min()),
        "n": int(x.size),
    }


# ---------- EF sampler class -------------------------------------------------

@dataclass
class EFSampler:
    """
        Loads metadata.csv, filters by split, and generates new EF values via controlled perturbations.

        Strategy:
            1) Sample EF ~ U[min_ef, max_ef] (bounds from TRAIN/VAL)
            2) Accept only if |EF_sampled - EF_original| > min_diff; otherwise resample
            3) Save result to CSV
    """
    metadata_path: Path
    ef_col: str = "EF_Area"
    split_col: str = "split"
    min_diff: float = 0.0
    splits_for_bounds: List[str] = field(default_factory=lambda: ["TRAIN", "VAL"])  # which splits define min/max EF
    random_state: int = 0

    def __post_init__(self):
        self.metadata_path = Path(self.metadata_path)
        self.df = pd.read_csv(self.metadata_path)
        self.rng = np.random.RandomState(self.random_state)
        self._compute_ef_bounds()

    def _compute_ef_bounds(self) -> None:
        """Compute min/max EF from specified splits."""
        mask = self.df[self.split_col].isin(self.splits_for_bounds)
        ef_vals = pd.to_numeric(self.df.loc[mask, self.ef_col], errors="coerce").dropna()
        self.min_ef = float(ef_vals.min())
        self.max_ef = float(ef_vals.max())

    def sample_new_ef(self, ef_original: float) -> float:
        """
        Sample new EF value:
            - EF_sampled ~ U[min_ef, max_ef] (integer sampling)
            - Require |EF_sampled - EF_original| > min_diff; otherwise resample
        """
        min_ef_int = int(np.ceil(self.min_ef))
        max_ef_int = int(np.floor(self.max_ef))
        
        while True:
            sampled = self.rng.randint(min_ef_int, max_ef_int + 1)
            if abs(sampled - ef_original) > self.min_diff:
                return float(sampled)


    def generate_and_save(self, split: Optional[str] = None, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Generate new EF values for rows matching split (or all if split=None).
        Save result to output_path. Return the modified DataFrame (filtered to split only).
        """
        # Filter to the split of interest
        if split is not None:
            df_out = self.df.loc[self.df[self.split_col] == split].copy()
        else:
            df_out = self.df.copy()
        
        # Sample new EF for all rows in the filtered dataframe
        df_out[self.ef_col] = df_out[self.ef_col].apply(
            lambda ef: self.sample_new_ef(float(ef)) if pd.notna(ef) else ef
        )
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(output_path, index=False)
        
        return df_out


# ---------- planner class -----------------------------------------------------

@dataclass
class EFGenerationPlanner:
    """
    End-to-end planner:
      1) Equal-frequency (quantile) binning on EF_Area (or chosen column)
      2) Per-bin generation plan: select sources so that generated windows ~= real windows
      3) Explicit original vs target info per selected source
      4) Distribution summaries with ASCII histograms
    """
    df: pd.DataFrame
    n_bins: int = 5
    ef_col: str = "EF_Area"
    video_col: str = "video_name"
    patient_col_out: str = "patient_id"
    nbframe_col: str = "NbFrame"

    split_for_bins: Optional[str] = "TEST"   # which subset defines bin edges/medians (None => use all)
    plan_on_split: Optional[str] = "TEST"    # which subset to build the plan on (None => use all)

    # window / overlap assumptions for estimating #windows per clip
    window_size: int = 10
    max_overlap: float = 0.5  # ≤50% overlap ⇒ stride >= 5

    # generation constraints
    delta: float = 5.0                 # require |EF - median_bin| >= delta (EF points)
    random_state: int = 0
    allow_replacement: bool = False    # allow re-using sources if candidate pool is small

    # planning strategy
    match_strategy: str = "match_real_windows"  # or "fixed_sources"
    fixed_sources_per_bin: Optional[int] = None # used if match_strategy="fixed_sources"

    # internals / outputs
    df_binned: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    edges_: Optional[np.ndarray] = field(init=False, default=None)
    labels_: List[str] = field(init=False, default_factory=list)
    medians_df_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    # -------------- public API ----------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes binning + planning + summaries. Returns:
          (df_binned, plan_df, summary_df, summaries_dict)
        """
        self._bin_and_annotate()
        plan_df, summary_df = self._build_plan()
        summaries = self.distribution_summaries(plan_df)
        return self.df_binned, plan_df, summary_df, summaries

    # -------------- binning (integrates add_ef_quantile_bins) -----------------

    def _bin_and_annotate(self) -> None:
        df = self.df.copy()

        # Which rows define bin edges / medians?
        if self.split_for_bins is None:
            idx_bins = df.index
        else:
            idx_bins = df.index[df["split"] == self.split_for_bins]

        # Ensure numeric EF
        s = pd.to_numeric(df.loc[idx_bins, self.ef_col], errors="coerce")

        # Equal-frequency cuts; drop duplicates if ties collapse boundaries
        bin_codes, edges = pd.qcut(
            s, q=self.n_bins, labels=False, retbins=True, duplicates="drop"
        )
        self.edges_ = edges
        self.labels_ = [f"({edges[i]:.2f}, {edges[i+1]:.2f}]" for i in range(len(edges) - 1)]

        # Assign ef_bin only for the subset used to define bins
        df.loc[idx_bins, "ef_bin"] = pd.cut(
            df.loc[idx_bins, self.ef_col].astype(float),
            bins=edges,
            labels=False,
            include_lowest=True,
            right=True,
        ).astype("Int64")

        # Human-friendly labels
        df["ef_bin_label"] = df["ef_bin"].map(lambda b: self.labels_[int(b)] if pd.notna(b) else pd.NA)

        # Per-bin medians (on the same subset)
        medians_df = (
            df.loc[idx_bins]
              .groupby("ef_bin", dropna=True)[self.ef_col]
              .median()
              .rename("ef_bin_median")
              .reset_index()
        )
        self.medians_df_ = medians_df

        # Attach median per row (for rows with a bin)
        df = df.merge(medians_df, on="ef_bin", how="left")

        # Patient id from video_name
        if self.patient_col_out:
            df[self.patient_col_out] = df[self.video_col].str.split("_").str[0]

        # Estimated # windows per clip (based on NbFrame, stride from max_overlap)
        df["est_windows"] = df[self.nbframe_col].astype(float).apply(
            lambda L: windows_count(int(L) if not pd.isna(L) else 0,
                                    window_size=self.window_size,
                                    max_overlap=self.max_overlap)
        )

        self.df_binned = df

    # -------------- planning ---------------------------------------------------

    def _rows_for_planning(self) -> pd.DataFrame:
        if self.plan_on_split is None:
            return self.df_binned.copy()
        return self.df_binned.loc[self.df_binned["split"] == self.plan_on_split].copy()

    def _bins_sorted(self, df_sub: pd.DataFrame) -> List[int]:
        return sorted(int(b) for b in df_sub["ef_bin"].dropna().unique())

    def _real_windows_per_bin(self, df_sub: pd.DataFrame) -> Dict[int, int]:
        Rb: Dict[int, int] = {}
        for b, grp in df_sub.dropna(subset=["ef_bin"]).groupby("ef_bin"):
            Rb[int(b)] = int(grp["est_windows"].sum())
        return Rb

    def _bin_median(self, b: int) -> float:
        row = self.medians_df_.loc[self.medians_df_["ef_bin"] == b]
        if row.empty:
            raise ValueError(f"No median found for bin {b}. Did binning drop it due to ties?")
        return float(row["ef_bin_median"].iloc[0])

    def _candidate_pool(self, df_sub: pd.DataFrame, target_bin: int) -> pd.DataFrame:
        """Videos at least delta EF away from the bin median, yielding >=1 window."""
        med = self._bin_median(target_bin)
        cand = df_sub.loc[
            (df_sub[self.ef_col].astype(float) - med).abs() >= self.delta
        ].copy()
        cand = cand.loc[cand["est_windows"] > 0]

        # Explicit original vs target columns
        cand["original_ef"] = cand[self.ef_col].astype(float)
        cand["original_ef_bin"] = cand["ef_bin"].astype("Int64")
        cand["original_ef_bin_label"] = cand["ef_bin_label"]
        cand["target_ef_bin"] = int(target_bin)
        cand["target_ef"] = med
        cand["target_ef_bin_label"] = self.labels_[int(target_bin)] if int(target_bin) < len(self.labels_) else pd.NA
        return cand

    def _select_sources(
        self,
        cand: pd.DataFrame,
        Rb: int,
        rng: np.random.RandomState,
    ) -> pd.DataFrame:
        """Select sources to either match real windows or take a fixed number of sources."""
        if cand.empty:
            return cand

        # Shuffle once for reproducibility
        sel = cand.sample(frac=1.0, random_state=rng).reset_index(drop=True)

        if self.match_strategy == "fixed_sources":
            if not self.fixed_sources_per_bin:
                raise ValueError("fixed_sources_per_bin must be set when match_strategy='fixed_sources'.")
            if self.allow_replacement and len(sel) < self.fixed_sources_per_bin:
                reps = int(np.ceil(self.fixed_sources_per_bin / len(sel)))
                sel = pd.concat([sel] * reps, ignore_index=True)
            return sel.iloc[: self.fixed_sources_per_bin].copy()

        # Default: match_real_windows
        if self.allow_replacement:
            picks = []
            total = 0
            i = 0
            while total < Rb:
                row = sel.iloc[i % len(sel)]
                picks.append(row)
                total += int(row["est_windows"])
                i += 1
            return pd.DataFrame(picks).reset_index(drop=True)

        # Without replacement: take rows until cumulative windows >= Rb
        sel["cum_windows"] = sel["est_windows"].cumsum()
        mask = sel["cum_windows"] >= max(Rb, 0)
        if mask.any():
            cutoff = mask.idxmax()
            sel = sel.iloc[: cutoff + 1].drop(columns=["cum_windows"])
        else:
            sel = sel.drop(columns=["cum_windows"])
        return sel

    def _build_plan(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_sub = self._rows_for_planning()
        bins_sorted = self._bins_sorted(df_sub)
        Rb = self._real_windows_per_bin(df_sub)

        plan_rows: List[pd.DataFrame] = []
        summary_rows: List[Dict] = []
        rng = np.random.RandomState(self.random_state)

        for b in bins_sorted:
            # real windows & target EF
            R_b = int(Rb.get(b, 0))
            target_ef = self._bin_median(b)

            # candidates for this bin
            cand = self._candidate_pool(df_sub, b)
            if cand.empty:
                summary_rows.append(dict(target_bin=b, target_ef=target_ef, R_b=R_b, G_b=0, S_b=0, note="no candidates"))
                continue

            # pick sources
            sel = self._select_sources(cand, R_b, rng=rng)
            if sel.empty:
                summary_rows.append(dict(target_bin=b, target_ef=target_ef, R_b=R_b, G_b=0, S_b=0, note="selection empty"))
                continue

            # summarize
            G_b = int(sel["est_windows"].sum())
            S_b = int(len(sel))
            summary_rows.append(dict(target_bin=b, target_ef=target_ef, R_b=R_b, G_b=G_b, S_b=S_b, note=""))

            # keep essentials for the plan (explicit original vs target columns)
            keep_cols = [
                self.video_col, self.patient_col_out, self.nbframe_col,
                "original_ef", "original_ef_bin", "original_ef_bin_label",
                "target_ef", "target_ef_bin", "target_ef_bin_label",
                "est_windows"
            ]
            existing = [c for c in keep_cols if c in sel.columns]
            plan_rows.append(sel[existing].copy())

        plan_df = pd.concat(plan_rows, ignore_index=True) if plan_rows else pd.DataFrame(
            columns=[
                self.video_col, self.patient_col_out, self.nbframe_col,
                "original_ef", "original_ef_bin", "original_ef_bin_label",
                "target_ef", "target_ef_bin", "target_ef_bin_label",
                "est_windows"
            ]
        )
        # sort for readability
        if not plan_df.empty:
            plan_df = plan_df.sort_values(["target_ef_bin", self.video_col]).reset_index(drop=True)

        summary_df = pd.DataFrame(summary_rows, columns=["target_bin", "target_ef", "R_b", "G_b", "S_b", "note"])
        return plan_df, summary_df

    # -------------- summaries --------------------------------------------------

    def distribution_summaries(self, plan_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Builds stats + ASCII histograms for REAL (planning subset) vs SYNTHETIC (plan targets).
        Returns a dict with human-readable strings and raw counts.
        """
        df_sub = self._rows_for_planning()
        edges = self.edges_
        labels = self.labels_

        # Real EF distribution (clip-level)
        real_vals = pd.to_numeric(df_sub[self.ef_col], errors="coerce").dropna()
        real_bins = pd.cut(real_vals, bins=edges, include_lowest=True, right=True, labels=False)
        real_counts = real_bins.value_counts(sort=False).reindex(range(len(edges)-1), fill_value=0).to_numpy()

        # Synthetic EF distribution (clip-level, by target EF/bin in the plan)
        if plan_df is not None and not plan_df.empty:
            synth_bins = plan_df["target_ef_bin"].astype(int)
            synth_counts = pd.Series(synth_bins).value_counts(sort=False).reindex(range(len(edges)-1), fill_value=0).to_numpy()
            synth_vals = plan_df["target_ef"]
        else:
            synth_counts = np.zeros(len(edges)-1, dtype=int)
            synth_vals = pd.Series([], dtype=float)

        real_hist = ascii_hist_from_counts(edges, real_counts, width=40, symbol="#")
        synth_hist = ascii_hist_from_counts(edges, synth_counts, width=40, symbol="#")

        out = {
            "real_stats": basic_stats(real_vals),
            "synthetic_stats": basic_stats(pd.to_numeric(synth_vals, errors="coerce")),
            "real_counts": real_counts,
            "synthetic_counts": synth_counts,
            "bin_edges": edges,
            "bin_labels": labels,
            "real_histogram": real_hist,
            "synthetic_histogram": synth_hist,
        }
        return out


# ---------- example usage -----------------------------------------------------

if __name__ == "__main__":
    metadata_path = Path("./data/CAMUS_Processed_Frames/metadata.csv")
    df = pd.read_csv(metadata_path)
    HISTOGRAM = False
    SAMPLER = True

    if SAMPLER:
        split = 'val'
        ef_col="EF_Area"
        output_path = f"./evaluation/eval_plans/{split.lower()}_ef_gen_metadata.csv"


        sampler = EFSampler(
            metadata_path=metadata_path,
            ef_col=ef_col,
            split_col="split",
            min_diff=5.0,
            splits_for_bounds=["TRAIN", "VAL"],
            random_state=0
        )

        df_sampled = sampler.generate_and_save(
            split=split.upper(),
            output_path=output_path
        )

        print(f"\n== Original vs Sampled EF stats ({split.upper()} split) ==")
        original_stats = basic_stats(pd.to_numeric(df.loc[df["split"] == split.upper(), ef_col], errors="coerce"))
        sampled_stats = basic_stats(pd.to_numeric(df_sampled.loc[df_sampled["split"] == split.upper(), ef_col], errors="coerce"))
        print("Original EF stats:", original_stats)
        print("Sampled EF stats:", sampled_stats)




    # OLD APPROACH. DIDN'T USE EVERY DATUM AND REPEATED SOME UNEVENLY.
    if HISTOGRAM:
        planner = EFGenerationPlanner(
            df=df,
            n_bins=5,
            ef_col=ef_col,
            video_col="video_name",
            patient_col_out="patient_id",
            nbframe_col="NbFrame",
            split_for_bins="TEST",     # define bins/medians on TEST split
            plan_on_split="TEST",      # build selection plan on TEST split
            window_size=10,
            max_overlap=0.5,           # ≤50% overlap ⇒ stride 5
            delta=5.0,                 # require |EF - bin_median| >= 5
            random_state=0,
            allow_replacement=False,
            match_strategy="match_real_windows",  # or "fixed_sources"
            fixed_sources_per_bin=None
        )

        df_binned, plan_df, summary_df, summaries = planner.run()

        plan_df.to_csv("./evaluation/eval_plans/play_plan.csv", index=False)

        print("\n== Bin edges ==")
        print(planner.edges_)
        print("\n== Bin medians ==")
        print(planner.medians_df_)
        print("\n== Per-bin summary ==")
        print(summary_df)

        n = 10
        print(f"\n== Generation plan (first {n} rows) ==")
        print(plan_df.head(n))


        print("\n== Real EF stats ==")
        print(summaries["real_stats"])
        print("\n== Synthetic (target) EF stats ==")
        print(summaries["synthetic_stats"])

        print("\n== Real EF ASCII histogram ==")
        print(summaries["real_histogram"])
        print("\n== Synthetic EF ASCII histogram ==")
        print(summaries["synthetic_histogram"])
