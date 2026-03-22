"""
Data downloading module for the R3-MM pipeline.

This module provides functionality to download single-cell RNA-seq data from GEO
and convert it to AnnData format, handling 10x Genomics h5 supplementary files.
"""

import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import GEOparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GEODataDownloader:
    """
    Download single-cell RNA-seq data from GEO (Gene Expression Omnibus).

    Handles 10x Genomics datasets where expression data is stored in
    supplementary h5 files rather than in the SOFT format tables.
    """

    def __init__(
        self,
        ncbi_api_key: Optional[str] = None,
        timeout: int = 300,
        retry_attempts: int = 3,
        batch_size: int = 32,
    ):
        self.ncbi_api_key = ncbi_api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.batch_size = batch_size

    def download_gse(
        self,
        accession: str,
        output_dir: Path,
        name: str = None,
        force_download: bool = False,
    ) -> Tuple[Path, Dict]:
        """
        Download a GEO series (GSE) and convert to AnnData.

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting download of {accession}...")

        try:
            gse = GEOparse.get_GEO(geo=accession, how="full")
            logger.info(f"Successfully retrieved {accession} from GEO")

            metadata = self._extract_metadata(gse, accession, name)

            # Try supplementary h5 files first (10x Genomics scRNA-seq)
            adata = self._download_supplementary_h5(gse, accession, output_dir)

            if adata is None:
                # Try series-level supplementary matrix files
                adata = self._download_series_matrix(gse, accession, output_dir)

            if adata is None:
                # Fallback to SOFT table-based processing
                adata = self._process_gse_to_adata(gse, accession)

            output_path = output_dir / f"{accession}.h5ad"
            adata.write_h5ad(output_path)
            logger.info(f"Saved {accession} to {output_path}")

            metadata["output_path"] = str(output_path)
            metadata["n_obs"] = adata.n_obs
            metadata["n_vars"] = adata.n_vars

            return output_path, metadata

        except Exception as e:
            logger.error(f"Failed to download {accession}: {str(e)}")
            raise ValueError(f"Failed to download {accession}: {str(e)}")

    def _download_supplementary_h5(
        self, gse, accession: str, output_dir: Path
    ) -> Optional[ad.AnnData]:
        """Download and read supplementary h5 files from GSM samples."""
        adatas = []

        for gsm_id, gsm in tqdm(gse.gsms.items(), desc=f"Downloading {accession} samples"):
            # Look for h5 supplementary files
            h5_urls = []
            for key in sorted(gsm.metadata.keys()):
                if key.startswith("supplementary_file"):
                    for url in gsm.metadata[key]:
                        if url.endswith(".h5"):
                            h5_urls.append(url)

            if not h5_urls:
                continue

            for url in h5_urls:
                try:
                    local_path = output_dir / f"{gsm_id}.h5"
                    if not local_path.exists():
                        logger.info(f"Downloading {gsm_id}: {url}")
                        urllib.request.urlretrieve(url, str(local_path))

                    adata = sc.read_10x_h5(str(local_path))
                    adata.var_names_make_unique()
                    adata.obs_names = [f"{gsm_id}_{bc}" for bc in adata.obs_names]

                    # Add sample metadata
                    adata.obs["sample_id"] = gsm_id
                    adata.obs["title"] = gsm.metadata.get("title", [""])[0]
                    adata.obs["source_name"] = gsm.metadata.get("source_name_ch1", [""])[0]
                    chars = gsm.metadata.get("characteristics_ch1", [])
                    adata.obs["characteristics"] = " | ".join(chars)

                    adatas.append(adata)
                    logger.info(f"  {gsm_id}: {adata.n_obs} cells, {adata.n_vars} genes")
                    break  # Use first h5 file per sample

                except Exception as e:
                    logger.warning(f"Failed to read h5 for {gsm_id}: {e}")
                    continue

        if not adatas:
            return None

        if len(adatas) == 1:
            merged = adatas[0]
        else:
            merged = ad.concat(adatas, join="outer")
            merged.var_names_make_unique()

        merged.obs["dataset"] = accession
        logger.info(f"Downloaded {accession}: {merged.n_obs} cells, {merged.n_vars} genes")
        return merged

    def _download_series_matrix(
        self, gse, accession: str, output_dir: Path
    ) -> Optional[ad.AnnData]:
        """Download series-level supplementary expression matrix files."""
        suppl_files = gse.metadata.get("supplementary_file", [])
        matrix_urls = [
            url for url in suppl_files
            if any(kw in url.lower() for kw in ["matrix", "tpm", "count", "expression"])
            and url.lower().endswith((".txt.gz", ".csv.gz", ".tsv.gz"))
        ]

        if not matrix_urls:
            return None

        for url in matrix_urls:
            try:
                fname = url.split("/")[-1]
                local_path = output_dir / fname
                if not local_path.exists():
                    logger.info(f"Downloading series matrix: {url}")
                    urllib.request.urlretrieve(url, str(local_path))

                logger.info(f"Reading matrix: {local_path}")
                df = pd.read_csv(local_path, sep="\t", index_col=0, compression="gzip")
                logger.info(f"Matrix shape: {df.shape}")

                # Determine orientation: genes as rows x samples as columns is typical
                # If many more rows than columns, genes are rows
                if df.shape[0] > df.shape[1]:
                    # genes x samples -> transpose to samples x genes
                    adata = ad.AnnData(X=df.T.values.astype(np.float32))
                    adata.obs_names = pd.Index(df.columns)
                    adata.var_names = pd.Index(df.index)
                else:
                    # samples x genes
                    adata = ad.AnnData(X=df.values.astype(np.float32))
                    adata.obs_names = pd.Index(df.index)
                    adata.var_names = pd.Index(df.columns)

                adata.var_names_make_unique()
                adata.obs["dataset"] = accession

                # Try to add clinical info
                clinical_urls = [
                    u for u in suppl_files
                    if "clinical" in u.lower()
                ]
                for clin_url in clinical_urls:
                    try:
                        clin_fname = clin_url.split("/")[-1]
                        clin_path = output_dir / clin_fname
                        if not clin_path.exists():
                            urllib.request.urlretrieve(clin_url, str(clin_path))
                        clin_df = pd.read_csv(clin_path, sep="\t", index_col=0, compression="gzip")
                        common = adata.obs_names.intersection(clin_df.index)
                        if len(common) > 0:
                            for col in clin_df.columns:
                                adata.obs[col] = clin_df.loc[adata.obs_names, col].values
                            logger.info(f"Added clinical info: {len(clin_df.columns)} columns")
                    except Exception as e:
                        logger.warning(f"Failed to load clinical info: {e}")

                logger.info(f"Series matrix AnnData: {adata.n_obs} cells, {adata.n_vars} genes")
                return adata

            except Exception as e:
                logger.warning(f"Failed to read series matrix {url}: {e}")
                continue

        return None

    def _extract_metadata(self, gse, accession: str, name: Optional[str] = None) -> Dict:
        """Extract metadata from GSE object."""
        metadata = {
            "accession": accession,
            "name": name or accession,
            "title": gse.metadata.get("title", [""])[0],
            "summary": gse.metadata.get("summary", [""])[0],
            "organism": self._extract_organism(gse),
            "platform": self._extract_platform(gse),
            "n_samples": len(gse.gsms),
            "submission_date": gse.metadata.get("submission_date", [""])[0],
            "publication_date": gse.metadata.get("publication_date", [""])[0],
        }
        return metadata

    def _extract_organism(self, gse) -> str:
        for gsm in gse.gsms.values():
            organism = gsm.metadata.get("organism_ch1", [""])[0]
            if organism:
                return organism
        return "Unknown"

    def _extract_platform(self, gse) -> str:
        if gse.gsms:
            first_gsm = next(iter(gse.gsms.values()))
            return first_gsm.metadata.get("platform_id", [""])[0]
        return "Unknown"

    def _process_gse_to_adata(self, gse, accession: str) -> ad.AnnData:
        """Fallback: process GEO SOFT tables to AnnData."""
        logger.info(f"Processing {accession} from SOFT tables...")

        expression_data = []
        sample_ids = []
        sample_metadata = {}

        for gsm_id, gsm in tqdm(gse.gsms.items(), desc=f"Processing {accession}"):
            try:
                expr = gsm.table
                if expr is not None and not expr.empty:
                    expression_data.append(expr)
                    sample_ids.append(gsm_id)
                    sample_metadata[gsm_id] = {
                        "title": gsm.metadata.get("title", [""])[0],
                        "source_name_ch1": gsm.metadata.get("source_name_ch1", [""])[0],
                        "characteristics_ch1": " | ".join(
                            gsm.metadata.get("characteristics_ch1", [])
                        ),
                    }
            except Exception as e:
                logger.warning(f"Failed to process {gsm_id}: {str(e)}")
                continue

        if not expression_data:
            raise ValueError(f"No expression data found in {accession}")

        merged_expr = pd.concat(expression_data, axis=1)
        adata = ad.AnnData(
            X=merged_expr.T.values,
            var=pd.DataFrame(index=merged_expr.index),
            obs=pd.DataFrame(index=merged_expr.columns),
        )

        for sample_id in merged_expr.columns:
            if sample_id in sample_metadata:
                for key, value in sample_metadata[sample_id].items():
                    adata.obs.at[sample_id, key] = value

        adata.obs["dataset"] = accession
        logger.info(f"Created AnnData object: {adata.shape}")
        return adata


def download_gse_data(
    accessions: List[str],
    output_dir: str = "./data/raw",
    ncbi_api_key: Optional[str] = None,
    timeout: int = 300,
    retry_attempts: int = 3,
) -> Dict[str, Tuple[Path, Dict]]:
    """Download multiple GEO series and convert to AnnData format."""
    downloader = GEODataDownloader(
        ncbi_api_key=ncbi_api_key,
        timeout=timeout,
        retry_attempts=retry_attempts,
    )

    results = {}
    for accession in tqdm(accessions, desc="Downloading datasets"):
        try:
            output_path, metadata = downloader.download_gse(
                accession, Path(output_dir)
            )
            results[accession] = (output_path, metadata)
            logger.info(f"Successfully downloaded {accession}")
        except Exception as e:
            logger.error(f"Failed to download {accession}: {str(e)}")
            continue

    logger.info(f"Downloaded {len(results)} out of {len(accessions)} datasets")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    accessions = ["GSE271107", "GSE106218"]
    results = download_gse_data(accessions, output_dir="./data/raw")
    for accession, (path, metadata) in results.items():
        print(f"\n{accession}:")
        print(f"  Path: {path}")
        print(f"  Cells: {metadata.get('n_obs', 'N/A')}")
        print(f"  Genes: {metadata.get('n_vars', 'N/A')}")
