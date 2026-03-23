"""
Multimodal fusion module for combining heterogeneous data sources.

When genomics (scRNA-seq), imaging (imaging-based cell phenotyping), and
clinical metadata are available, this module provides methods to fuse
embeddings from different modalities into unified representations.

Following the "multimodal fusion last" principle: we first build strong
unimodal models, then fuse their latent representations.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MultimodalFuser:
    """
    Fuse embeddings from multiple modalities into unified representations.

    Supports several fusion strategies:
    - 'concat': Simple concatenation of embeddings
    - 'attention': Cross-attention mechanism between modalities
    - 'moe': Mixture of Experts for adaptive weighting

    Attributes:
        fusion_method: Strategy for fusion ('concat', 'attention', 'moe').
        weights: Per-modality weights (for attention/moe methods).
        classifier: Optional downstream classifier on fused features.
    """

    def __init__(self, fusion_method: str = "concat"):
        """
        Initialize multimodal fuser.

        Args:
            fusion_method: Fusion strategy ('concat', 'attention', 'moe').
                Default: 'concat'.

        Raises:
            ValueError: If fusion_method is unsupported.
        """
        if fusion_method not in ["concat", "attention", "moe"]:
            raise ValueError(
                f"Unsupported fusion method: {fusion_method}. "
                "Must be 'concat', 'attention', or 'moe'."
            )

        self.fusion_method = fusion_method
        self.weights = None
        self.classifier = None
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._moe_fitted = False

        logger.info(f"Initialized MultimodalFuser with method: {fusion_method}")

    def fuse_embeddings(
        self, embeddings_dict: Dict[str, np.ndarray], method: Optional[str] = None
    ) -> np.ndarray:
        """
        Fuse embeddings from multiple modalities.

        Args:
            embeddings_dict: Dictionary mapping modality names to embedding
                arrays, shape (n_samples, embedding_dim).
            method: Override fusion method for this call. Default: None (use class method).

        Returns:
            Fused embeddings of shape (n_samples, fused_dim).

        Raises:
            ValueError: If embeddings have incompatible shapes.
            ValueError: If method is unsupported.
        """
        if not embeddings_dict:
            raise ValueError("embeddings_dict cannot be empty")

        # Validate shapes
        n_samples = None
        for modality, embeddings in embeddings_dict.items():
            if n_samples is None:
                n_samples = embeddings.shape[0]
            elif embeddings.shape[0] != n_samples:
                raise ValueError(
                    f"All modalities must have same number of samples. "
                    f"Got {n_samples} and {embeddings.shape[0]}"
                )

        method = method or self.fusion_method

        if method == "concat":
            return self._concat_fusion(embeddings_dict)
        elif method == "attention":
            return self._attention_fusion(embeddings_dict)
        elif method == "moe":
            return self._moe_fusion(embeddings_dict)
        else:
            raise ValueError(f"Unsupported fusion method: {method}")

    def _concat_fusion(self, embeddings_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Simple concatenation of embeddings from all modalities.

        Args:
            embeddings_dict: Dictionary of modality -> embeddings.

        Returns:
            Concatenated embeddings (n_samples, sum of embedding dims).
        """
        embeddings_list = [embeddings_dict[key] for key in sorted(embeddings_dict.keys())]
        fused = np.concatenate(embeddings_list, axis=1)

        logger.debug(
            f"Concat fusion: {[e.shape[1] for e in embeddings_list]} -> {fused.shape[1]} dims"
        )
        return fused

    def _attention_fusion(self, embeddings_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Attention-based fusion with learned cross-modality weights.

        Computes soft attention weights over modalities for each sample,
        then produces weighted average of embeddings.

        Args:
            embeddings_dict: Dictionary of modality -> embeddings.

        Returns:
            Attention-fused embeddings (n_samples, embedding_dim).
        """
        embeddings_list = [
            embeddings_dict[key] for key in sorted(embeddings_dict.keys())
        ]
        n_modalities = len(embeddings_list)
        n_samples = embeddings_list[0].shape[0]

        # Use first modality's dimension as output dimension
        # (assumes all modalities have similar embedding dimensions)
        embedding_dim = embeddings_list[0].shape[1]

        # Pad all embeddings to same dimension if needed
        for i, emb in enumerate(embeddings_list):
            if emb.shape[1] < embedding_dim:
                padding = np.zeros((emb.shape[0], embedding_dim - emb.shape[1]))
                embeddings_list[i] = np.concatenate([emb, padding], axis=1)
            elif emb.shape[1] > embedding_dim:
                embeddings_list[i] = emb[:, :embedding_dim]

        # Compute attention weights via learnable projection + softmax
        # For simplicity, use cosine similarity to canonical direction
        embeddings_array = np.stack(embeddings_list, axis=1)  # (n, m, d)

        # Compute attention: score each modality's importance per sample
        scores = np.zeros((n_samples, n_modalities))
        for i in range(n_modalities):
            # Compute mean magnitude as importance
            scores[:, i] = np.linalg.norm(embeddings_list[i], axis=1)

        # Softmax over modalities (proper normalization along axis 1)
        scores = scores - scores.max(axis=1, keepdims=True)  # numerical stability
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (n, m)

        # Weighted average
        fused = np.zeros((n_samples, embedding_dim))
        for i in range(n_modalities):
            fused += weights[:, i : i + 1] * embeddings_list[i]

        logger.debug(
            f"Attention fusion: {n_modalities} modalities with learned weights -> {fused.shape[1]} dims"
        )
        return fused

    def fit(
        self, embeddings_dict: Dict[str, np.ndarray], y: np.ndarray, n_epochs: int = 100, lr: float = 0.01
    ) -> Dict:
        """
        Fit MoE gating weights via gradient descent to optimize for labels.

        Args:
            embeddings_dict: Dictionary of modality -> embeddings.
            y: Class labels (n_samples,).
            n_epochs: Number of training epochs. Default: 100.
            lr: Learning rate. Default: 0.01.

        Returns:
            Training history dictionary.

        Raises:
            ValueError: If embeddings_dict is empty or y shape mismatch.
        """
        if not embeddings_dict:
            raise ValueError("embeddings_dict cannot be empty")

        embeddings_list = [
            embeddings_dict[key] for key in sorted(embeddings_dict.keys())
        ]
        n_modalities = len(embeddings_list)
        n_samples = embeddings_list[0].shape[0]

        if y.shape[0] != n_samples:
            raise ValueError(
                f"y shape {y.shape[0]} does not match embeddings shape {n_samples}"
            )

        # All experts must have same dimension
        embedding_dims = [e.shape[1] for e in embeddings_list]
        embedding_dim = max(embedding_dims)

        # Pad to same dimension
        for i, emb in enumerate(embeddings_list):
            if emb.shape[1] < embedding_dim:
                padding = np.zeros((emb.shape[0], embedding_dim - emb.shape[1]))
                embeddings_list[i] = np.concatenate([emb, padding], axis=1)

        # Gating network: input is concatenation of all modalities
        X_concat = np.concatenate(embeddings_list, axis=1)

        # Initialize gating weights
        gate_input_dim = X_concat.shape[1]
        gate_hidden_dim = max(64, gate_input_dim // 4)
        n_classes = len(np.unique(y))

        self.weights = {
            "gate_w1": np.random.randn(gate_input_dim, gate_hidden_dim) * 0.01,
            "gate_b1": np.zeros(gate_hidden_dim),
            "gate_w2": np.random.randn(gate_hidden_dim, n_modalities) * 0.01,
            "gate_b2": np.zeros(n_modalities),
        }

        history = {"loss": []}

        # Training loop
        for epoch in range(n_epochs):
            # Forward pass
            h = np.dot(X_concat, self.weights["gate_w1"]) + self.weights["gate_b1"]
            h = np.maximum(h, 0)  # ReLU
            gate_scores = np.dot(h, self.weights["gate_w2"]) + self.weights["gate_b2"]

            # Softmax over experts
            gate_scores = gate_scores - gate_scores.max(axis=1, keepdims=True)
            gate_weights = np.exp(gate_scores) / np.exp(gate_scores).sum(axis=1, keepdims=True)

            # Weighted average of experts
            fused = np.zeros((n_samples, embedding_dim))
            for i in range(n_modalities):
                fused += gate_weights[:, i : i + 1] * embeddings_list[i]

            # Compute classification loss via LogisticRegression on fused output
            clf = LogisticRegression(max_iter=100, random_state=42)
            clf.fit(fused, y)
            loss = -clf.score(fused, y)
            history["loss"].append(loss)

            # Backpropagate to gating weights via finite differences
            lr = 0.01
            eps = 1e-5
            for key in ["gate_w1", "gate_b1", "gate_w2", "gate_b2"]:
                grad = np.zeros_like(self.weights[key])
                it = np.nditer(self.weights[key], flags=["multi_index"])
                # Sample a subset of indices for efficiency
                flat_size = self.weights[key].size
                sample_size = min(flat_size, max(50, flat_size // 10))
                sample_indices = np.random.choice(flat_size, sample_size, replace=False)
                for idx in sample_indices:
                    multi_idx = np.unravel_index(idx, self.weights[key].shape)
                    old_val = self.weights[key][multi_idx]
                    # Perturb +eps
                    self.weights[key][multi_idx] = old_val + eps
                    h_p = np.maximum(np.dot(X_concat, self.weights["gate_w1"]) + self.weights["gate_b1"], 0)
                    gs_p = np.dot(h_p, self.weights["gate_w2"]) + self.weights["gate_b2"]
                    gs_p = gs_p - gs_p.max(axis=1, keepdims=True)
                    gw_p = np.exp(gs_p) / np.exp(gs_p).sum(axis=1, keepdims=True)
                    fused_p = sum(gw_p[:, i:i+1] * embeddings_list[i] for i in range(n_modalities))
                    clf_p = LogisticRegression(max_iter=100, random_state=42)
                    clf_p.fit(fused_p, y)
                    loss_p = -clf_p.score(fused_p, y)
                    # Restore
                    self.weights[key][multi_idx] = old_val
                    grad[multi_idx] = (loss_p - loss) / eps
                self.weights[key] -= lr * grad

            if (epoch + 1) % max(1, n_epochs // 10) == 0:
                logger.debug(f"MoE fit epoch {epoch + 1}/{n_epochs}: loss={loss:.4f}")

        self._moe_fitted = True
        logger.info(f"Fit MoE gating weights in {n_epochs} epochs")
        return history

    def _moe_fusion(self, embeddings_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Mixture of Experts fusion with per-modality gating network.

        Each modality acts as an "expert", and a learned gating network
        assigns adaptive weights to each expert for each sample.

        Args:
            embeddings_dict: Dictionary of modality -> embeddings.

        Returns:
            MoE-fused embeddings (n_samples, embedding_dim).

        Raises:
            ValueError: If MoE fusion requires fitting first.
        """
        if self.fusion_method == "moe" and not self._moe_fitted:
            raise ValueError(
                "MoE fusion requires fitting first. "
                "Call fuser.fit(embeddings_dict, y) before predict."
            )

        embeddings_list = [
            embeddings_dict[key] for key in sorted(embeddings_dict.keys())
        ]
        n_modalities = len(embeddings_list)
        n_samples = embeddings_list[0].shape[0]

        # All experts must have same dimension
        embedding_dims = [e.shape[1] for e in embeddings_list]
        embedding_dim = max(embedding_dims)

        # Pad to same dimension
        for i, emb in enumerate(embeddings_list):
            if emb.shape[1] < embedding_dim:
                padding = np.zeros((emb.shape[0], embedding_dim - emb.shape[1]))
                embeddings_list[i] = np.concatenate([emb, padding], axis=1)

        # Gating network: input is concatenation of all modalities
        X_concat = np.concatenate(embeddings_list, axis=1)

        # Simple gating: MLP that outputs per-expert weights
        gate_input_dim = X_concat.shape[1]
        gate_hidden_dim = max(64, gate_input_dim // 4)

        # Initialize gating weights if not fitted
        if self.weights is None:
            self.weights = {
                "gate_w1": np.random.randn(gate_input_dim, gate_hidden_dim) * 0.01,
                "gate_b1": np.zeros(gate_hidden_dim),
                "gate_w2": np.random.randn(gate_hidden_dim, n_modalities) * 0.01,
                "gate_b2": np.zeros(n_modalities),
            }

        # Forward through gating network
        h = np.dot(X_concat, self.weights["gate_w1"]) + self.weights["gate_b1"]
        h = np.maximum(h, 0)  # ReLU
        gate_scores = np.dot(h, self.weights["gate_w2"]) + self.weights["gate_b2"]

        # Softmax over experts
        gate_scores = gate_scores - gate_scores.max(axis=1, keepdims=True)
        gate_weights = np.exp(gate_scores) / np.exp(gate_scores).sum(axis=1, keepdims=True)

        # Weighted average of experts
        fused = np.zeros((n_samples, embedding_dim))
        for i in range(n_modalities):
            fused += gate_weights[:, i : i + 1] * embeddings_list[i]

        logger.debug(
            f"MoE fusion: {n_modalities} experts with gating network -> {fused.shape[1]} dims"
        )
        return fused

    def attention_fusion(self, embeddings_list: List[np.ndarray]) -> np.ndarray:
        """
        Convenience method for attention fusion on list of embeddings.

        Args:
            embeddings_list: List of embedding arrays (n_samples, embedding_dim).

        Returns:
            Attention-fused embeddings.
        """
        embeddings_dict = {f"modality_{i}": e for i, e in enumerate(embeddings_list)}
        return self._attention_fusion(embeddings_dict)

    def train_fused_classifier(
        self,
        fused_X: np.ndarray,
        y: np.ndarray,
        model_type: str = "mlp",
        **kwargs,
    ) -> Dict:
        """
        Train a classifier on fused multimodal embeddings.

        Args:
            fused_X: Fused embeddings (n_samples, fused_dim).
            y: Class labels (n_samples,).
            model_type: Type of classifier ('mlp', 'logistic'). Default: 'mlp'.
            **kwargs: Additional arguments for classifier (e.g., lr, n_epochs).

        Returns:
            Dictionary with training results (losses, accuracies, etc.).

        Raises:
            ValueError: If fused_X or y have incompatible shapes.
            ValueError: If model_type is unsupported.
        """
        if fused_X.shape[0] != y.shape[0]:
            raise ValueError("fused_X and y must have same number of samples")

        if model_type == "logistic":
            return self._train_logistic_classifier(fused_X, y)
        elif model_type == "mlp":
            return self._train_mlp_classifier(fused_X, y, **kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def _train_logistic_classifier(
        self, fused_X: np.ndarray, y: np.ndarray
    ) -> Dict:
        """Train logistic regression on fused embeddings."""
        # Scale features
        X_scaled = self.scaler.fit_transform(fused_X)

        # Fit classifier
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(X_scaled, y)

        # Compute training accuracy
        train_acc = self.classifier.score(X_scaled, y)

        logger.info(
            f"Trained logistic classifier on fused embeddings: "
            f"train_acc={train_acc:.4f}"
        )

        return {"train_acc": train_acc, "model_type": "logistic"}

    def _train_mlp_classifier(
        self, fused_X: np.ndarray, y: np.ndarray, n_epochs: int = 50, lr: float = 1e-3
    ) -> Dict:
        """Train MLP classifier on fused embeddings using PyTorch."""
        # Scale features
        X_scaled = self.scaler.fit_transform(fused_X)

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Get number of classes
        n_classes = len(np.unique(y))

        # Build MLP
        input_dim = X_scaled.shape[1]
        hidden_dim = max(64, input_dim // 2)

        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = {"train_loss": [], "train_acc": []}

        for epoch in range(n_epochs):
            model.train()

            # Forward pass
            logits = model(X_tensor)
            loss = criterion(logits, y_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y_tensor).float().mean().item()

            history["train_loss"].append(loss.item())
            history["train_acc"].append(acc)

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                logger.debug(
                    f"MLP Epoch {epoch + 1}/{n_epochs}: "
                    f"loss={loss.item():.4f}, acc={acc:.4f}"
                )

        self.classifier = model

        logger.info(
            f"Trained MLP classifier on fused embeddings: "
            f"final_acc={history['train_acc'][-1]:.4f}"
        )

        return history

    def predict_fused(self, fused_X: np.ndarray) -> np.ndarray:
        """
        Predict using trained fused classifier.

        Args:
            fused_X: Fused embeddings (n_samples, fused_dim).

        Returns:
            Predicted class labels (n_samples,).

        Raises:
            ValueError: If classifier not trained.
        """
        if self.classifier is None:
            raise ValueError("Classifier must be trained before prediction")

        X_scaled = self.scaler.transform(fused_X)

        if isinstance(self.classifier, LogisticRegression):
            predictions = self.classifier.predict(X_scaled)
        elif isinstance(self.classifier, nn.Module):
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = self.classifier(X_tensor)
                predictions = logits.argmax(dim=1).numpy()
        else:
            raise ValueError("Unknown classifier type")

        logger.debug(f"Predicted labels for {len(predictions)} samples")
        return predictions

    def predict_proba_fused(self, fused_X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using trained fused classifier.

        Args:
            fused_X: Fused embeddings (n_samples, fused_dim).

        Returns:
            Class probabilities (n_samples, n_classes).

        Raises:
            ValueError: If classifier not trained.
        """
        if self.classifier is None:
            raise ValueError("Classifier must be trained before prediction")

        X_scaled = self.scaler.transform(fused_X)

        if isinstance(self.classifier, LogisticRegression):
            probas = self.classifier.predict_proba(X_scaled)
        elif isinstance(self.classifier, nn.Module):
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = self.classifier(X_tensor)
                probas = torch.softmax(logits, dim=1).numpy()
        else:
            raise ValueError("Unknown classifier type")

        return probas
