# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from collections import OrderedDict
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any, Optional

import icontract
import torch
from beartype import beartype
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.metric import Metric
from tqdm import tqdm

from equine import EquineGP
from equine.equine_gp import _Laplace

from .equine import Equine, EquineOutput
from .utils import generate_train_summary

BatchType = tuple[torch.Tensor, ...]


class _Laplace2(_Laplace):
    """
    A private class to compute a Laplace approximation to a Gaussian Process (GP)
    """

    @icontract.require(lambda self: self.training_parameters_set)
    def forward(self, x, mask=None, accumulate_precision: bool = True):
        if mask is None:
            f = self.feature_extractor(x)
        else:
            f = self.feature_extractor(x, mask)

        f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize(f_reduc)
        k = self.rff(f_reduc)
        pred = self.beta(k)

        if self.training:
            if accumulate_precision:  # <-- gate
                precision_minibatch = k.t() @ k
                self.precision = self.precision + precision_minibatch
                self.seen_data += x.shape[0]
                assert self.seen_data <= self.num_data, (
                    "Did not reset precision matrix at start of epoch"
                )

        else:
            assert self.seen_data > (self.num_data - self.train_batch_size), (
                "Not seen sufficient data for precision matrix"
            )

            if self.recompute_covariance:
                with torch.no_grad():
                    eps = 1e-7
                    jitter = eps * torch.eye(
                        self.precision.shape[1],
                        device=self.precision.device,
                    )
                    u, info = torch.linalg.cholesky_ex(self.precision + jitter)
                    assert (info == 0).all(), "Precision matrix inversion failed!"
                    torch.cholesky_inverse(u, out=self.covariance)

                self.recompute_covariance: bool = False

            with torch.no_grad():
                pred_cov = k @ ((self.covariance @ k.t()) * self.ridge_penalty)

            if self.mean_field_factor is None:
                return pred, pred_cov
            else:
                pred = self.mean_field_logits(pred, pred_cov, self.mean_field_factor)

        return pred, f


def latent_hilbert_transform(x):
    """
    Computes the Hilbert transform of a real-valued signal using the FFT.

    Args:
        x (torch.Tensor): Input signal, must be real-valued and 1D.

    Returns:
        torch.Tensor: The Hilbert transform of the input signal.
    """

    N = x.shape[-1]
    xf = torch.fft.fft(x).to(x.device)

    # Create the Hilbert transform filter
    h = torch.zeros(N).to(x.device)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    # Apply the filter in the frequency domain
    xf = xf * h

    # Inverse FFT to get the Hilbert transform
    x_sequence = torch.fft.ifft(xf)
    # return torch.fft.ifft(xf).real.contiguous().to(x.device), torch.fft.ifft(xf).imag.contiguous().to(x.device)
    return torch.cat((x_sequence.real, x_sequence.imag), dim=-1).to(x.device)


# --- 2. Collate: pad ragged pulses + mask ---
def pulse_collate_fn(batch):
    features, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in features])
    max_w = int(lengths.max())
    D = features[0].shape[-1]
    B = len(features)
    padded = torch.zeros(B, max_w, D)
    mask = torch.zeros(B, max_w, dtype=torch.bool)
    for i, f in enumerate(features):
        padded[i, : f.shape[0]] = f
        mask[i, : f.shape[0]] = True
    return padded, mask, torch.stack(labels)


def paired_pulse_collate_fn(batch):
    clean_list, noise_list, labels = zip(*batch)

    def pad_and_mask(feats):
        lengths = torch.tensor([f.shape[0] for f in feats])
        max_w = int(lengths.max())
        D = feats[0].shape[-1]
        B = len(feats)
        padded = torch.zeros(B, max_w, D)
        mask = torch.zeros(B, max_w, dtype=torch.bool)
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f
            mask[i, : f.shape[0]] = True
        return padded, mask

    clean_padded, clean_mask = pad_and_mask(clean_list)
    noise_padded, noise_mask = pad_and_mask(noise_list)
    return clean_padded, clean_mask, noise_padded, noise_mask, torch.stack(labels)


# -------------------------------------------------------------------------------
# EquineGP, below, demonstrates how to adapt that approach in EQUINE
@beartype
class EquineGP2(EquineGP):
    """
    An example of an EQUINE model that builds upon the approach in "Spectral Norm
    Gaussian Processes" (SNGP). This wraps any pytorch embedding neural network and provides
    the `forward`, `predict`, `save`, and `load` methods required by Equine.

    Notes
    -----
    Although this model build upon the approach in SNGP, it does not enforce the spectral normalization
    and ResNet architecture required for SNGP. Instead, it is a simple wrapper around
    any pytorch embedding neural network. Your mileage may vary.
    """

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        emb_out_dim: int,
        num_classes: int,
        num_random_features: int = 1024,
        init_temperature: float = 1.0,
        device: str = "cpu",
        feature_names: Optional[list[str]] = None,
        label_names: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the EquineGP model.

        Parameters
        ----------
        embedding_model : torch.nn.Module
            Neural Network feature embedding.
        emb_out_dim : int
            The number of deep features from the feature embedding.
        num_classes : int
            The number of output classes this model predicts.
        num_random_features : int
            The dimension of the output of the RandomFourierFeatures operation
        init_temperature : float, optional
            What to use as the initial temperature (1.0 has no effect).
        device : str, optional
            Either 'cuda' or 'cpu'.
        feature_names : list[str], optional
            List of strings of the names of the tabular features (ex ["duration", "fiat_mean", ...])
        label_names : list[str], optional
            List of strings of the names of the labels (ex ["streaming", "voip", ...])
        """
        laplace_model = _Laplace2(
            embedding_model,
            emb_out_dim,
            emb_out_dim,
            True,
            num_random_features,
            num_classes,
            2.0,
            25,
            1,
        )
        super().__init__(
            embedding_model,
            emb_out_dim,
            num_classes,
            num_random_features,
            init_temperature,
            device,
            feature_names=feature_names,
            label_names=label_names,
            laplace_model=laplace_model,
        )

    def train_model(
        self,
        dataset: Dataset,
        loss_fn: Callable,
        opt: torch.optim.Optimizer,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        batch_size: int = 64,
        validation_dataset: Optional[Dataset] = None,
        val_metrics: Optional[Iterable[Metric]] = None,
        vis_support: bool = False,
        support_size: int = 25,
        sampler: WeightedRandomSampler | None = None,
        use_collate_fn: bool = False,
        paired: bool = False,
    ) -> dict[str, Any]:
        """
        Train or fine-tune an EquineGP model.

        Parameters
        ----------
        dataset : TensorDataset
            An iterable, pytorch TensorDataset.
        loss_fn : Callable
            A pytorch loss function, e.g., torch.nn.CrossEntropyLoss().
        opt : torch.optim.Optimizer
            A pytorch optimizer, e.g., torch.optim.Adam().
        num_epochs : int
            The desired number of epochs to use for training.
        scheduler : torch.optim.LRScheduler
            A pytorch scheduler, if one is desired
        validation_dataset: Dataset
            If provided, will compute validation metrics on this dataset after each epoch of training
        batch_size : int, optional
            The number of samples to use per batch.

        Returns
        -------
        dict[str, Any]
            A dict containing a dict of summary stats and a dataloader for the calibration data.

        """

        self.validate_feature_label_names(dataset[0][0].shape[-1], self.num_outputs)
        collate_fn_of_interest = paired_pulse_collate_fn if paired else pulse_collate_fn
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if sampler is None else False,
            drop_last=False,
            sampler=sampler,
            collate_fn=collate_fn_of_interest if use_collate_fn else None,
        )

        val_loader: Optional[DataLoader] = None
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn_of_interest if use_collate_fn else None,
            )

        self.model.set_training_params(len(dataset), batch_size)
        val_metrics_outputs: Optional[list[list[float]]] = None

        if validation_dataset is not None and val_metrics is not None:
            val_metrics_outputs = [[] for _ in range(len(list(val_metrics)))]

        for _ in tqdm(range(num_epochs)):
            self.model.train()
            self.model.reset_precision_matrix()
            epoch_loss = 0.0
            if use_collate_fn and paired:
                for i, (xs_c, mask_c, xs_n, mask_n, labels) in enumerate(train_loader):
                    opt.zero_grad()
                    xs_c, mask_c = xs_c.to(self.device), mask_c.to(self.device)
                    xs_n, mask_n = xs_n.to(self.device), mask_n.to(self.device)
                    labels = labels.to(self.device)

                    # Clean forward: accumulates precision (defines ID density)
                    logits_c, emb_c = self.model(
                        xs_c, mask_c, accumulate_precision=True
                    )
                    # Noise forward: invariance only, NO precision accumulation
                    logits_n, emb_n = self.model(
                        xs_n, mask_n, accumulate_precision=False
                    )

                    loss = loss_fn(
                        (logits_c, emb_c, logits_n, emb_n),
                        labels.to(torch.long),
                        step=i,
                        train_flag=True,
                    )
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()

            elif use_collate_fn:
                for i, (xs, mask, labels) in enumerate(train_loader):
                    opt.zero_grad()
                    xs = xs.to(self.device)
                    mask = mask.to(self.device)
                    labels = labels.to(self.device)
                    yhats = self.model(xs, mask)
                    loss = loss_fn(
                        yhats, labels.to(torch.long), step=i, train_flag=True
                    )
                    if i % 1000 == 0:
                        print(f"LOSS DETAILS = {loss}")
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
            else:
                # for i, (xs, labels) in enumerate(train_loader):
                #     opt.zero_grad()
                #     xs, xs_noise = xs[:, 0, :], xs[:, 1, :]
                #     xs = xs.to(self.device)
                #     xs_noise = xs_noise.to(self.device)
                #     labels = labels.to(self.device)
                #     yhats = self.model(xs)
                #     yhats_noise = self.model(xs_noise)
                #     loss = loss_fn((yhats, yhats_noise), labels.to(torch.long), step = i, train_flag = True)

                for i, (xs_c, xs_n, labels) in enumerate(train_loader):
                    opt.zero_grad()
                    xs_c = xs_c.to(self.device)
                    xs_n = xs_n.to(self.device)
                    labels = labels.to(self.device)

                    # Clean forward: accumulates precision (defines ID density)
                    logits_c, emb_c = self.model(xs_c, accumulate_precision=True)
                    # Noise forward: invariance only, NO precision accumulation
                    logits_n, emb_n = self.model(xs_n, accumulate_precision=False)

                    loss = loss_fn(
                        (logits_c, emb_c, logits_n, emb_n),
                        labels.to(torch.long),
                        step=i,
                        train_flag=True,
                    )

                    if i % 1000 == 0:
                        print(f"LOSS DETAILS = {loss}")
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
            if scheduler is not None:
                scheduler.step()
            self.model.eval()
            # compute the validation metrics
            if (
                validation_dataset is not None
                and val_loader is not None
                and val_metrics is not None
                and val_metrics_outputs is not None
            ):
                if use_collate_fn:
                    for _, (xs_val, mask_val, labels_val) in enumerate(val_loader):
                        xs_val = xs_val.to(self.device)
                        labels_val = labels_val.to(self.device)
                        mask_val = mask_val.to(self.device)
                        yhats_val = self.model(xs_val, mask_val)
                        for metric in val_metrics:
                            metric.update(yhats_val, labels_val)
                else:
                    for _, (xs_val, labels_val) in enumerate(val_loader):
                        xs_val, _xs_noise_val = xs_val[:, 0, :], xs_val[:, 1, :]
                        xs_val = xs_val.to(self.device)
                        # xs_noise_val = xs_noise_val.to(self.device)
                        labels_val = labels_val.to(self.device)
                        yhats_val = self.model(xs_val)
                        # yhats_noise_val = self.model(xs_noise_val)
                        for metric in val_metrics:
                            metric.update(yhats_val, labels_val)
                for i, metric in enumerate(val_metrics):
                    val_metrics_outputs[i].append(metric.compute())
                    print(f"VAL METRIC OUTPUT: {metric}-{val_metrics_outputs[i]}")
        if vis_support:
            self.update_support(dataset.tensors[0], dataset.tensors[1], support_size)

        # _, train_y = dataset[:]
        if hasattr(dataset, "get_all_labels"):
            train_y = dataset.get_all_labels()
        elif hasattr(dataset, "tensors"):
            train_y = dataset.tensors[1]
        else:
            train_y = torch.stack([dataset[i][-1] for i in range(len(dataset))])

        date_trained = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.train_summary: dict[str, Any] = generate_train_summary(
            self, train_y, date_trained
        )

        return_dict: dict[str, Any] = dict()
        return_dict["train_summary"] = self.train_summary
        if validation_dataset is not None:
            return_dict["val_metrics"] = val_metrics_outputs

        return return_dict

    def compute_embeddings(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Deep GP-feature embeddings (post-RFF), mask-aware. Inference-safe.
        Method for computing deep embeddings for given input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for generating embeddings.

        Returns
        -------
        torch.Tensor
            Output embeddings .
        """
        x = x.to(self.device)
        if mask is None:
            f = self.model.feature_extractor(x)
        else:
            mask = mask.to(self.device)
            f = self.model.feature_extractor(x, mask)
        f_reduc = self.model.jl(f)
        if self.model.normalize_gp_features:
            f_reduc = self.model.normalize(f_reduc)
        return self.model.rff(f_reduc)

    @icontract.require(lambda num_calibration_epochs: 0 < num_calibration_epochs)
    @icontract.require(lambda calibration_lr: calibration_lr > 0.0)
    def calibrate_model(
        self,
        dataset: torch.utils.data.Dataset,
        num_calibration_epochs: int = 1,
        calibration_lr: float = 0.01,
        calibration_batch_size: int = 256,
        use_collate_fn: bool = False,
    ) -> None:
        """
        Fine-tune the temperature after training. Note this function is also run at the conclusion of train_model.

        Parameters
        ----------
        dataset : TensorDataset
            An iterable, pytorch TensorDataset.
        num_calibration_epochs : int, optional
            Number of epochs to tune temperature.
        calibration_lr : float, optional
            Learning rate for temperature optimization.
        """

        calibration_loader = DataLoader(
            dataset,
            batch_size=calibration_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=pulse_collate_fn if use_collate_fn else None,
        )

        self.temperature.requires_grad = True
        loss_fn = torch.nn.functional.cross_entropy
        optimizer = torch.optim.Adam([self.temperature], lr=calibration_lr)
        for _ in range(num_calibration_epochs):
            if not use_collate_fn:
                for xs, labels in calibration_loader:
                    optimizer.zero_grad()
                    xs, xs_noise = xs[:, 0, :], xs[:, 1, :]
                    xs = xs.to(self.device)
                    xs_noise = xs_noise.to(self.device)
                    labels = labels.to(self.device)
                    with torch.no_grad():
                        logits, emb = self.model(xs)
                        logits_noise, emb_noise = self.model(xs_noise)
                    logits = logits / self.temperature
                    logits_noise = logits_noise / self.temperature
                    loss = loss_fn(
                        ((logits, emb), (logits_noise, emb_noise)),
                        labels.to(torch.long),
                    )
                    loss.backward()
                    optimizer.step()
            else:
                for xs, mask, labels in calibration_loader:
                    optimizer.zero_grad()
                    xs = xs.to(self.device)
                    labels = labels.to(self.device)
                    mask = mask.to(self.device)
                    with torch.no_grad():
                        logits, emb = self.model(xs, mask)
                    logits = logits / self.temperature
                    loss = loss_fn((logits, emb), labels.to(torch.long))
                    loss.backward()
                    optimizer.step()
        self.temperature.requires_grad = False

    def forward(
        self, X: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inference forward: returns temperature-scaled logits only.
        Training does NOT call this — it calls self.model(...) directly to get embeddings.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor for generating predictions.

        Returns
        -------
        torch.Tensor
            Output probabilities computed.
        """
        X = X.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            preds, _ = self.model(
                X, mask
            )  # _Laplace returns (pred, f) — discard f at inference
        else:
            preds, _ = self.model(X)
        return preds / self.temperature.to(self.device)

    @icontract.ensure(
        lambda result: bool(
            ((0.0 <= result.ood_scores) & (result.ood_scores <= 1.0)).all()
        )
    )
    def predict(
        self, X: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> EquineOutput:
        """
        Inference entry point. Forces eval mode; never touches training-only logic
        (no paired/noise views, no split_and_align, no precision accumulation).

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        EquineOutput
            Output object containing prediction probabilities and OOD scores.
        """
        self.eval()  # guarantee eval: no precision update
        X = X.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        with torch.no_grad():
            logits = self.forward(X, mask)  # (B, C) temperature-scaled logits
            preds = torch.softmax(logits, dim=1)  # FIX: single tensor, no tuple unpack

            equiprobable = (
                torch.ones(self.num_outputs, device=preds.device) / self.num_outputs
            )
            max_entropy = torch.sum(torch.special.entr(equiprobable))
            ood_score = torch.sum(torch.special.entr(preds), dim=1) / max_entropy

            embeddings = self.compute_embeddings(
                X, mask
            )  # mask-aware, single forward path

        return EquineOutput(classes=preds, ood_scores=ood_score, embeddings=embeddings)

    def save(self, path: str) -> None:
        """
        Function to save all model parameters to a file.

        Parameters
        ----------
        path : str
            Filename to write the model.
        """
        model_settings = {
            "emb_out_dim": self.num_deep_features,
            "num_classes": self.num_outputs,
            "num_random_features": self.num_random_features,
            "init_temperature": self.temperature.item(),
            "device": self.device_type,
        }

        # --- Option A: save the embedder as a plain state_dict (NO torch.jit.script) ---
        embedder_state_dict = self.model.feature_extractor.state_dict()

        # Strip feature_extractor keys from the laplace state_dict (as before),
        # since we save the embedder separately above.
        laplace_sd = self.model.state_dict()
        keys_to_delete = [k for k in laplace_sd if "feature_extractor" in k]
        for key in keys_to_delete:
            del laplace_sd[key]

        save_data = {
            "embedder_state_dict": embedder_state_dict,  # <-- replaces "embed_jit_save"
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "laplace_model_save": laplace_sd,
            "num_data": self.model.num_data,
            "settings": model_settings,
            "support": getattr(self, "support", OrderedDict()),  # safe if never set
            "train_batch_size": self.model.train_batch_size,
            "train_summary": self.train_summary,
        }

        torch.save(save_data, path)

    @classmethod
    def load_with_embedder(cls, path: str, embedder: torch.nn.Module) -> Equine:
        """
        Load a previously saved EquineGP model (Option A: state_dict-based).

        Parameters
        ----------
        path : str
            Input filename.

        embedder : torch.nn.Module
            A freshly-constructed embedder with the SAME architecture/config used
            at training time. Its weights will be overwritten by the saved ones.
        """
        model_save = torch.load(path, weights_only=False)

        # Build EquineGP around the provided (fresh) embedder
        eq_model = cls(embedder, **model_save.get("settings"))

        # Load the embedder weights
        eq_model.model.feature_extractor.load_state_dict(
            model_save.get("embedder_state_dict")
        )

        eq_model.feature_names = model_save.get("feature_names")
        eq_model.label_names = model_save.get("label_names")
        eq_model.train_summary = model_save.get("train_summary")

        # Load the non-embedder laplace state (beta, normalize, rff buffers, precision,
        # covariance, seen_data, etc.) with strict=False since feature_extractor keys absent.
        missing, unexpected = eq_model.model.load_state_dict(
            model_save.get("laplace_model_save"), strict=False
        )
        # sanity: only feature_extractor keys should be "missing"
        unexpected_bad = list(unexpected)
        missing_bad = [k for k in missing if "feature_extractor" not in k]
        assert not unexpected_bad, f"Unexpected keys on load: {unexpected_bad}"
        assert not missing_bad, f"Unexpectedly missing keys on load: {missing_bad}"

        # Restore GP bookkeeping
        eq_model.model.seen_data = model_save.get("laplace_model_save").get("seen_data")
        eq_model.model.set_training_params(
            model_save.get("num_data"), model_save.get("train_batch_size")
        )
        eq_model.eval()

        support = model_save.get("support")
        if support is not None and len(support) > 0:
            eq_model.support = support
            eq_model.prototypes = eq_model.compute_prototypes()

        return eq_model
