# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Tests that EQUINE models survive ``torch.compile``, ``torch.export``, and
AOTInductor.

These paths are easy to break: both tracers refuse to specialize on
tensor-valued ``assert``s and ``icontract`` postconditions, and any
``.shape ==`` comparison silently burns the batch size into the graph.
``torch.compile`` is exercised with the default ``fullgraph=False`` because
``predict`` legitimately graph-breaks (``scipy`` KDE, ``.item()`` calls).

Compilation is expensive, so ``max_examples`` is kept low; the drawn datasets
still vary the feature count, which is what reaches the traced graph.
"""

import io
import os
import shutil
import tempfile

import pytest
import torch
from conftest import BasicEmbeddingModel, random_dataset, use_basic_embedding_model
from hypothesis import HealthCheck, given, settings, strategies as st
from torch._inductor import cpp_builder

import equine as eq

ATOL = 1e-5
AOTI_ATOL = 1e-4  # AOTInductor fuses kernels, so it is looser than eager


def train_equine_gp(random_dataset) -> tuple[eq.Equine, int, torch.Tensor]:
    dataset, num_classes, X, embedding_model = use_basic_embedding_model(random_dataset)
    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001
    )
    model.train_model(dataset, torch.nn.CrossEntropyLoss(), optimizer, num_epochs=2)
    model.eval()
    return model, num_classes, X


def train_equine_protonet(random_dataset) -> tuple[eq.Equine, int, torch.Tensor]:
    dataset, num_classes, way = random_dataset
    X, _ = dataset.tensors
    num_deep_features = 32
    embed_model = BasicEmbeddingModel(X.shape[1], num_deep_features)
    model = eq.EquineProtonet(embed_model, num_deep_features)
    model.train_model(
        dataset, way=way, support_size=3, num_episodes=5, episode_size=512
    )
    model.eval()
    return model, num_classes, X


# Both EQUINE flavors must trace; each has its own tracing hazards.
train_model_fns = pytest.mark.parametrize(
    "train_model_fn",
    [train_equine_gp, train_equine_protonet],
    ids=["gp", "protonet"],
)

# AOTInductor emits and builds C++; without a toolchain there is nothing to test
# and the failure would be about the environment, not about EQUINE.
aoti_test = pytest.mark.aoti(
    pytest.mark.skipif(
        shutil.which(cpp_builder.get_cpp_compiler()) is None,
        reason="no C++ compiler available for AOTInductor",
    )
)


def aoti_package(exported_program, directory, name: str = "model") -> str:
    """Compile an ExportedProgram into a .pt2 archive and return its path."""
    return torch._inductor.aoti_compile_and_package(
        exported_program, package_path=f"{directory}/{name}.pt2"
    )


# --------------------------------------------------------------------------
# torch.compile
# --------------------------------------------------------------------------


@train_model_fns
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=2)
def test_compile_forward_matches_eager(train_model_fn, random_dataset, batch) -> None:
    model, num_classes, X = train_model_fn(random_dataset)
    x = X[:batch]

    compiled = torch.compile(model)(x)

    assert compiled.shape == (batch, num_classes)
    assert torch.allclose(compiled, model(x), atol=ATOL)


@train_model_fns
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=2)
def test_compile_predict_matches_eager(train_model_fn, random_dataset, batch) -> None:
    # predict() graph-breaks on purpose; it must still produce usable output.
    model, num_classes, X = train_model_fn(random_dataset)
    x = X[:batch]

    eq_out = model.predict(x)
    compiled_out = torch.compile(model.predict)(x)

    assert torch.allclose(compiled_out.classes, eq_out.classes, atol=ATOL)
    assert torch.allclose(compiled_out.ood_scores, eq_out.ood_scores, atol=ATOL)
    assert len(compiled_out.classes) == batch
    assert len(compiled_out.ood_scores) == batch
    assert torch.all(
        (compiled_out.ood_scores >= 0.0) & (compiled_out.ood_scores <= 1.0)
    )


@train_model_fns
@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_compile_handles_varying_batch_size(train_model_fn, random_dataset) -> None:
    # A recompile for a new batch size is fine; a crash or wrong shape is not.
    model, num_classes, X = train_model_fn(random_dataset)
    compiled = torch.compile(model)

    for batch in (1, 4, 16):
        out = compiled(X[:batch])
        assert out.shape == (batch, num_classes)
        assert torch.allclose(out, model(X[:batch]), atol=ATOL)


@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=2)
def test_compile_is_differentiable(random_dataset, batch) -> None:
    # Fine-tuning a compiled model must still produce gradients.
    model, _, X = train_equine_gp(random_dataset)
    x = X[:batch].clone().requires_grad_(True)

    torch.compile(model)(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------
# torch.export
# --------------------------------------------------------------------------


@train_model_fns
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=2)
def test_export_matches_eager(train_model_fn, random_dataset, batch) -> None:
    model, num_classes, X = train_model_fn(random_dataset)
    x = X[:batch]

    exported = torch.export.export(model, (x,))

    out = exported.module()(x)
    assert out.shape == (batch, num_classes)
    assert torch.allclose(out, model(x), atol=ATOL)


@train_model_fns
@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_export_dynamic_batch_runs_at_other_batch_sizes(
    train_model_fn, random_dataset
) -> None:
    # Regression test: a `.shape == torch.Size(...)` comparison in forward
    # specializes the batch dim and makes this raise.
    model, num_classes, X = train_model_fn(random_dataset)
    batch_dim = torch.export.Dim("batch", min=1, max=X.shape[0])

    exported = torch.export.export(
        model, (X[:8],), dynamic_shapes={"X": {0: batch_dim}}
    )

    for batch in (1, 5, 32):
        out = exported.module()(X[:batch])
        assert out.shape == (batch, num_classes)
        assert torch.allclose(out, model(X[:batch]), atol=ATOL)


@train_model_fns
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=1)
def test_exported_program_serialization_roundtrip(
    train_model_fn, random_dataset, batch
) -> None:
    model, _, X = train_model_fn(random_dataset)
    x = X[:batch]

    buffer = io.BytesIO()
    torch.export.save(torch.export.export(model, (x,)), buffer)
    buffer.seek(0)
    reloaded = torch.export.load(buffer)

    assert torch.allclose(reloaded.module()(x), model(x), atol=ATOL)


@train_model_fns
@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_exported_output_is_usable_downstream(train_model_fn, random_dataset) -> None:
    # The point of exporting is to consume the result: logits must still feed
    # the metrics helpers and produce a valid probability distribution.
    model, num_classes, X = train_model_fn(random_dataset)
    _, Y = random_dataset[0].tensors

    logits = torch.export.export(model, (X,)).module()(X)
    probabilities = torch.softmax(logits, dim=1)

    assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(X)), atol=ATOL), (
        "exported logits are not a valid distribution"
    )
    metrics = eq.utils.generate_model_metrics(
        eq.EquineOutput(
            classes=probabilities,
            ood_scores=torch.zeros(len(X)),
            embeddings=torch.zeros(len(X), 1),
        ),
        Y.long(),
    )
    assert metrics["brierScore"] >= 0.0


@train_model_fns
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=1)
def test_export_then_compile(train_model_fn, random_dataset, batch) -> None:
    # The common deployment path: export once, then compile the artifact.
    model, _, X = train_model_fn(random_dataset)
    x = X[:batch]

    exported_module = torch.export.export(model, (x,)).module()

    assert torch.allclose(torch.compile(exported_module)(x), model(x), atol=ATOL)


@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=2)
def test_export_does_not_mutate_model_state(random_dataset, batch) -> None:
    # _Laplace tracks `seen_data`/`covariance` as buffers; tracing in eval mode
    # must not advance them, or subsequent predictions would drift.
    model, _, X = train_equine_gp(random_dataset)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    recompute_before = model.model.recompute_covariance

    torch.export.export(model, (X[:batch],))

    after = model.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"export mutated buffer {key}"
    assert model.model.recompute_covariance == recompute_before


# --------------------------------------------------------------------------
# AOTInductor
#
# Ahead-of-time compilation of an ExportedProgram into a self-contained .pt2
# archive for non-Python inference. Follows
# https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_inductor.html
# These tests shell out to a C++ compiler, so they are marked `aoti` and can be
# deselected with `-m "not aoti"`.
# --------------------------------------------------------------------------


@aoti_test
@train_model_fns
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=1)
def test_aoti_compile_and_package_matches_eager(
    train_model_fn, random_dataset, batch
) -> None:
    model, num_classes, X = train_model_fn(random_dataset)
    x = X[:batch]

    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        path = aoti_package(torch.export.export(model, (x,)), directory)
        assert os.path.exists(path), "no .pt2 archive produced"
        assert os.path.getsize(path) > 0

        out = torch._inductor.aoti_load_package(path)(x)

    assert out.shape == (batch, num_classes)
    assert torch.allclose(out, model(x), atol=AOTI_ATOL)


@aoti_test
@train_model_fns
@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_aoti_dynamic_batch_runs_at_other_batch_sizes(
    train_model_fn, random_dataset
) -> None:
    model, num_classes, X = train_model_fn(random_dataset)
    # min=2: torch.export specializes dims of extent 0 and 1, so a batch of 1
    # is rejected by the AOTI runtime even when the dim is marked dynamic.
    batch_dim = torch.export.Dim("batch", min=2, max=X.shape[0])

    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        exported = torch.export.export(
            model, (X[:8],), dynamic_shapes={"X": {0: batch_dim}}
        )
        compiled = torch._inductor.aoti_load_package(
            aoti_package(exported, directory, "dynamic")
        )

        for batch in (2, 5, 32):
            out = compiled(X[:batch])
            assert out.shape == (batch, num_classes)
            assert torch.allclose(out, model(X[:batch]), atol=AOTI_ATOL)


@aoti_test
@train_model_fns
@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_aoti_static_package_is_unreliable_at_other_batch_sizes(
    train_model_fn, random_dataset
) -> None:
    # Sharp edge worth pinning: a package compiled without dynamic_shapes does
    # not reliably reject a different batch size, it can silently return the
    # wrong shape or wrong values (EquineProtonet returns the exported batch
    # size no matter what it is fed). So always pass dynamic_shapes if the batch
    # size varies. This is a canary: raising is fine, silently correct is not.
    model, num_classes, X = train_model_fn(random_dataset)
    exported_batch, other = 8, 9

    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        compiled = torch._inductor.aoti_load_package(
            aoti_package(
                torch.export.export(model, (X[:exported_batch],)), directory, "static"
            )
        )
        try:
            out = compiled(X[:other])
        except RuntimeError:
            return  # raising is an acceptable outcome

    silently_correct = out.shape == (other, num_classes) and torch.allclose(
        out, model(X[:other]), atol=AOTI_ATOL
    )
    assert not silently_correct, (
        "static AOTI package now handles other batch sizes correctly; "
        "the dynamic_shapes guidance in this test file can be relaxed"
    )


@aoti_test
@train_model_fns
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=1)
def test_aoti_rejects_wrong_dtype(train_model_fn, random_dataset, batch) -> None:
    # A dtype mismatch cannot be papered over: the kernels are compiled float32.
    model, _, X = train_model_fn(random_dataset)

    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        compiled = torch._inductor.aoti_load_package(
            aoti_package(torch.export.export(model, (X[:batch],)), directory, "dtype")
        )
        with pytest.raises(RuntimeError):
            compiled(X[:batch].double())


@aoti_test
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(
    deadline=None,
    max_examples=1,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_aoti_runtime_check_inputs_rejects_bad_inputs(
    random_dataset, batch, monkeypatch
) -> None:
    # AOTI_RUNTIME_CHECK_INPUTS=1 adds size/dtype/stride guards to the artifact.
    # It is read at codegen time, so the inductor cache must be bypassed or a
    # previously-built unguarded artifact gets reused.
    monkeypatch.setenv("AOTI_RUNTIME_CHECK_INPUTS", "1")
    model, _, X = train_equine_gp(random_dataset)
    x = X[:batch]

    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        compiled = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(
                torch.export.export(model, (x,)),
                package_path=f"{directory}/guarded.pt2",
                inductor_configs={"force_disable_caches": True},
            )
        )

        assert torch.allclose(compiled(x), model(x), atol=AOTI_ATOL)
        with pytest.raises(RuntimeError, match="unmatched dim value"):
            compiled(X[: batch + 1])
        with pytest.raises(RuntimeError, match="unmatched dtype"):
            compiled(x.double())


@aoti_test
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=1)
def test_aoti_from_saved_exported_program(random_dataset, batch) -> None:
    # The deployment split the docs call out: export on the training host, then
    # load and AOT-compile on the inference host.
    model, _, X = train_equine_gp(random_dataset)
    x = X[:batch]

    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        exported_path = f"{directory}/exported.pt2"
        torch.export.save(torch.export.export(model, (x,)), exported_path)
        reloaded = torch.export.load(exported_path)

        compiled = torch._inductor.aoti_load_package(
            aoti_package(reloaded, directory, "from_saved")
        )
        assert torch.allclose(compiled(x), model(x), atol=AOTI_ATOL)


@aoti_test
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=1)
def test_aoti_packages_both_models_in_one_archive(random_dataset, batch) -> None:
    # Multi-model archive: a webapp serving both EQUINE flavors ships one file.
    from torch._inductor.package import load_package, package_aoti

    models = {
        "gp": train_equine_gp(random_dataset)[0],
        "protonet": train_equine_protonet(random_dataset)[0],
    }
    X = random_dataset[0].tensors[0]
    x = X[:batch]

    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        shared_libraries = {
            name: torch._inductor.aot_compile(
                # aot_compile wants a GraphModule, not the ExportedProgram the
                # docs' multi-model snippet passes.
                torch.export.export(model, (x,)).module(),
                (x,),
                options={"aot_inductor.package": True},
            )
            for name, model in models.items()
        }
        archive = f"{directory}/both.pt2"
        package_aoti(archive, shared_libraries)

        for name, model in models.items():
            out = load_package(archive, name)(x)
            assert torch.allclose(out, model(x), atol=AOTI_ATOL), f"{name} differs"


@aoti_test
@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_aoti_output_is_usable_downstream(random_dataset) -> None:
    # An AOTI artifact is only useful if its logits still feed the EQUINE
    # post-processing that `predict` would have applied.
    model, num_classes, X = train_equine_gp(random_dataset)
    _, Y = random_dataset[0].tensors

    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        compiled = torch._inductor.aoti_load_package(
            aoti_package(torch.export.export(model, (X,)), directory, "downstream")
        )
        logits = compiled(X)

    probabilities = torch.softmax(logits, dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(X)), atol=AOTI_ATOL)

    equiprobable = torch.ones(num_classes) / num_classes
    max_entropy = torch.sum(torch.special.entr(equiprobable))
    ood_scores = torch.sum(torch.special.entr(probabilities), dim=1) / max_entropy
    assert torch.all((ood_scores >= 0.0) & (ood_scores <= 1.0 + ATOL))

    metrics = eq.utils.generate_model_metrics(
        eq.EquineOutput(
            classes=probabilities,
            ood_scores=ood_scores,
            embeddings=torch.zeros(len(X), 1),
        ),
        Y.long(),
    )
    assert metrics["brierScore"] >= 0.0


@aoti_test
@given(random_dataset=random_dataset(), batch=st.integers(min_value=2, max_value=16))
@settings(deadline=None, max_examples=1)
def test_aoti_compile_does_not_mutate_model_state(random_dataset, batch) -> None:
    # Inductor traces forward several times while autotuning; the _Laplace
    # covariance/seen_data buffers must come out unchanged.
    model, _, X = train_equine_gp(random_dataset)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    recompute_before = model.model.recompute_covariance

    with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        aoti_package(torch.export.export(model, (X[:batch],)), directory, "nomutate")

    after = model.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"AOTI compile mutated buffer {key}"
    assert model.model.recompute_covariance == recompute_before
