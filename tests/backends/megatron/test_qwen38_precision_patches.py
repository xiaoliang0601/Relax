# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU integration tests; numerical kernel validation requires the GPU
suite."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_PATCHES = _ROOT / "relax/backends/megatron/megatron_patch"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(monkeypatch):
    module = _load("precision_env", _ROOT / "relax/utils/env.py")
    monkeypatch.setitem(sys.modules, "relax.utils.env", module)
    return module


@pytest.fixture
def router(env, monkeypatch):
    module = _load("precision_router", _PATCHES / "router_sorted_replay_patch.py")
    torch = SimpleNamespace(
        float32=object(),
        is_grad_enabled=Mock(return_value=False),
        topk=Mock(return_value=SimpleNamespace(indices=object())),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return module, torch


def test_router_flag_defaults_off_and_install_does_not_import_megatron(env, router, monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_ROUTER_SORTED_REPLAY", raising=False)
    assert "RELAX_QWEN38_ROUTER_SORTED_REPLAY" in env.known_env_names()
    assert env.Envs.RELAX_QWEN38_ROUTER_SORTED_REPLAY is False
    assert router[0].install() is False
    monkeypatch.setenv("RELAX_QWEN38_ROUTER_SORTED_REPLAY", "true")
    assert env.Envs.RELAX_QWEN38_ROUTER_SORTED_REPLAY is True


def test_router_no_grad_uses_sorted_indices_and_preserves_original_call(router):
    module, torch = router
    original = Mock(return_value=object())
    logits = SimpleNamespace(ndim=2, shape=(7, 512), dtype=torch.float32, is_cuda=True)
    result = module.make_wrapper(original)(logits, 10)
    torch.topk.assert_called_once_with(logits, k=10, dim=1, sorted=True)
    assert result is original.return_value
    assert original.call_args.kwargs["logits"] is logits
    assert original.call_args.kwargs["precomputed_indices"] is torch.topk.return_value.indices


@pytest.mark.parametrize("case", ["grad", "cpu", "dtype", "shape", "topk", "indices", "replay", "fused", "aux"])
def test_router_unsupported_paths_preserve_original_indices(router, case):
    module, torch = router
    logits = SimpleNamespace(ndim=2, shape=(7, 512), dtype=torch.float32, is_cuda=True)
    kwargs = {}
    topk = 10
    if case == "grad":
        torch.is_grad_enabled.return_value = True
    elif case == "cpu":
        logits.is_cuda = False
    elif case == "dtype":
        logits.dtype = object()
    elif case == "shape":
        logits.shape = (7, 64)
    elif case == "topk":
        topk = 2
    else:
        key = {"indices": "precomputed_indices", "replay": "router_replay", "fused": "fused", "aux": "scaling_factor"}[
            case
        ]
        kwargs[key] = object()
    original = Mock()
    module.make_wrapper(original)(logits, topk, **kwargs)
    torch.topk.assert_not_called()
    for key, value in kwargs.items():
        assert original.call_args.kwargs[key] is value
    assert original.call_args.kwargs["precomputed_indices"] is kwargs.get("precomputed_indices")


@pytest.fixture
def gdn(env, monkeypatch, tmp_path):
    module = _load("precision_gdn", _PATCHES / "gdn_output_config_patch.py")
    configs = [
        SimpleNamespace(kwargs={"BK": n, "BV": n}, num_warps=w, num_stages=3) for n, w in [(128, 8), (64, 4), (32, 2)]
    ]
    tuner = type("CachedAutotuner", (), {})()
    tuner.fn = type("JITFunction", (), {})()
    tuner.configs = configs
    tuner.cache = {"stale": configs[0]}
    tuner.best_config = configs[0]
    kernel = type("Heuristics", (), {})()
    kernel.fn = tuner
    path = tmp_path / "chunk_o.py"
    path.write_text("audited test fixture\n")
    dependency = SimpleNamespace(__file__=str(path), chunk_fwd_kernel_o=kernel)
    loader = Mock(return_value=dependency)
    monkeypatch.setattr(module, "importlib", SimpleNamespace(import_module=loader))
    expected = "f174d431ce67139204a0916f638b5af158dd5917e2cb37c0ca03f5ce735466a0"
    monkeypatch.setattr(
        module, "hashlib", SimpleNamespace(sha256=lambda _: SimpleNamespace(hexdigest=lambda: expected))
    )
    monkeypatch.setitem(sys.modules, "relax.utils.logging_utils", SimpleNamespace(get_logger=lambda _: Mock()))
    return module, kernel, tuner, configs, loader


def test_gdn_disabled_does_not_import_fla(gdn, env, monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG", raising=False)
    assert "RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG" in env.known_env_names()
    module, _, tuner, configs, loader = gdn
    assert module.install() is False
    loader.assert_not_called()
    assert tuner.configs is configs


def test_gdn_retains_original_safe_config_clears_cache_and_is_idempotent(gdn, monkeypatch):
    monkeypatch.setenv("RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG", "1")
    module, _, tuner, configs, _ = gdn
    assert module.install() is True
    assert tuner.configs == [configs[1]]
    assert tuner.configs[0] is configs[1]
    assert tuner.cache == {}
    assert not hasattr(tuner, "best_config")
    assert module.install() is True
    tuner.configs[0].num_stages = 2
    with pytest.raises(RuntimeError, match="changed after installation"):
        module.install()


@pytest.mark.parametrize("case", ["source", "decorators", "configs"])
def test_gdn_rejects_unknown_dependency_before_mutation(gdn, monkeypatch, case):
    monkeypatch.setenv("RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG", "1")
    module, _, tuner, configs, _ = gdn
    if case == "source":
        monkeypatch.setattr(
            module, "hashlib", SimpleNamespace(sha256=lambda _: SimpleNamespace(hexdigest=lambda: "changed"))
        )
    elif case == "decorators":
        tuner.fn = object()
    else:
        configs[0].num_stages = 2
    with pytest.raises(RuntimeError):
        module.install()
    assert tuner.configs is configs
    assert tuner.cache == {"stale": configs[0]}


@pytest.fixture
def pool(env, monkeypatch, tmp_path):
    torch = SimpleNamespace(bfloat16=object())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(jit=lambda fn: fn, language=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "triton.language", SimpleNamespace())
    name = "relax.backends.megatron.megatron_patch.qsa_fixed_pool_patch"
    module = _load(name, _PATCHES / "qsa_fixed_pool_patch.py")
    original = lambda *args: None
    indexer = SimpleNamespace(_pool_keys=original)
    package = SimpleNamespace(qsa=SimpleNamespace(QSAIndexer=indexer))
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.experimental_attention_variant", package)
    layout = SimpleNamespace(get=lambda: (0, 4, 8))
    monkeypatch.setitem(
        sys.modules, "relax.backends.megatron.megatron_patch.qsa_select_patch", SimpleNamespace(_layout=layout)
    )
    monkeypatch.setitem(sys.modules, "relax.utils.logging_utils", SimpleNamespace(get_logger=lambda _: Mock()))
    # Pin checks still read a real file; hash mismatch remains a separate test.
    import hashlib
    import inspect

    path = tmp_path / "qsa.py"
    path.write_text("audited test fixture\n")
    monkeypatch.setattr(inspect, "getsourcefile", lambda _: str(path))
    expected = "38e2cb7a6d3d61f3b96a38894605b16884d330944cb097d0e667eb3f131e9137"
    monkeypatch.setattr(hashlib, "sha256", lambda _: SimpleNamespace(hexdigest=lambda: expected))
    return module, torch, indexer, original


def test_pool_disabled_or_missing_selector_does_not_patch(pool, env, monkeypatch):
    module, _, indexer, original = pool
    monkeypatch.delenv("RELAX_QWEN38_QSA_FIXED_POOL", raising=False)
    assert "RELAX_QWEN38_QSA_FIXED_POOL" in env.known_env_names()
    module.install()
    assert indexer._pool_keys is original
    monkeypatch.setenv("RELAX_QWEN38_QSA_FIXED_POOL", "1")
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_SELECTOR", "0")
    with pytest.raises(RuntimeError, match="FUSED_SELECTOR"):
        module.install()
    assert indexer._pool_keys is original


def test_pool_installer_preserves_metadata_contract_and_is_idempotent(pool, monkeypatch):
    module, _, indexer, _ = pool
    monkeypatch.setenv("RELAX_QWEN38_QSA_FIXED_POOL", "1")
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_SELECTOR", "1")
    module.install()
    installed = indexer._pool_keys
    module.install()
    assert indexer._pool_keys is installed
    model = SimpleNamespace(
        compress_ratio=4,
        config=SimpleNamespace(qsa_indexer_loss_coeff=0, context_parallel_size=1),
        parameters=lambda: [],
    )
    raw = SimpleNamespace(shape=(8, 3), device="cuda")
    docs = SimpleNamespace(shape=(8,), device="cuda")
    consumer = Mock(return_value=object())
    monkeypatch.setattr(module, "fixed_pool", consumer)
    assert installed(model, raw, docs, docs, 4) is consumer.return_value
    consumer.assert_called_once_with(raw, (0, 4, 8), 4)
    for field, bad, good in [("qsa_indexer_loss_coeff", 0.1, 0), ("context_parallel_size", 2, 1)]:
        setattr(model.config, field, bad)
        with pytest.raises(RuntimeError, match="frozen packed"):
            installed(model, raw, docs, docs, 4)
        setattr(model.config, field, good)
    model.parameters = lambda: [SimpleNamespace(requires_grad=True)]
    with pytest.raises(RuntimeError, match="frozen packed"):
        installed(model, raw, docs, docs, 4)
    assert consumer.call_count == 1


def test_pool_rejects_changed_source_before_install(pool, monkeypatch):
    import hashlib

    module, _, indexer, original = pool
    monkeypatch.setenv("RELAX_QWEN38_QSA_FIXED_POOL", "1")
    monkeypatch.setenv("RELAX_QWEN38_QSA_FUSED_SELECTOR", "1")
    monkeypatch.setattr(hashlib, "sha256", lambda _: SimpleNamespace(hexdigest=lambda: "changed"))
    with pytest.raises(RuntimeError, match="audited original"):
        module.install()
    assert indexer._pool_keys is original


@pytest.mark.parametrize(
    "offsets,max_len", [(None, 4), ((0,), 4), ((1, 8), 7), ((0, 4, 7), 4), ((0, 5, 3, 8), 5), ((0, 4, 8), 5)]
)
def test_pool_rejects_missing_or_inconsistent_cpu_layout_before_launch(pool, offsets, max_len):
    module, torch, _, _ = pool
    raw = SimpleNamespace(
        ndim=2, dtype=torch.bfloat16, is_cuda=True, shape=(8, 3), stride=lambda: (3, 1), requires_grad=False
    )
    with pytest.raises(ValueError, match="packed QSA offsets"):
        module.fixed_pool(raw, offsets, max_len)


@pytest.mark.parametrize("case", ["valid", "aliases", "signature", "source"])
def test_router_install_checks_dependencies_before_replacing_both_aliases(router, monkeypatch, case):
    import hashlib

    module, _ = router
    monkeypatch.setenv("RELAX_QWEN38_ROUTER_SORTED_REPLAY", "1")

    def original(
        logits,
        topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
        fused=False,
        router_replay=None,
        dense_output=False,
        precomputed_indices=None,
    ):
        return precomputed_indices

    if case == "signature":
        original = lambda logits: logits
    utils = SimpleNamespace(topk_routing_with_score_function=original)
    owner = SimpleNamespace(topk_routing_with_score_function=original if case != "aliases" else object())
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe", SimpleNamespace(moe_utils=utils, router=owner))
    monkeypatch.setitem(sys.modules, "relax.utils.logging_utils", SimpleNamespace(get_logger=lambda _: Mock()))
    expected = "6ece33ddea2596e64d6584a6d40ccbf9322f38a58ee8d04fd474b97d7ca78039"
    digest = "changed" if case == "source" else expected
    monkeypatch.setattr(hashlib, "sha256", lambda _: SimpleNamespace(hexdigest=lambda: digest))
    before = owner.topk_routing_with_score_function
    if case != "valid":
        with pytest.raises(RuntimeError):
            module.install()
        assert utils.topk_routing_with_score_function is original
        assert owner.topk_routing_with_score_function is before
    else:
        assert module.install() is True
        installed = utils.topk_routing_with_score_function
        assert installed is owner.topk_routing_with_score_function
        assert installed is not original
        assert module.install() is True
        assert utils.topk_routing_with_score_function is installed
        owner.topk_routing_with_score_function = original
        with pytest.raises(RuntimeError, match="aliases diverged"):
            module.install()


@pytest.mark.parametrize("enabled", [None, "ROUTER_SORTED_REPLAY", "GDN_OUTPUT_SAFE_CONFIG", "QSA_FIXED_POOL"])
def test_precision_import_gates_load_only_the_requested_patch(env, monkeypatch, enabled):
    flags = {
        "RELAX_QWEN38_ROUTER_SORTED_REPLAY": "router_sorted_replay_patch",
        "RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG": "gdn_output_config_patch",
        "RELAX_QWEN38_QSA_FIXED_POOL": "qsa_fixed_pool_patch",
    }
    for flag in flags:
        monkeypatch.delenv(flag, raising=False)
    if enabled:
        monkeypatch.setenv("RELAX_QWEN38_" + enabled, "1")
    tree = ast.parse((_PATCHES / "__init__.py").read_text())
    gates = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and node.test.attr in flags
    ]
    assert len(gates) == len(flags)
    imports = []
    installs = []

    def import_patch(name, globals=None, locals=None, fromlist=(), level=0):
        assert name in flags.values()
        imports.append(name)
        return SimpleNamespace(install=lambda: installs.append(name))

    code = compile(ast.Module(body=gates, type_ignores=[]), "precision-import-gates", "exec")
    exec(code, {"Envs": env.Envs, "__builtins__": {"__import__": import_patch}})
    expected = [] if enabled is None else [flags["RELAX_QWEN38_" + enabled]]
    assert imports == installs == expected
