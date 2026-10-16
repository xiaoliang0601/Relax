# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang ServerArgs field discovery across dataclass and msgspec releases."""

import ast
import dataclasses
from pathlib import Path


def load_field_discovery():
    path = Path(__file__).resolve().parents[3] / "relax/backends/sglang/sglang_engine.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_server_arg_field_names"
    )
    namespace = {"dataclasses": dataclasses}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_server_arg_field_names_support_dataclass_and_struct():
    namespace = load_field_discovery()

    @dataclasses.dataclass
    class LegacyArgs:
        model_path: str
        tp_size: int

    class StructArgs:
        __struct_fields__ = ("model_path", "tp_size")

    for args_type in (LegacyArgs, StructArgs):
        namespace["ServerArgs"] = args_type
        assert namespace["_server_arg_field_names"]() == ("model_path", "tp_size")
