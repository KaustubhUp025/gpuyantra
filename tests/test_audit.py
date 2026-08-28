"""The whole-model audit (spec 7 / Task 10).

No GPU and no network here. Every test builds its own `nn.Module` tree, because that is
the only way to assert on classification and on FLOP/byte arithmetic without making the
suite depend on a 3 GB checkpoint download — and the arithmetic is what can silently rot.

The two things worth guarding are:

1. **`estimate_flops_and_bytes` matching its documented rules.** The numbers below are
   computed by hand from the rules in that function's docstring, not copied from its
   output, so an accidental change to a coefficient fails here instead of quietly moving
   a module across the ridge point and out of the recommendation.

2. **Nothing unmeasured being reported as measured.** On CPU there is no `do_bench`, so
   `bandwidth_utilization_pct` must stay 0.0 and `measured` must stay False; the report
   text has to say "ESTIMATED" in as many words (red line #3).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kernelsmith.config import AUDIT_PROBE_BATCH, AUDIT_PROBE_SEQ, AUDIT_REPORT_WIDTH
from kernelsmith.tools.profiler_tool import (
    RIDGE_POINT,
    AuditEntry,
    AuditReport,
    assign_priority,
    audit_model_for_agent,
    build_recommendation,
    classify_op_family_from_module,
    estimate_flops_and_bytes,
    format_audit_report,
    representative_input_shape,
    resolve_model_id,
)
from kernelsmith.tools.profiler_tool import _audit_entries as audit_entries

HIDDEN = 64
FP32 = 4
FP16 = 2


def tiny_model() -> nn.Sequential:
    """The spec's example tree: one of each of the three families that matter."""
    return nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, 32), nn.ReLU())


def build_report(model: nn.Module, hidden: int = HIDDEN) -> AuditReport:
    entries = audit_entries(model, hidden, on_cuda=False, dtype=torch.float32)
    return AuditReport(
        model_name="tiny",
        total_modules=sum(entry.count for entry in entries),
        unique_types=len(entries),
        module_entries=entries,
        top_target=entries[0].module_type if entries else "",
        recommendation=build_recommendation(entries, "tiny"),
        device="cpu",
        hidden_size=hidden,
    )


# --------------------------------------------------------------------------- #
# audit over a tiny model
# --------------------------------------------------------------------------- #


def test_audit_finds_every_unique_module_type_in_a_tiny_model():
    report = build_report(tiny_model())

    assert report.unique_types == 3
    assert report.total_modules == 3
    by_type = {entry.module_type: entry for entry in report.module_entries}
    assert set(by_type) == {"LayerNorm", "Linear", "ReLU"}
    assert by_type["LayerNorm"].op_family == "norm"
    assert by_type["Linear"].op_family == "linear"
    assert by_type["ReLU"].op_family == "activation"


def test_the_sequential_container_is_not_reported_as_a_target():
    """`nn.Sequential` does no arithmetic; an inventory of TARGETS must not list it."""
    report = build_report(tiny_model())
    assert "Sequential" not in {entry.module_type for entry in report.module_entries}


def test_instance_counts_are_per_class_not_per_module():
    model = nn.Sequential(nn.LayerNorm(HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU())
    report = build_report(model)
    counts = {entry.module_type: entry.count for entry in report.module_entries}
    assert counts == {"LayerNorm": 2, "ReLU": 1}


def test_top_target_is_the_highest_priority_entry():
    report = build_report(tiny_model())

    assert report.top_target == report.module_entries[0].module_type
    assert report.module_entries[0].priority == "HIGH"
    # HIGH before MEDIUM before LOW, always.
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    ranks = [order[entry.priority] for entry in report.module_entries]
    assert ranks == sorted(ranks)


def test_a_norm_outranks_an_equally_memory_bound_activation_by_instance_count():
    """Within a priority band the tie-break is how many instances there are."""
    model = nn.Sequential(nn.LayerNorm(HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU())
    report = build_report(model)
    assert report.top_target == "LayerNorm"


def test_param_shapes_come_from_a_representative_instance():
    report = build_report(tiny_model())
    by_type = {entry.module_type: entry for entry in report.module_entries}
    assert by_type["LayerNorm"].param_shapes == {"weight": [HIDDEN], "bias": [HIDDEN]}
    assert by_type["Linear"].param_shapes == {"weight": [32, HIDDEN], "bias": [32]}
    assert by_type["ReLU"].param_shapes == {}


def test_cpu_audit_measures_nothing_and_says_so():
    """The one thing an honest audit must never do is imply it measured something."""
    report = build_report(tiny_model())

    assert report.measured is False
    assert all(entry.bandwidth_utilization_pct == 0.0 for entry in report.module_entries)
    assert all(entry.measured is False for entry in report.module_entries)

    text = format_audit_report(report)
    assert "ESTIMATED" in text
    assert "n/a" in text  # the blank bandwidth column


def test_an_empty_model_is_an_empty_report_not_a_crash():
    report = build_report(nn.Sequential())
    assert report.module_entries == []
    assert report.top_target == ""
    assert "No profilable modules" in report.recommendation
    assert "(no profilable modules)" in format_audit_report(report)


# --------------------------------------------------------------------------- #
# classify_op_family_from_module
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (nn.LayerNorm(HIDDEN), "norm"),
        (nn.GroupNorm(2, HIDDEN), "norm"),
        (nn.BatchNorm1d(HIDDEN), "norm"),
        (nn.BatchNorm2d(HIDDEN), "norm"),
        (nn.BatchNorm3d(HIDDEN), "norm"),
        (nn.RMSNorm(HIDDEN), "norm"),
        (nn.Linear(HIDDEN, 32), "linear"),
        (nn.Conv1d(3, 8, 3), "conv"),
        (nn.Conv2d(3, 8, 3), "conv"),
        (nn.Conv3d(3, 8, 3), "conv"),
        (nn.Embedding(10, HIDDEN), "embedding"),
        (nn.SiLU(), "activation"),
        (nn.GELU(), "activation"),
        (nn.ReLU(), "activation"),
        (nn.Dropout(0.1), "dropout"),
        (nn.Identity(), "other"),
    ],
)
def test_classify_op_family_from_module(module, expected):
    assert classify_op_family_from_module(module) == expected


def test_a_subclass_of_layernorm_is_still_a_norm():
    """isinstance before name matching, so a wrapped norm is not filed as "other"."""

    class FusedLayerNorm(nn.LayerNorm):
        pass

    assert classify_op_family_from_module(FusedLayerNorm(HIDDEN)) == "norm"


def test_a_norm_that_subclasses_nothing_in_torch_is_recognized_by_name():
    """Qwen2RMSNorm subclasses plain nn.Module; the name is all there is to go on."""

    class Qwen2RMSNorm(nn.Module):
        pass

    assert classify_op_family_from_module(Qwen2RMSNorm()) == "norm"


def test_transformers_conv1d_is_a_linear_not_a_convolution():
    """GPT-2's projections are `transformers.pytorch_utils.Conv1D` — a transposed Linear.

    Calling it a convolution left every one of GPT-2's 48 matmuls unestimated, which
    reported GPT2Block as memory-bound when its arithmetic is entirely in them.
    """

    class Conv1D(nn.Module):
        def __init__(self, nf: int, nx: int):
            super().__init__()
            self.nf = nf
            self.weight = nn.Parameter(torch.empty(nx, nf))

    module = Conv1D(3 * HIDDEN, HIDDEN)
    assert classify_op_family_from_module(module) == "linear"
    flops, byte_count = estimate_flops_and_bytes(module, (1, 8, HIDDEN), FP16)
    assert flops > 0 and byte_count > 0


def test_a_rotary_embedding_is_not_classified_as_an_embedding_lookup():
    class Qwen2RotaryEmbedding(nn.Module):
        pass

    assert classify_op_family_from_module(Qwen2RotaryEmbedding()) == "other"


# --------------------------------------------------------------------------- #
# estimate_flops_and_bytes — the documented rules, computed by hand
# --------------------------------------------------------------------------- #


def test_rmsnorm_is_five_flops_per_element_over_three_tensors():
    module = nn.RMSNorm(HIDDEN)
    shape = (1, 8, HIDDEN)
    numel = 8 * HIDDEN

    flops, byte_count = estimate_flops_and_bytes(module, shape, FP16)
    assert flops == 5 * numel
    assert byte_count == 3 * numel * FP16


def test_layernorm_is_seven_flops_per_element_over_four_tensors():
    """Two more flops and one more tensor than RMSNorm — the extra tensor is `bias`."""
    module = nn.LayerNorm(HIDDEN)
    shape = (1, 8, HIDDEN)
    numel = 8 * HIDDEN

    flops, byte_count = estimate_flops_and_bytes(module, shape, FP16)
    assert flops == 7 * numel
    assert byte_count == 4 * numel * FP16


def test_a_layernorm_without_affine_params_is_counted_as_an_rmsnorm():
    """No bias means one fewer tensor to read, whatever the class is called."""
    module = nn.LayerNorm(HIDDEN, elementwise_affine=False)
    numel = 8 * HIDDEN
    flops, byte_count = estimate_flops_and_bytes(module, (1, 8, HIDDEN), FP16)
    assert (flops, byte_count) == (5 * numel, 3 * numel * FP16)


def test_batchnorm_is_seven_flops_per_element_over_four_tensors():
    module = nn.BatchNorm2d(16)
    shape = (2, 16, 8, 8)
    numel = 2 * 16 * 8 * 8

    flops, byte_count = estimate_flops_and_bytes(module, shape, FP16)
    assert flops == 7 * numel
    assert byte_count == 4 * numel * FP16


def test_linear_is_two_flops_per_mac_and_counts_the_weight_matrix_once():
    module = nn.Linear(HIDDEN, 32)
    rows = 1 * 8

    flops, byte_count = estimate_flops_and_bytes(module, (1, 8, HIDDEN), FP16)
    assert flops == 2 * rows * HIDDEN * 32
    assert byte_count == (rows * HIDDEN + HIDDEN * 32 + rows * 32) * FP16


def test_conv2d_counts_the_output_volume_and_the_convolution_geometry():
    module = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    shape = (1, 3, 16, 16)
    spatial_out = 16  # stride 1, padding 1, kernel 3 -> same size

    flops, byte_count = estimate_flops_and_bytes(module, shape, FP16)
    assert flops == 2 * 1 * 8 * spatial_out * spatial_out * 3 * 3 * 3
    expected_elements = (1 * 3 * 16 * 16) + (8 * 3 * 3 * 3) + (1 * 8 * spatial_out * spatial_out)
    assert byte_count == expected_elements * FP16


def test_conv2d_stride_two_halves_the_output_side():
    module = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
    flops, _ = estimate_flops_and_bytes(module, (1, 3, 16, 16), FP16)
    assert flops == 2 * 1 * 8 * 8 * 8 * 3 * 3 * 3


def test_a_grouped_conv_divides_its_macs_by_the_group_count():
    dense = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=1)
    depthwise = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
    shape = (1, 8, 16, 16)

    assert estimate_flops_and_bytes(dense, shape, FP16)[0] == (
        8 * estimate_flops_and_bytes(depthwise, shape, FP16)[0]
    )


def test_relu_is_one_flop_per_element_and_silu_is_three():
    shape = (1, 8, HIDDEN)
    numel = 8 * HIDDEN

    assert estimate_flops_and_bytes(nn.ReLU(), shape, FP16) == (numel, 2 * numel * FP16)
    assert estimate_flops_and_bytes(nn.SiLU(), shape, FP16) == (3 * numel, 2 * numel * FP16)


def test_embedding_and_dropout_are_skipped():
    """A gather and an eval-time identity: nothing to fuse, so nothing is claimed."""
    assert estimate_flops_and_bytes(nn.Embedding(10, HIDDEN), (1, 8), FP16) == (0, 0)
    assert estimate_flops_and_bytes(nn.Dropout(0.1), (1, 8, HIDDEN), FP16) == (0, 0)


def test_element_size_defaults_to_the_modules_own_dtype():
    """fp32 weights mean 4 bytes per element, so twice the traffic of fp16."""
    module = nn.LayerNorm(HIDDEN)  # fp32 by default
    _, fp32_bytes = estimate_flops_and_bytes(module, (1, 8, HIDDEN))
    _, explicit = estimate_flops_and_bytes(module, (1, 8, HIDDEN), FP32)
    assert fp32_bytes == explicit == 4 * 8 * HIDDEN * FP32


def test_a_composite_module_is_the_sum_of_its_children():
    """This is what gives a Qwen2MLP a real intensity instead of a zero."""
    block = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU())
    shape = (1, 8, HIDDEN)

    total = estimate_flops_and_bytes(block, shape, FP16)
    parts = [estimate_flops_and_bytes(child, shape, FP16) for child in block]
    # ReLU's own probe shape comes from the parent; Linear's from in_features.
    assert total[0] == sum(flops for flops, _ in parts)
    assert total[1] > 0


def test_an_mlp_shaped_block_lands_on_the_compute_side_of_the_ridge():
    """The direction of the answer is the point, and it must not depend on fine detail."""
    block = nn.Sequential(nn.Linear(1536, 8960), nn.SiLU(), nn.Linear(8960, 1536))
    flops, byte_count = estimate_flops_and_bytes(block, (1, 512, 1536), FP16)
    assert flops / byte_count > RIDGE_POINT


def test_a_norm_lands_far_on_the_memory_side_of_the_ridge():
    flops, byte_count = estimate_flops_and_bytes(nn.LayerNorm(1536), (1, 512, 1536), FP16)
    assert flops / byte_count < RIDGE_POINT / 10


def test_an_unrecognized_leaf_is_zero_not_a_guess():
    assert estimate_flops_and_bytes(nn.Identity(), (1, 8, HIDDEN), FP16) == (0, 0)


# --------------------------------------------------------------------------- #
# representative_input_shape
# --------------------------------------------------------------------------- #


def test_probe_shapes_come_from_the_modules_own_declared_dimensions():
    batch, seq = AUDIT_PROBE_BATCH, AUDIT_PROBE_SEQ
    assert representative_input_shape(nn.Linear(128, 32), HIDDEN) == (batch, seq, 128)
    assert representative_input_shape(nn.LayerNorm(128), HIDDEN) == (batch, seq, 128)
    assert representative_input_shape(nn.Embedding(10, 128), HIDDEN) == (batch, seq)
    assert representative_input_shape(nn.Conv2d(3, 8, 3), HIDDEN)[:2] == (batch, 3)
    assert representative_input_shape(nn.BatchNorm2d(16), HIDDEN)[:2] == (batch, 16)


def test_a_shapeless_module_falls_back_to_the_models_hidden_size():
    assert representative_input_shape(nn.ReLU(), 1536) == (
        AUDIT_PROBE_BATCH,
        AUDIT_PROBE_SEQ,
        1536,
    )


# --------------------------------------------------------------------------- #
# priority
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("family", "memory_bound", "expected"),
    [
        ("norm", True, "HIGH"),
        ("activation", True, "HIGH"),
        ("linear", True, "MEDIUM"),
        ("conv", True, "MEDIUM"),
        ("other", True, "MEDIUM"),
        ("norm", False, "LOW"),
        ("linear", False, "LOW"),
    ],
)
def test_assign_priority(family, memory_bound, expected):
    assert assign_priority(family, memory_bound) == expected


def test_a_skipped_family_is_never_recommended_however_bandwidth_bound_it_looks():
    assert assign_priority("embedding", True) == "LOW"
    assert assign_priority("dropout", True) == "LOW"


def test_a_module_with_no_estimate_is_low_because_zero_is_not_a_bottleneck():
    assert assign_priority("norm", True, has_estimate=False) == "LOW"


# --------------------------------------------------------------------------- #
# format_audit_report
# --------------------------------------------------------------------------- #


def test_the_table_is_exactly_the_configured_width():
    text = format_audit_report(build_report(tiny_model()))
    box_lines = [line for line in text.splitlines() if line[:1] in "┌├└│"]
    assert box_lines, "the report drew no table"
    assert {len(line) for line in box_lines} == {AUDIT_REPORT_WIDTH}


def test_the_table_has_a_row_per_entry_and_the_expected_columns():
    report = build_report(tiny_model())
    text = format_audit_report(report)

    for heading in ("Module Type", "Count", "Regime", "AI (F/B)", "BW %", "Priority"):
        assert heading in text
    for entry in report.module_entries:
        assert entry.module_type in text
    assert report.recommendation in text
    assert f"Top target: {report.top_target}" in text


def test_a_very_long_class_name_is_truncated_rather_than_breaking_alignment():
    entry = AuditEntry(
        module_type="A" * 200,
        count=1,
        op_family="norm",
        bottleneck="memory",
        arithmetic_intensity=0.5,
        bandwidth_utilization_pct=0.0,
        priority="HIGH",
        param_shapes={},
    )
    report = AuditReport("m", 1, 1, [entry], entry.module_type, "r")
    box_lines = [line for line in format_audit_report(report).splitlines() if line[:1] in "┌├└│"]
    assert {len(line) for line in box_lines} == {AUDIT_REPORT_WIDTH}


def test_an_unestimated_entry_prints_no_regime_and_no_numbers():
    """ "memory" is the conservative default in the dataclass; the table must not assert it."""
    entry = AuditEntry(
        module_type="Dropout",
        count=3,
        op_family="dropout",
        bottleneck="memory",
        arithmetic_intensity=0.0,
        bandwidth_utilization_pct=0.0,
        priority="LOW",
        param_shapes={},
    )
    row = next(
        line
        for line in format_audit_report(
            AuditReport("m", 3, 1, [entry], "Dropout", "r")
        ).splitlines()
        if "Dropout" in line
    )
    assert "memory" not in row
    assert row.count("n/a") == 2  # AI and BW


def test_a_measurement_taken_off_an_l4_carries_a_comparability_warning():
    """BW % is a fraction of the L4's 300 GB/s; measured elsewhere it is not comparable."""
    entry = AuditEntry(
        module_type="LayerNorm",
        count=25,
        op_family="norm",
        bottleneck="memory",
        arithmetic_intensity=0.88,
        bandwidth_utilization_pct=39.0,
        priority="HIGH",
        param_shapes={},
        measured=True,
    )
    report = AuditReport(
        "gpt2", 25, 1, [entry], "LayerNorm", "r", device="cuda", measured=True, gpu_name="RTX A500"
    )
    text = format_audit_report(report)
    assert "MEASURED" in text
    assert "RTX A500" in text
    assert "not comparable" in text

    on_l4 = format_audit_report(
        AuditReport(
            "gpt2",
            25,
            1,
            [entry],
            "LayerNorm",
            "r",
            device="cuda",
            measured=True,
            gpu_name="NVIDIA L4",
        )
    )
    assert "not comparable" not in on_l4


def test_the_recommendation_names_the_target_and_the_reason():
    report = build_report(tiny_model())
    assert report.top_target in report.recommendation
    assert "FLOP/byte" in report.recommendation


def test_a_model_with_nothing_memory_bound_says_so_instead_of_recommending_fusion():
    entry = AuditEntry(
        module_type="Linear",
        count=196,
        op_family="linear",
        bottleneck="compute",
        arithmetic_intensity=154.0,
        bandwidth_utilization_pct=0.0,
        priority="LOW",
        param_shapes={},
    )
    text = build_recommendation([entry], "some-model")
    assert "Nothing" in text
    assert "tile and tensor-core work" in text


# --------------------------------------------------------------------------- #
# resolve_model_id and the ADK tool surface
# --------------------------------------------------------------------------- #


def test_registry_keys_resolve_to_huggingface_ids_and_anything_else_passes_through():
    assert resolve_model_id("gpt2") == "openai-community/gpt2"
    assert resolve_model_id("GPT2") == "openai-community/gpt2"
    assert resolve_model_id("qwen2.5-1.5b") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert resolve_model_id("some/other-model") == "some/other-model"


def test_the_tool_reports_an_unloadable_model_as_an_error_not_an_empty_audit():
    """An empty audit reads like "this model has nothing to optimize". It must not."""
    payload = audit_model_for_agent("definitely/not-a-real-model-id", device="cpu")

    assert payload["error"]
    assert payload["module_entries"] == []
    assert payload["top_target"] == ""


def test_the_tool_is_published_under_the_name_the_prompt_uses():
    from kernelsmith.tools.profiler_tool import audit_tool

    assert audit_tool.name == "audit_model"


# --------------------------------------------------------------------------- #
# The CLI (Task 10, Part D)
# --------------------------------------------------------------------------- #
#
# `make demo` passes only `$(DEMO_ARGS)` — no subcommand — so the argv shim is the one
# piece of this that can break something that already worked. A `make demo` that started
# failing on argument parsing would look like a broken demo, not a broken parser.


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], "optimize"),
        (["--no-server"], "optimize"),
        (["--op", "rmsnorm"], "optimize"),
        (["--hidden-size", "768"], "optimize"),
        (["optimize"], "optimize"),
        (["optimize", "--no-server"], "optimize"),
        (["audit"], "audit"),
        (["audit", "--all"], "audit"),
        (["full", "--no-server"], "full"),
    ],
)
def test_argv_without_a_subcommand_still_means_optimize(argv, expected):
    from kernelsmith.run_demo import build_parser, normalize_argv

    args = build_parser().parse_args(normalize_argv(argv))
    assert args.command == expected


def test_the_pre_subcommand_demo_invocation_parses_identically():
    """`python -m kernelsmith.run_demo --no-server` is what the Makefile runs today."""
    from kernelsmith.run_demo import build_parser, normalize_argv

    args = build_parser().parse_args(normalize_argv(["--no-server"]))
    assert (args.command, args.op, args.hidden_size, args.no_server) == (
        "optimize",
        "rmsnorm",
        1536,
        True,
    )


def test_help_is_not_swallowed_by_the_subcommand_shim():
    from kernelsmith.run_demo import normalize_argv

    assert normalize_argv(["--help"]) == ["--help"]
    assert normalize_argv(["-h"]) == ["-h"]


def test_audit_defaults_to_the_served_model_and_auto_device():
    from kernelsmith.run_demo import DEFAULT_AUDIT_MODEL, build_parser, normalize_argv

    args = build_parser().parse_args(normalize_argv(["audit"]))
    assert args.model == DEFAULT_AUDIT_MODEL
    assert args.device is None  # resolved by default_audit_device() at call time
    assert args.output == "text"
    assert args.all is False


def test_the_device_default_follows_cuda_availability():
    import torch

    from kernelsmith.run_demo import default_audit_device

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert default_audit_device() == expected


def test_the_comparison_table_puts_every_audited_model_on_its_own_row():
    from kernelsmith.config import MODEL_REGISTRY
    from kernelsmith.run_demo import format_comparison_table

    reports = [
        AuditReport(
            model_name=str(entry["hf_id"]),
            total_modules=10,
            unique_types=1,
            module_entries=[
                AuditEntry(
                    module_type=f"{entry['norm_type']}Impl",
                    count=7,
                    op_family="norm",
                    bottleneck="memory",
                    arithmetic_intensity=0.44,
                    bandwidth_utilization_pct=0.0,
                    priority="HIGH",
                    param_shapes={},
                )
            ],
            top_target=f"{entry['norm_type']}Impl",
            recommendation="r",
        )
        for entry in MODEL_REGISTRY.values()
    ]

    text = format_comparison_table(reports)

    for key, entry in MODEL_REGISTRY.items():
        assert key in text
        assert str(entry["norm_type"]) in text
    assert text.count("HIGH") == len(MODEL_REGISTRY)


def test_the_comparison_table_survives_a_model_that_could_not_be_audited():
    """`--all` keeps going past one failure; the table must render what it did get."""
    from kernelsmith.run_demo import format_comparison_table

    empty = AuditReport("openai-community/gpt2", 0, 0, [], "", "nothing found")
    text = format_comparison_table([empty])
    assert "gpt2" in text
    assert "(none)" in text


def test_the_all_models_sweep_defaults_to_cpu_even_where_cuda_is_available():
    """`make audit-all` passes no --device, and three models on CUDA does not finish.

    A single-model audit follows `default_audit_device()`; the sweep does not, because
    CUDA there costs ~3.4 GB of weight downloads and minutes of do_bench to fill a
    bandwidth column the comparison table has no room for — and which is flagged
    non-comparable anywhere but an L4 anyway.
    """
    from unittest.mock import patch

    from kernelsmith.run_demo import run_audit_all

    with patch("kernelsmith.run_demo.run_audit") as audit:
        audit.return_value = AuditReport("m", 0, 0, [], "", "r")
        run_audit_all()

    devices = {call.kwargs["device"] for call in audit.call_args_list}
    assert devices == {"cpu"}


def test_the_sweep_still_honours_an_explicit_device():
    from unittest.mock import patch

    from kernelsmith.run_demo import run_audit_all

    with patch("kernelsmith.run_demo.run_audit") as audit:
        audit.return_value = AuditReport("m", 0, 0, [], "", "r")
        run_audit_all(device="cuda")

    assert {call.kwargs["device"] for call in audit.call_args_list} == {"cuda"}


def test_the_sweep_covers_every_registered_model():
    from unittest.mock import patch

    from kernelsmith.config import MODEL_REGISTRY
    from kernelsmith.run_demo import run_audit_all

    with patch("kernelsmith.run_demo.run_audit") as audit:
        audit.return_value = AuditReport("m", 0, 0, [], "", "r")
        reports = run_audit_all()

    assert [call.args[0] for call in audit.call_args_list] == list(MODEL_REGISTRY)
    assert len(reports) == len(MODEL_REGISTRY)


def test_one_unloadable_model_does_not_abort_the_sweep():
    """`--all` on a box that cannot reach one model must still report the others."""
    from unittest.mock import patch

    from kernelsmith.config import MODEL_REGISTRY
    from kernelsmith.run_demo import run_audit_all

    def flaky(key, **_kwargs):
        if key == list(MODEL_REGISTRY)[0]:
            raise OSError("no network")
        return AuditReport(key, 0, 0, [], "", "r")

    with patch("kernelsmith.run_demo.run_audit", side_effect=flaky):
        reports = run_audit_all()

    assert len(reports) == len(MODEL_REGISTRY) - 1
