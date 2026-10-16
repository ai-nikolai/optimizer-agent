"""Benchmarking CUDA kernels.

Two ways to time a kernel:

* **Event based** (:func:`time_event`) — warm up, then time each iteration with
  ``torch.cuda.Event``s. Simple and robust.
* **CUDA-graph based** (:func:`time_cuda_graph`) — capture the forward pass into
  a CUDA graph *before* benchmarking, then replay it repeatedly on fixed input
  and time the replay. CUDA graphs:

  - remove host-side launch / dispatch overhead so the measurement reflects the
    kernel itself rather than the Python/CUDA launch loop,
  - let us **fail fast** if a graph is illegal/uncapturable, which surfaces
    otherwise-silent CUDA errors before we spend time benchmarking,
  - produce very stable timings (no per-iteration Python variance).

If CUDA graphs cannot be captured (e.g. the kernel uses an operation that a graph
does not permit, or the driver is too old) we transparently fall back to event
timing. Callers decide whether a fallback is acceptable.

All timing helpers return a :class:`TimingStats` dataclass so the rest of the
agent never touches ``torch`` directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# torch / numpy are imported lazily inside the helpers below so that modules
# which merely want the dataclass / API do not force a CUDA import.


@dataclass
class TimingStats:
    """Timing results plus optional device metadata."""

    mean: float
    std: float
    min: float
    max: float
    num_trials: int
    hardware: Optional[str] = None
    device: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_device(device):
    import torch

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.cuda.current_device()
    return device


def _warmup(model, inputs, warmup: int, device):
    import torch

    model.to(device)
    model.eval()
    for _ in range(warmup):
        with torch.no_grad():
            model(*inputs)
        torch.cuda.synchronize(device=device)


def time_event(
    model,
    inputs,
    num_trials=3,
    num_warmup=3,
    *,
    device=None,
    extra: Optional[Dict[str, Any]] = None,
) -> TimingStats:
    """Time a model with CUDA events, after a warm-up.

    Args:
        model: a callable model (e.g. an ``nn.Module``) taking ``*inputs``.
        inputs: the (already device-placed) input tensors.
        num_trials / num_warmup: number of timing / warm-up iterations.
        device: CUDA device; defaults to the current device.
        extra: optional dict merged into the returned stats.
    """
    import numpy as np
    import torch

    device = _get_device(device)
    _warmup(model, inputs, num_warmup, device)

    timings = []
    for _ in range(num_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(*inputs)
        end.record()
        torch.cuda.synchronize(device=device)
        timings.append(start.elapsed_time(end))

    return TimingStats(
        mean=float(np.mean(timings)),
        std=float(np.std(timings)),
        min=float(np.min(timings)),
        max=float(np.max(timings)),
        num_trials=len(timings),
        hardware=torch.cuda.get_device_name(device=device),
        device=str(device),
        extra=dict(extra or {}),
    )


def _capture_graph(model, tensors, *, device=None, static_input_shapes=None):
    """Capture ``model(*tensors)`` into a replayable ``CUDAGraph``.

    Returns ``(graph, static_inputs)`` if capture succeeds, or ``None`` if a CUDA
    graph cannot be created for this model (e.g. an illegal/ungraphable op).

    When ``static_input_shapes`` is given the captured inputs are freshly
    allocated buffers of those shapes and the caller is responsible for copying
    concrete data into them before each replay.
    """
    import torch

    try:
        if static_input_shapes is None:
            static_inputs = [t.clone() for t in tensors]
        else:
            static_inputs = [
                torch.empty(*shape, dtype=torch.float32, device=device)
                for shape in static_input_shapes
            ]
            if tensors:
                static_inputs[0].copy_(tensors[0])

        # Prime once (required before capture; also surfaces illegal ops).
        with torch.no_grad():
            model(*static_inputs)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model(*static_inputs)
        return graph, static_inputs
    except Exception:
        return None


def time_cuda_graph(
    model,
    inputs,
    num_trials=10,
    num_warmup=3,
    *,
    device=None,
    max_input_shape: Optional[List[tuple]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> TimingStats:
    """Time a model by capturing and replaying a CUDA graph.

    The graph is captured once from warm-up iterations; timing then replays the
    graph on *fixed* inputs. If capture fails for any reason we fall back to
    :func:`time_event` so an ungraphable kernel is still benchmarkable.

    Args:
        model: callable model taking ``*inputs``.
        inputs: input tensors (already on ``device``).
        num_trials / num_warmup: timing / warm-up iterations.
        device: CUDA device; defaults to the current device.
        max_input_shape: if given, inputs are treated as variable-shaped and a
            static buffer of this shape is captured. When ``None`` the concrete
            input shapes are used (recommended when input shapes are fixed).
        extra: optional dict merged into the returned stats.
    """
    import numpy as np
    import torch

    device = _get_device(device)
    _warmup(model, inputs, num_warmup, device)

    capture_result = _capture_graph(
        model, inputs, device=device, static_input_shapes=max_input_shape
    )
    if capture_result is None:
        # Could not capture -> fall back to event timing.
        return time_event(
            model, inputs, num_trials=num_trials, num_warmup=num_warmup,
            device=device, extra=extra,
        )
    graph, static_inputs = capture_result

    timings = []
    for _ in range(num_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize(device=device)
        timings.append(start.elapsed_time(end))

    return TimingStats(
        mean=float(np.mean(timings)),
        std=float(np.std(timings)),
        min=float(np.min(timings)),
        max=float(np.max(timings)),
        num_trials=len(timings),
        hardware=torch.cuda.get_device_name(device=device),
        device=str(device),
        extra={
            "graph": "cuda_graph",
            "capture_method": "fixed_shape" if max_input_shape is None else "variable_shape",
            **(extra or {}),
        },
    )


def benchmark(
    model,
    inputs,
    num_trials=3,
    num_warmup=3,
    *,
    device=None,
    method: str = "auto",
    max_input_shape: Optional[List[tuple]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> TimingStats:
    """High-level entry point.

    Args:
        method: ``"auto"`` (try CUDA graph, fall back to events), ``"graph"``
            (graph only, raise on failure), or ``"event"`` (events only).
    """
    if method == "event":
        return time_event(
            model, inputs, num_trials=num_trials, num_warmup=num_warmup,
            device=device, extra=extra,
        )
    if method == "graph":
        return time_cuda_graph(
            model, inputs, num_trials=num_trials, num_warmup=num_warmup,
            device=device, max_input_shape=max_input_shape, extra=extra,
        )
    if method == "auto":
        try:
            return time_cuda_graph(
                model, inputs, num_trials=num_trials, num_warmup=num_warmup,
                device=device, max_input_shape=max_input_shape, extra=extra,
            )
        except Exception:
            return time_event(
                model, inputs, num_trials=num_trials, num_warmup=num_warmup,
                device=device, extra=extra,
            )
    raise ValueError(f"Unknown benchmark method: {method!r}")
