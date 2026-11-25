import traceback

import torch

from .utils_data import CompileResult

from .utils_evaluate_common import (
    exec_and_get_output,
    compile_and_get_output,
    format_exception_string
)

def load_original_model_and_inputs(
    model_original_src: str, context: dict, build_dir: str,
    ): #-> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """
    if build_dir:
        context["BUILD_DIRECTORY"] = build_dir
        # Add import at the start of the source code
        model_original_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + model_original_src

    try:
        comp_out, output, potential_exception = compile_and_get_output(model_original_src, "<string>", "exec")
        if potential_exception:
            raise(potential_exception)
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None, "compilation_error", (output, None)

    try:
        exec_out, output2, potential_exception = exec_and_get_output(model_original_src, context)  # expose to current namespace
        if potential_exception:
            raise(potential_exception)
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None, "runtime_error", (output, output2)

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn), "success", (output, output2)

def load_custom_model(
    model_custom_src: str,
    context: dict,
    build_directory: str | None = None,
    device=None,
):
    """Load the generated module and return its ``ModelNew`` class.

    A build/exec failure is raised with a ``kernel_output_added`` payload so
    ``run_compilation`` can classify the stage and keep the diagnostics.
    """
    print("[Compilation] start of load custom model...")
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    comp_out, output, exc = compile_and_get_output(model_custom_src, "<string>", "exec")
    if exc:
        exc.kernel_output_added = (
            None,
            "pre_compilation_error",
            (output, None),
            (comp_out, None),
        )
        raise exc
    print("[Compilation] source compiled")

    exec_out, output2, exc = exec_and_get_output(model_custom_src, context)
    if exc:
        status = (
            "compilation_error"
            if isinstance(exc, RuntimeError) and "Error building extension" in str(exc)
            else "runtime_error"
        )
        exc.kernel_output_added = (
            None,
            status,
            (output, output2),
            (comp_out, exec_out),
        )
        raise exc
    print("[Exec] generated code executed")

    model_new = context.get("ModelNew")
    if model_new is None:
        exc = RuntimeError("ModelNew was not defined by the generated code")
        exc.kernel_output_added = (
            None,
            "model_new_not_found_error",
            (output, output2),
            (comp_out, exec_out),
        )
        raise exc
    return model_new, "success", (output, output2), (comp_out, exec_out)


def run_compilation(device, custom_model_src, context, build_dir_new):
    """Compile one generated kernel and classify the outcome as a ``CompileResult``."""
    metadata = {
        "hardware": torch.cuda.get_device_name(device=device),
        "device": str(device),
        "execution_status": None,
    }

    if not custom_model_src:
        message = (
            "[Formatting Error] Code was empty. Make sure to produce valid code, "
            "ideally wrap it inside '```python' and '```'."
        )
        return CompileResult(
            format_passed=False,
            metadata=metadata,
            main_output=message,
            main_output_runtime=message,
            main_traceback=message,
        )

    try:
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_USE_CUDA_DSA'] = '1'\n"
        ) + custom_model_src
        _, compilation_status, (std_compile, std_runtime), _ = load_custom_model(
            custom_model_src,
            context,
            build_dir_new,
            device,
        )
        metadata["compilation_status"] = compilation_status
    except (Exception, SystemExit) as exc:
        payload = getattr(exc, "kernel_output_added", None)
        if payload:
            _, compilation_status, (std_compile, std_runtime), _ = payload
        else:
            compilation_status = "compilation_error"
            std_compile = getattr(exc, "output", "") or getattr(exc, "stderr", "") or repr(exc)
            std_runtime = ""

        # A lock-file error usually means another compile touched the same
        # build tree; the caller records it and can retry.
        if "lock" in str(exc) or "No such file or directory" in str(exc):
            raise

        trace = traceback.format_exc()
        if compilation_status == "pre_compilation_error":
            metadata.update(
                compilation_error=exc,
                compilation_output=std_compile,
                compilation_traceback=trace,
            )
            return CompileResult(
                pre_compiled=False,
                compiled=False,
                runtime_success=False,
                metadata=metadata,
                main_output=std_compile,
                main_output_runtime=std_runtime,
                main_traceback=trace,
            )
        if compilation_status == "compilation_error":
            metadata.update(
                compilation_error=exc,
                compilation_output=std_compile,
                compilation_traceback=trace,
            )
            return CompileResult(
                pre_compiled=True,
                compiled=False,
                runtime_success=False,
                metadata=metadata,
                main_output=std_compile,
                main_output_runtime=std_runtime,
                main_traceback=trace,
            )
        if compilation_status == "runtime_error":
            metadata.update(
                runtime_error=exc,
                runtime_output=std_compile,
                compilation_status=compilation_status,
                runtime_traceback=trace,
            )
            return CompileResult(
                pre_compiled=True,
                compiled=True,
                runtime_success=False,
                metadata=metadata,
                main_output=std_compile,
                main_output_runtime=std_runtime,
                main_error=format_exception_string(exc),
                main_traceback=trace,
            )
        if compilation_status == "model_new_not_found_error":
            metadata.update(
                runtime_error=exc,
                runtime_output=std_runtime,
                compilation_status=compilation_status,
                runtime_traceback=trace,
            )
            return CompileResult(
                pre_compiled=True,
                compiled=True,
                runtime_success=True,
                metadata=metadata,
                main_output=std_compile,
                main_output_runtime=std_runtime,
                main_error=format_exception_string(exc),
                main_traceback=trace,
            )
        raise

    return CompileResult(
        pre_compiled=True,
        compiled=True,
        runtime_success=True,
        model_new_available=True,
        metadata=metadata,
        main_output=std_compile,
        main_output_runtime=std_runtime,
    )
