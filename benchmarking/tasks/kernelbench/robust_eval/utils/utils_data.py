from pydantic import BaseModel

class CompileResult(BaseModel):
    """
    Single Kernel Compilation
    """
    # cuda_error: bool = False
    format_passed: bool = True #whether there is any code... (or it is empty)
    pre_compiled: bool = False #whether compile() was successful
    compiled: bool = False #whether when exec() failed with 'Error building extension'
    runtime_success: bool = False #whether exec() was successful
    model_new_available: bool = False #whether ModelNew is available.
    metadata: dict = {}
    main_output: str = ""
    main_output_runtime: str = ""
    main_error: str = ""
    main_traceback: str = ""


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    cuda_success: bool = True #whether there is an illegal memory encountered
    compiled: bool = False #whether compile() was successful
    runtime_success: bool = False #whether exec() was successful
    loading_success: bool = False #whether ModelNew(*inits) was successful
    correctness_success: bool = False #whether check_correctness() was successfully executed (not whether the output is correct)
    timing_success: bool = False #whether measuring timing was successfully executed
    correctness: bool = False #whether the output is actually correct.
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance
    runtime_original: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_original_stats: dict = {}  # only recorded if we decide to measure performance
    main_output: str = ""
    main_error: str = ""
    main_traceback: str = ""
