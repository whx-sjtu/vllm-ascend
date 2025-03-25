from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Callable, Union

import torch_npu

from vllm.logger import init_logger

logger = init_logger(__name__)


_VLLM_ASCEND_OPS = {
    "silu_and_mul": torch_npu.npu_swiglu,
    "rms_norm": torch_npu.npu_rms_norm,
    "add_rms_norm": torch_npu.npu_add_rms_norm,
    "rotary_embedding": torch_npu._npu_rotary_embedding,
    "paged_attention": torch_npu._npu_paged_attention,
    "reshape_and_cache": torch_npu._npu_reshape_and_cache,
    "flash_attention": torch_npu._npu_flash_attention,
}


@dataclass
class _OpRegistry:
    ops: Dict[str, Callable] = field(default_factory=dict)

    @lru_cache
    def register_op(self, op_name: str, op_func: Callable) -> None:
        if op_name in self.ops:
            logger.warning(
                "Op %s is already registered, and will be "
                "overwriting by new Op func %s",
                op_name,
                op_func.__name__,
            )
        self.ops[op_name] = op_func

    def register(self, op_name: Callable):
        def wrapper(op_func: Callable) -> None:
            self.register_op(op_name, op_func)
            return op_func

        return wrapper

    def get_op(self, op_name: str) -> Union[Callable, None]:
        if op_name not in self.ops:
            raise ValueError(
                f"Op name {op_name} has not been registered. Registered op list: {self.ops.keys()}"
            )
        return self.ops[op_name]


OpRegistry = _OpRegistry(_VLLM_ASCEND_OPS)