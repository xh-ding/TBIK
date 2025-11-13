from .patch_batch_invariant import patch_batch_invariant, enable_batch_invariant_backward_mode
from .patch_tp_invariant import patch_tp_invariant
from .patch_training_inference_align import patch_vllm_rotary_embedding, patch_triton_attn, patch_unified_triton_attn, patch_kernel_unified_attention_2d

__all__ = ["apply_patches"]

def apply_patches():
    from utils import batch_invariant_is_enabled, tp_invariant_is_enabled, compatible_mode_is_enabled
    if not batch_invariant_is_enabled() and not tp_invariant_is_enabled() and not compatible_mode_is_enabled():
        print("No patches applied to vLLM\n")
        return
    if compatible_mode_is_enabled():
        patch_batch_invariant()
        enable_batch_invariant_backward_mode()
        patch_tp_invariant()
        # TODO: patch triton attention
        patch_kernel_unified_attention_2d()
        patch_unified_triton_attn()
        patch_triton_attn()
        # for aligning huggingface
        patch_vllm_rotary_embedding()
        print("Inference Engine(vLLM) and Training Engine(TorchTitan) are aligned, applying both batch invariance and tp invariance patches\n")
    elif not compatible_mode_is_enabled() and batch_invariant_is_enabled() and not tp_invariant_is_enabled():
        patch_batch_invariant()
        print("Applying batch invariance patches to vLLM\n")
    elif not compatible_mode_is_enabled() and batch_invariant_is_enabled() and tp_invariant_is_enabled():
        patch_batch_invariant()
        patch_tp_invariant()
        print("Applying both batch invariance and tp invariance patches to vLLM\n")
    else:
        raise ValueError("Invalid combination of batch invariance, tp invariance, and compatible mode")
    print("Successfully applied patches\n")
    return