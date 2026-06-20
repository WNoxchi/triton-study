import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# PARAMETERS
# X pointer to 1st input vector
# Y pointer to 2nd input vector
# Out pointer to output vector
# n_elements - vector size
# BLOCK_SIZE: tl.constexpr - number of elements each program should process.

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # setup
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # compute
    out = x + y

    # store
    tl.store(out_ptr + offsets, out, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE and out.device == DEVICE

    n_elements = out.numel()

    # number of parallel kernel instances; Tuple[int] or Callable(metaparameters) -> Tuple[int]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)

    return out

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)



