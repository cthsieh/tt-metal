import torch
import tt_lib as ttl
from tests.python_api_testing.models.utility_functions_new import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 3, 6, 4]),
        torch.Size([2, 35, 9, 6]),
        torch.Size([1, 2, 64, 32]),
        torch.Size([5, 10, 23, 32]),
    ],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("on_device", [True, False])
def test_softmax_fallback(input_shape, dim, on_device):
    torch.manual_seed(1234)

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.randn(input_shape).bfloat16().float()
    pt_out = torch.nn.functional.softmax(x, dim)

    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.fallback_ops.softmax(t0, dim)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass

    del t1

    ttl.device.CloseDevice(device)
