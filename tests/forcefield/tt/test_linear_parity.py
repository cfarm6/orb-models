import pytest
import torch
from torch import nn

from orb_models.forcefield.tt.modules import TTLinear


@pytest.mark.parametrize("batch,in_features,out_features", [(7, 9, 16), (32, 64, 32)])
def test_tt_linear_matches_cpu_sim(tt_sim_device, batch, in_features, out_features):
    torch.manual_seed(0)
    linear = nn.Linear(in_features, out_features, bias=True).to(dtype=torch.float32)
    tt_linear = TTLinear(linear, tt_sim_device.runner)

    x = torch.randn(batch, in_features, dtype=torch.float32)
    cpu_out = linear(x)
    tt_out = tt_linear(x)

    torch.testing.assert_close(tt_out, cpu_out, atol=5e-2, rtol=5e-2)


@pytest.mark.tt_hardware
@pytest.mark.parametrize("batch,in_features,out_features", [(7, 9, 16)])
def test_tt_linear_matches_cpu_hardware(tt_hw_device, batch, in_features, out_features):
    torch.manual_seed(0)
    linear = nn.Linear(in_features, out_features, bias=True).to(dtype=torch.bfloat16)
    tt_linear = TTLinear(linear, tt_hw_device.runner)

    x = torch.randn(batch, in_features, dtype=torch.bfloat16)
    cpu_out = linear(x.float()).to(dtype=torch.bfloat16)
    tt_out = tt_linear(x)

    torch.testing.assert_close(tt_out.float(), cpu_out.float(), atol=1e-1, rtol=1e-1)
