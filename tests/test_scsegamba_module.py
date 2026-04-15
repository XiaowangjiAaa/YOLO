import torch

from ultralytics.nn.modules import C2fSAVSS, SAVSS2D, SAVSSBlock


def test_savss2d_fast_shape():
    m = SAVSS2D(32, scan_impl="fast")
    x = torch.randn(2, 32, 16, 16)
    y = m(x)
    assert y.shape == x.shape


def test_savss2d_dynamic_ssm_shape():
    m = SAVSS2D(16, scan_impl="ssm", ssm_mode="dynamic")
    x = torch.randn(1, 16, 8, 8)
    y = m(x)
    assert y.shape == x.shape


def test_savss_block_shape():
    m = SAVSSBlock(32, scan_impl="fast")
    x = torch.randn(1, 32, 12, 12)
    y = m(x)
    assert y.shape == x.shape


def test_c2fsavss_shape():
    m = C2fSAVSS(c1=32, c2=64, n=2, e=0.5, scan_impl="fast")
    x = torch.randn(1, 32, 20, 20)
    y = m(x)
    assert y.shape == (1, 64, 20, 20)
