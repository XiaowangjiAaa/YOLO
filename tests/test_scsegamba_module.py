import torch

from ultralytics.nn.modules import C2fSAVSS, SASS2D, SAVSSBlock


def test_sass2d_shape():
    m = SASS2D(64)
    x = torch.randn(2, 64, 32, 32)
    y = m(x)
    assert y.shape == x.shape


def test_savss_block_shape_and_residual_path():
    m = SAVSSBlock(64)
    x = torch.randn(2, 64, 32, 32)
    y = m(x)
    assert y.shape == x.shape


def test_c2fsavss_shape():
    m = C2fSAVSS(c1=64, c2=128, n=2, e=0.5)
    x = torch.randn(1, 64, 40, 40)
    y = m(x)
    assert y.shape == (1, 128, 40, 40)
