import torch

from ultralytics.nn.modules import C2fSAVSS, SAVSS2D, SAVSSBlock


def test_savss2d_shape():
    m = SAVSS2D(64)
    x = torch.randn(2, 64, 16, 16)
    y = m(x)
    assert y.shape == x.shape


def test_savss_block_shape_and_residual_path():
    m = SAVSSBlock(64)
    x = torch.randn(2, 64, 16, 16)
    y = m(x)
    assert y.shape == x.shape


def test_c2fsavss_shape():
    m = C2fSAVSS(c1=64, c2=128, n=2, e=0.5)
    x = torch.randn(1, 64, 20, 20)
    y = m(x)
    assert y.shape == (1, 128, 20, 20)
