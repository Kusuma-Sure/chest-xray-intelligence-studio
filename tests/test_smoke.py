import numpy as np
import torch
from backend.config import FINDINGS
from backend.data import make_transform
from backend.metrics import calculate_metrics
from backend.models import build_model, forward_logits
from PIL import Image


def test_label_count():
    assert len(FINDINGS) == 13


def test_transform_shapes():
    image=Image.new('RGB',(300,220),'gray')
    assert make_transform('densenet')(image).shape == (3,224,224)
    assert make_transform('capsnet')(image).shape == (1,224,224)


def test_metric_calculation():
    y_true=np.array([[0],[1],[0],[1]])
    y_prob=np.array([[.1],[.8],[.3],[.9]])
    summary, per_finding=calculate_metrics(np.tile(y_true,(1,13)),np.tile(y_prob,(1,13)))
    assert summary['validation_samples']==4
    assert len(per_finding)==13


def test_architecture_output_shapes():
    # CPU smoke test with random initialization; this does not measure model performance.
    x_rgb=torch.zeros(1,3,224,224)
    x_gray=torch.zeros(1,1,224,224)
    for name in ('densenet','vit','gat'):
        model=build_model(name,pretrained=False).eval()
        with torch.no_grad(): y=forward_logits(model,x_rgb)
        assert tuple(y.shape)==(1,13), (name,tuple(y.shape))
        del model
    model=build_model('capsnet',pretrained=False).eval()
    with torch.no_grad(): y=forward_logits(model,x_gray)
    assert tuple(y.shape)==(1,13)
