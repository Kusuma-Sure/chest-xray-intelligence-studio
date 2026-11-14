from __future__ import annotations
import base64, io
import numpy as np
import torch
from PIL import Image
from .models import forward_logits


def _normalize(x: np.ndarray) -> np.ndarray:
    x=np.maximum(x,0)
    lo=float(x.min()); hi=float(x.max())
    if hi-lo < 1e-8: return np.zeros_like(x,dtype=np.float32)
    return ((x-lo)/(hi-lo)).astype(np.float32)


def _overlay_data_url(original: Image.Image, heat: np.ndarray) -> str:
    heat_img=Image.fromarray(np.uint8(np.clip(heat,0,1)*255)).resize(original.size,Image.Resampling.BILINEAR)
    h=np.asarray(heat_img,dtype=np.float32)/255.0
    # Blue -> cyan -> yellow attribution palette, generated from the measured map.
    r=np.clip(1.8*h-0.45,0,1); g=np.clip(1.6*h,0,1); b=np.clip(1.0-0.8*h,0,1)
    colored=np.stack([r,g,b],axis=-1)*255
    base=np.asarray(original.convert('RGB'),dtype=np.float32)
    blended=np.clip(0.62*base+0.38*colored,0,255).astype(np.uint8)
    out=Image.fromarray(blended)
    buf=io.BytesIO(); out.save(buf,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')


def explain_prediction(model, model_name: str, input_tensor: torch.Tensor,
                       original: Image.Image, target_index: int) -> dict:
    """Create an activation/attention map from the selected trained model only."""
    model.eval(); model.zero_grad(set_to_none=True)
    activation={}; handle=None
    target=None
    if model_name=='densenet': target=model.features.denseblock4
    elif model_name=='capsnet': target=model.conv3
    elif model_name=='vit': target=model.encoder.layers[-1]
    elif model_name=='gat':
        with torch.no_grad():
            _=forward_logits(model,input_tensor)
            amap=getattr(model,'last_attention_map',None)
        if amap is None: return {'available':False,'method':'GAT region attention','reason':'The GAT model did not return an attention map.'}
        heat=amap[0].detach().cpu().numpy()
        return {'available':True,'method':'GAT learned 7x7 region attention','overlay':_overlay_data_url(original,_normalize(heat))}
    else:
        return {'available':False,'method':'Not supported','reason':'No attribution implementation is configured for this model.'}

    def hook(_module,_inputs,output):
        tensor=output[0] if isinstance(output,(tuple,list)) else output
        activation['value']=tensor
        if tensor.requires_grad:
            tensor.retain_grad()
    handle=target.register_forward_hook(hook)
    try:
        logits=forward_logits(model,input_tensor)
        if not logits.requires_grad: return {'available':False,'reason':'Gradients are unavailable.'}
        logits[0,target_index].backward()
        act=activation.get('value'); grad=act.grad if act is not None else None
        if grad is None: return {'available':False,'reason':'No activation gradient was returned.'}
        if act.ndim==4:
            weights=grad.mean(dim=(2,3),keepdim=True)
            cam=torch.relu((weights*act).sum(dim=1))[0]
        elif act.ndim==3:
            # TorchVision ViT sequence: token 0 is CLS; 196 patch tokens form 14x14.
            score=(grad*act).sum(dim=-1)[0,1:]
            side=int(round(score.numel()**0.5))
            if side*side != score.numel():
                return {'available':False,'reason':'ViT patch tokens did not form a square grid.'}
            cam=torch.relu(score.reshape(side,side))
        else:
            return {'available':False,'reason':f'Unsupported activation shape: {tuple(act.shape)}'}
        heat=_normalize(cam.detach().float().cpu().numpy())
        method={'densenet':'Grad-CAM · DenseNet121 denseblock4','capsnet':'Gradient CAM · CapsNet conv3','vit':'Gradient × Activation · ViT-B/16'}[model_name]
        return {'available':True,'method':method,'overlay':_overlay_data_url(original,heat)}
    finally:
        if handle is not None: handle.remove()
        model.zero_grad(set_to_none=True)
