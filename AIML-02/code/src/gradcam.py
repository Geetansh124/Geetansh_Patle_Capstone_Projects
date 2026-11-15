"""Grad-CAM for DenseNet121 classifier checkpoints; visualization is not clinical evidence."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from train_classifier import make_model, DISCLAIMER


def generate(image_path: Path, checkpoint: Path, output: Path):
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False); model=make_model(len(ckpt['labels'])); model.load_state_dict(ckpt['state_dict']); model.eval()
    activation={}; gradient={}; target=model.features.denseblock4
    target.register_forward_hook(lambda _m,_i,o: activation.setdefault('value',o))
    target.register_full_backward_hook(lambda _m,_gi,go: gradient.setdefault('value',go[0]))
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    x=transform(Image.open(image_path).convert('RGB')).unsqueeze(0); logits=model(x); class_id=int(logits.argmax(1)); model.zero_grad(); logits[0,class_id].backward()
    weights=gradient['value'].mean((2,3),keepdim=True); cam=torch.relu((weights*activation['value']).sum(1,keepdim=True)); cam=torch.nn.functional.interpolate(cam,size=(224,224),mode='bilinear',align_corners=False)[0,0].detach().numpy(); cam=(cam/(cam.max()+1e-8)*255).astype('uint8')
    base=np.asarray(Image.open(image_path).convert('RGB').resize((224,224))).astype('float32'); heat=np.zeros_like(base); heat[:,:,0]=cam; overlay=(.55*base+.45*heat).clip(0,255).astype('uint8'); output.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(overlay).save(output)
    print(f'{DISCLAIMER} predicted_label={ckpt["labels"][class_id]} model_score={float(torch.sigmoid(logits[0,class_id])):.4f} overlay={output}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=DISCLAIMER); p.add_argument('--image',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('artifacts/gradcam.png')); a=p.parse_args(); generate(a.image,a.checkpoint,a.output)
