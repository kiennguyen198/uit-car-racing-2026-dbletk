"""Portable detector: only Torch, Torchvision and OpenCV from BTC's image.

Input is native BGR GetRaw. This module returns visible boxes, never a route.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms


class SignObstacleDetector:
    def __init__(self, checkpoint, config, device='cuda:0'):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; check GPU setup before driving')
        self.config = json.loads(Path(config).read_text())
        self.names = self.config['class_names']
        self.size = self.config['imgsz']
        self.model = torch.jit.load(str(checkpoint),map_location=self.device).eval()
        self.dtype = next(self.model.parameters()).dtype
        self.thresholds = torch.tensor([self.config['confidence'][n] for n in self.names],device=self.device)

    @torch.inference_mode()
    def predict(self, bgr):
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError('Expected HWC uint8 BGR GetRaw image')
        h,w = bgr.shape[:2]
        ratio = min(self.size/w,self.size/h)
        nw,nh = round(w*ratio),round(h*ratio)
        left,top = round((self.size-nw)/2-0.1),round((self.size-nh)/2-0.1)
        right,bottom = self.size-nw-left,self.size-nh-top
        resized = cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
        chw = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))
        tensor = torch.from_numpy(chw).to(self.device,dtype=self.dtype).unsqueeze(0)/255.0
        prediction = self.model(tensor)
        if isinstance(prediction,(list,tuple)):
            prediction = prediction[0]
        values = prediction[0].transpose(0,1).float()
        scores,classes = values[:,4:].max(dim=1)
        selected = scores >= self.thresholds[classes]
        values,scores,classes = values[selected],scores[selected],classes[selected]
        if not len(values):return []
        center,size = values[:,:2],values[:,2:4]
        boxes = torch.cat((center-size/2,center+size/2),dim=1)
        keep = batched_nms(boxes,scores,classes,self.config['iou'])[:30]
        boxes = boxes[keep]
        boxes[:,[0,2]] = (boxes[:,[0,2]]-left)/ratio
        boxes[:,[1,3]] = (boxes[:,[1,3]]-top)/ratio
        boxes[:,0].clamp_(0,w);boxes[:,2].clamp_(0,w)
        boxes[:,1].clamp_(0,h);boxes[:,3].clamp_(0,h)
        packed = torch.cat((boxes,scores[keep,None],classes[keep,None].float()),dim=1).cpu().numpy()
        return [{'class':self.names[int(row[5])],'confidence':float(row[4]),'box':row[:4].tolist()} for row in packed]

    def warmup(self, repetitions=10):
        image = np.zeros((180,320,3),dtype=np.uint8)
        for _ in range(repetitions):self.predict(image)
        if self.device.type == 'cuda':torch.cuda.synchronize()
