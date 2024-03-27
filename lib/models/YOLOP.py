import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
#sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Concat, Detect,  RepNCSPELAN4 , ADown, SPPELAN
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized




# YOLOv9 backbone
YOLOP = [
[22, 31],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx  
   [-1, Conv, [3, 64, 3, 2]],  # 0-P1/2
   [-1, Conv, [64, 128, 3, 2]],  # 1-P2/4
   [-1, RepNCSPELAN4, [128, 256, 128, 64, 1]],  # 2
   [-1, ADown, [256,256]],  # 3-P3/8
   [-1, RepNCSPELAN4, [256,512, 256, 128, 1]],  # 4
   [-1, ADown, [512,512]],  # 5-P4/16
   [-1, RepNCSPELAN4, [512,512, 512, 256, 1]],  # 6
   [-1, ADown, [512,512]],  # 7-P5/32
   [-1, RepNCSPELAN4, [512,512, 512, 256, 1]],  # 8


  
   # elan-spp block
   [-1,SPPELAN, [512, 512, 256]],  # 9
   [-1,nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], Concat, [1]],  # cat backbone P4
   [-1,RepNCSPELAN4, [1024,512, 512,256, 1]],  # 12
   [-1,nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], Concat, [1]],  # cat backbone P3
   [-1,RepNCSPELAN4, [1024, 256, 256, 128, 1]],  # 15 (P3/8-small)
   [-1,ADown, [256,256]],
   [[-1, 12], Concat, [1]],  # cat head P4
   [-1, RepNCSPELAN4, [768,512, 512, 256, 1]],  # 18 (P4/16-medium)
   [-1, ADown, [512,512]],
   [[-1, 9], Concat, [1]],  # cat head P5

   # elan-2 block
   [-1, RepNCSPELAN4, [1024, 512,512, 256, 1]],  # 21 (P5/32-large)

    # [ [15, 18, 21], Detect,  [1, []], #Detection head 22
    [[15, 18, 21], Detect, [1,[256,512,512]]],
    # [[15,18,21],  DualDDetect, [1, [256, 512, 512]]], # DualDDetect(A3, A4, A5, P3, P4, P5)


    [ 15, Conv, [256, 128, 3, 1]],   #23
    [ -1, Upsample, [None, 2, 'nearest']],  #24
    [ -1, BottleneckCSP, [128, 64, 1, False]],  #25
    [ -1, Conv, [64, 32, 3, 1]],    #26
    [ -1, Upsample, [None, 2, 'nearest']],  #27
    [ -1, Conv, [32, 16, 3, 1]],    #28
    [ -1, BottleneckCSP, [16, 8, 1, False]],    #29
    [ -1, Upsample, [None, 2, 'nearest']],  #30
    [ -1, Conv, [8, 2, 3, 1]]] #31 Driving area segmentation head


#    [[31, 34, 37, 16, 19, 22],  DualDDetect, [1, [512, 512, 512,256,512,512]]]]  # DualDDetect(A3, A4, A5, P3, P4, P5)


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 512  ### from 128 # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
                # print("stride:",Detector.stride)
            # print("stride"+str(Detector.stride ))
            # Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            # check_anchor_order(Detector)
            self.stride = Detector.stride
            # self._initialize_biases() ####
            Detector.bias_init() ####
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            # if type(x) == list:
            #       print("i:", i)
            #       print(x[0].shape, x[1].shape, x[2].shape)
            # else:
            #     print("i:",i)
            #     print(x.shape)
            if i in self.seg_out_idx:     #save driving area segment result
                m=nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        return out
            
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_net(cfg, **kwargs): 
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    total_params = sum(p.numel() for p in model.parameters()) 
    total_params_m = total_params/1e6 
    print(f"Total parameters: {total_params_m}") 
    # input_ = torch.randn((1, 3, 256, 256))
    input_ = torch.randn((1, 3, 384, 640))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out, lane_line_seg = model(input_) ##added sad2
    # detects, dring_area_seg, lane_line_seg = model_out  ###
    detects, anchor2_detects, anchor3_detects = model_out    ##changed them to this
    # Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(anchor2_detects.shape)
    print(anchor3_detects.shape)
    
    print(lane_line_seg.shape)
