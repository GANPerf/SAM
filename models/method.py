import torch
import torch.nn as nn
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from torchvision.models import resnet50


class SAM(nn.Module):
    def __init__(self,
                 network,
                 backbone,
                 use_bilinear: bool,
                 projector_dim=1024,
                 class_num=200,
                 pretrained=True,
                 pretrained_path=None):
        """
        network: the network of the backbone
        backbone: the name of the backbone
        use_bilinear: If true, use bilinear pooling
        feature_dim: the dimension of the output from the backbone
        class_num: the class number of the dataset
        pretrained: loading from pre-trained model or not (default: True)
        pretrained_path: the path of the pre-trained model
        """
        super(SAM, self).__init__()
        self.use_bilinear = use_bilinear
        self.class_num = class_num
        self.backbone = backbone
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path

        # create the encoders

        self.encoder = network(projector_dim=projector_dim)

        self.load_pretrained(network)

        self.conv16 = nn.Conv2d(2048, 16, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(16)

        self.bncbp = nn.BatchNorm1d(500)
        self.bncbp.bias.requires_grad_(False)
        nn.init.constant_(self.bncbp.weight, 1)
        nn.init.constant_(self.bncbp.bias, 0)

        self.mcb = CompactBilinearPooling(2048, 16, 500).cuda()

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, im_q):

        q_c, q_f, featmap = self.encoder(im_q)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)

    
        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        img = torch.from_numpy(img)
        img = img[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)
        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)
 

        '''
        #CBP
        cbp_dim = 500
        N,C1,H,W = featcov16.shape
        _,C2,_,_ = featmap_q.shape
        feat_part = featcov16.clone().permute(0,2,3,1).contiguous().view(-1,C1)
        feat_whole = featmap_q.clone().permute(0,2,3,1).contiguous().view(-1,C2)
        bp_out_feat = self.mcb(feat_whole.cuda(), feat_part.cuda())
        bp_out_feat = bp_out_feat.view(N,H,W,cbp_dim).permute(0,3,1,2).contiguous()
        bp_out_feat = bp_out_feat.view(N,cbp_dim,-1).sum(-1)
        bp_out_feat = self.bncbp(bp_out_feat.cuda())
        '''

        return q_f, featmap, featcov16, bp_out_feat, self.encoder

    def load_pretrained(self, network):
        if 'resnet' in self.backbone:
            q = network(projector_dim=1000, pretrained=self.pretrained)
            q.fc = self.encoder.fc
            self.encoder = q

    def inference(self, img):
        y, feat, featmap = self.encoder(img)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)

        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        img = torch.from_numpy(img)
        img = img[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)
        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)


        '''
        # CBP
        cbp_dim = 500
        N, C1, H, W = featcov16.shape
        _, C2, _, _ = featmap_q.shape
        feat_part = featcov16.clone().permute(0, 2, 3, 1).contiguous().view(-1, C1)
        feat_whole = featmap_q.clone().permute(0, 2, 3, 1).contiguous().view(-1, C2)
        bp_out_feat = self.mcb(feat_whole.cuda(), feat_part.cuda())
        bp_out_feat = bp_out_feat.view(N, H, W, cbp_dim).permute(0, 3, 1, 2).contiguous()
        bp_out_feat = bp_out_feat.view(N, cbp_dim, -1).sum(-1)
        bp_out_feat = self.bncbp(bp_out_feat.cuda())
        '''
        # TODO: Changed this
        if self.use_bilinear:
            return bp_out_feat.cuda(), featcov16
        else:
            return feat.cuda(), featcov16
