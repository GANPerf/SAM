import sys
import os
proj_dir = os.path.abspath(os.getcwd())
print("proj_dir: ", proj_dir)
sys.path.append(proj_dir)

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models.classifier import Classifier
from models.method import SAM
from tensorboardX import SummaryWriter
from src.utils import load_network, load_data

#from src.CompactBilinearPooling import CompactBilinearPooling
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

from PIL import Image
from torchvision import models
import argparse
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

import torch.nn.functional as F


def test(loader, model, classifier, device):
    with torch.no_grad():
        model.eval()
        classifier.eval()
        start_test = True
        val_len = len(loader['test0'])
        iter_val = [iter(loader['test' + str(i)]) for i in range(10)]
        for _ in range(val_len):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].to(device)
            labels = labels.to(device)
            outputs = []
            for j in range(10):
                feat,_ = model.inference(inputs[j]) 
                output,_ = classifier(feat.cuda())
                outputs.append(output)
            outputs = sum(outputs)
            if start_test:
                all_outputs = outputs.data.float()
                all_labels = labels.data.float()
                start_test = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.data.float()), 0)
                all_labels = torch.cat((all_labels, labels.data.float()), 0)
        _, predict = torch.max(all_outputs, 1)
        
        accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / float(all_labels.size()[0])
    return accuracy

def train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=None, writer=None, model_path = None):

    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    best_acc = 0.0
    best_model = None

    for iter_num in range(1, args.max_iter + 1):
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])
        

        data_labeled = iter_labeled.next()

        img_labeled_q = data_labeled[0][0].to(device)
        label = data_labeled[1].to(device)

        feat_labeled, featmap_q, featcov16, bp_out_feat = model(img_labeled_q)
		
        out, cam_weight = classifier(feat_labeled.cuda())

        #CAM

        weight = cam_weight[label,:]
        weight = weight.to(device)
        weight = weight[:, :, None, None]
        weight_cam = weight.repeat(1, 1, 7, 7)  

        #----------------------------------------
        
        classifier_loss = criterions['CrossEntropy'](out, label)

        
        methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
        network, feature_dim = load_network(args.backbone)
        modelcam = network()
        target_layers = [modelcam.layer4[-1]]

        input_tensor_labeled_q = img_labeled_q

        target_category = label

        #GradCAM
        cam_algorithm = methods[args.method] 

        with cam_algorithm(model=modelcam,
                           target_layers=target_layers,
                           use_cuda=args.use_cuda) as cam:
        
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 24
        
            grayscale_cam, weights_gradcam = cam(input_tensor=input_tensor_labeled_q,
                                target_category=target_category,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)
            

            # weight_gradcam/weight_cam
            featmapcam = featmap_q*weight_cam
            featmapcam = torch.sum(featmapcam, dim=1)


            relu = nn.ReLU(inplace=True)
            featmapcam = relu(featmapcam)
			

            predictcam, _ = torch.max(featcov16, dim=1)
            predict_cam = predictcam.to(device)
            #temperature parameter in softmax
            t = 0.4
            featmapcam = featmapcam.view(featmapcam.size(0), -1)
            featmapcam = (featmapcam/t).float()
            featmapcam = featmapcam.detach()
            predict_cam = predict_cam.view(predict_cam.size(0),-1)
            predict_cam = (predict_cam/t).float()


            loss_cam_labeled_q = F.kl_div(predict_cam.softmax(dim=-1).log(), featmapcam.softmax(dim=-1), reduction='sum')
          
       

        total_loss = classifier_loss + 0.01*loss_cam_labeled_q
        total_loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()


        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {}; current acc: {}".format(iter_num, hit_num / float(sample_num)))

        ## Show Loss in TensorBoard
        writer.add_scalar('loss/cam_loss', loss_cam_labeled_q, iter_num)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/total_loss', total_loss, iter_num)
        
        if iter_num % args.test_interval == 1 or iter_num == 500:
            model.eval()
            classifier.eval()
            test_acc = test(dataset_loaders, model, classifier, device=device)
            print("iter_num: {}; test_acc: {}".format(iter_num, test_acc))
            writer.add_scalar('acc/test_acc', test_acc, iter_num)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = {'model': model.state_dict(),
                              'classifier': classifier.state_dict(),
                              'step': iter_num
                              }

    print("best acc: %.4f" % (best_acc))
    torch.save(best_model, model_path)
    print("The best model has been saved in ", model_path)

def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--label_ratio', type=int, default=15)
    parser.add_argument('--logdir', type=str, default='../vis/')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--seed', type=int, default='666666')
    parser.add_argument('--workers', type=int, default='4')
    parser.add_argument('--lr_ratio', type=float, default='10')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum hyperparameter')
    parser.add_argument('--projector_dim', type=int, default=1024)
    parser.add_argument('--class_num', type=int, default=200)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--max_iter', type=float, default=30000)
    parser.add_argument('--test_interval', type=float, default=3000)
    parser.add_argument("--pretrained", action="store_true", help="use the pre-trained model")
    parser.add_argument("--pretrained_path", type=str, default='~/.torch/models/moco_v2_800ep_pretrain.pth.tar')

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/test.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    

    configs = parser.parse_args()
    configs.use_cuda = configs.use_cuda and torch.cuda.is_available()
    if configs.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return configs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    args = read_config()
    set_seed(args.seed)

    # Prepare data
    if 'CUB200' in args.root:
        args.class_num = 200
    elif 'StanfordCars' in args.root:
        args.class_num = 196
    elif 'Aircraft' in args.root:
        args.class_num = 100

    dataset_loaders = load_data(args)
    print("class_num: ", args.class_num)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')


    model_name = "%s_%s_%s" % (args.backbone, os.path.basename(args.root), str(args.label_ratio))

    logdir = os.path.join(args.logdir, model_name)
    method_name = 'SAM_logConfid'
    method_name += '_qdim' + str(args.projector_dim)

    method_name += '_seed' + str(args.seed)

    logdir = os.path.join(args.logdir, method_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    model_path = os.path.join(logdir, "%s_best.pkl" % (model_name))

    # Initialize model
    network, feature_dim = load_network(args.backbone)
    model = SAM(network=network, backbone=args.backbone, projector_dim=args.projector_dim,
                       class_num=args.class_num, pretrained=args.pretrained, pretrained_path=args.pretrained_path).to(device)
    classifier = Classifier(2048, args.class_num).to(device)

    print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
    print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))

    ## Define Optimizer
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio},

    ], lr= args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000, 30000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # Train model
    train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=device, writer=writer, model_path=model_path)

   
if __name__ == '__main__':
    main()
