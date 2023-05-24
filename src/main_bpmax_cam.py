import sys
import os

proj_dir = os.path.abspath(os.getcwd())
print("proj_dir: ", proj_dir)
sys.path.append(proj_dir)

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.classifier import Classifier
from models.method import SAM
from tensorboardX import SummaryWriter
from models.resnet import ResBase, FeatB
from src.utils import load_network, load_data

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

import torch.nn.functional as F
import copy


def test(loader, model, classifier, device):
    with torch.no_grad():
        model.eval()
        classifier.eval()
        start_test = True
        val_len = len(loader['test0'])
        iter_val = [iter(loader['test' + str(i)]) for i in range(10)]
        for _ in range(val_len):
            data = [next(iter_val[j]) for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].to(device)
            labels = labels.to(device)
            outputs = []
            for j in range(10):
                feat, _ = model.inference(inputs[j])
                output = classifier(feat.cuda())
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


def train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=None, writer=None, model_path=None):
    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    best_acc = 0.0
    best_model = None

    for iter_num in range(1, args.max_iter + 1):
        model.train(True)
        classifier.train(True)
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])

        data_labeled = next(iter_labeled)

        img_labeled_q = data_labeled[0][0].to(device)
        label = data_labeled[1].to(device)

        feat_labeled, featmap_q, featcov16, bp_out_feat, network = model(img_labeled_q)

        # TODO: Changed here
        if args.use_bilinear:
            out = classifier(feat_labeled.cuda())
        else:
            out = classifier(feat_labeled.cuda())
        cam_weight = classifier.classifier_layer.weight


        # ----------------------------------------

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

        # TODO: Changed here
        if args.use_bilinear:
            # model for gradcam
            model_f = ResBase().cuda()
            model_f.conv1 = copy.deepcopy(network.conv1)
            model_f.bn1 = copy.deepcopy(network.bn1)
            model_f.relu = copy.deepcopy(network.relu)
            model_f.maxpool = copy.deepcopy(network.maxpool)
            model_f.layer1 = copy.deepcopy(network.layer1)
            model_f.layer2 = copy.deepcopy(network.layer2)
            model_f.layer3 = copy.deepcopy(network.layer3)
            model_f.layer4 = copy.deepcopy(network.layer4)

            net_bilinear = FeatB().cuda()
            net_bilinear.conv16 = copy.deepcopy(model.conv16)
            net_bilinear.bn16 = copy.deepcopy(model.bn16)

            model_cam = nn.Sequential(model_f, net_bilinear, classifier)  # classifier should be 16*2048->class_num
            target_layers = [model_f.layer4[-1]]
        else:
            model_cam = ResBase().cuda()

            # for cam
            for paramback, param_cam in zip(network.parameters(), model_cam.parameters()):
                param_cam.data.copy_(paramback.data)

            target_layers = [model_cam.layer4[-1]]

        # TODO: Changed here
        if args.use_bilinear:
            weight = cam_weight[label, :]
            weight = weight.to(device)
            weight = weight[:, :, None, None]
            weight_cam = weight.repeat(1, 1, 7, 7)

        # TODO: Changed here
        featmapcam = featmap_q * weight_cam
        featmapcam = torch.sum(featmapcam, dim=1)

        relu = nn.ReLU(inplace=True)
        featmapcam = relu(featmapcam)

        predictcam, _ = torch.max(featcov16, dim=1)
        predict_cam = predictcam.to(device)
        # temperature parameter in softmax
        t = 0.4
        featmapcam = featmapcam.view(featmapcam.size(0), -1)
        featmapcam = (featmapcam / t).float()
        featmapcam = featmapcam.detach()
        predict_cam = predict_cam.view(predict_cam.size(0), -1)
        predict_cam = (predict_cam / t).float()

        loss_cam_labeled_q = F.kl_div(predict_cam.softmax(dim=-1).log(), featmapcam.softmax(dim=-1),
                                      reduction='sum')

        # TODO: Remove this...
        if torch.isinf(loss_cam_labeled_q).any():
            """
            There was a problem before I moved down optimiser.zero(). I have to think that the gradients
            were getting polluted above.
            
            The classifier and kl div losses leading up to the infs are this:
            5.046670913696289 57.968990325927734
            9.408784866333008 58.49024200439453
            33.74223709106445 52.83906173706055
            38.71967315673828 51.91801452636719
            83.05970764160156 44.7819709777832
            98.13899993896484 47.25043487548828
            337.8395080566406 75.17976379394531
            5082.45654296875 87.66537475585938
            
            By comparison, the new code has the following losses:
            5.6416015625 58.445308685302734
            5.469156265258789 38.68390655517578
            5.374486923217773 53.212135314941406
            5.509685516357422 39.79650115966797
            6.054706573486328 44.83692169189453
            5.932514667510986 41.78059387207031
            5.885040283203125 31.523330688476562
            5.862998962402344 25.361289978027344
            6.09160852432251 27.82227897644043
            6.265558242797852 33.06561279296875
            6.15684175491333 20.362117767333984
            """
            predicted_has_inf = torch.isinf(predict_cam.softmax(dim=-1).log()).any()
            target_has_zeroes = (featmapcam.softmax(dim=-1) == 0).any()
            print('Is Inf')
        # print(iter_num, classifier_loss.item(), loss_cam_labeled_q.item())

        total_loss = classifier_loss + 0.01 * loss_cam_labeled_q

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {}; current acc: {}".format(iter_num, hit_num / float(sample_num)))
            print("cam_loss: {}; classifier_loss: {}".format(loss_cam_labeled_q, classifier_loss))

        # Show Loss in TensorBoard
        writer.add_scalar('loss/cam_loss', loss_cam_labeled_q, iter_num)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/total_loss', total_loss, iter_num)

        if iter_num % args.test_interval == 1:
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

    print("best acc: %.4f" % best_acc)
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
    parser.add_argument('--use_bilinear', action='store_true', default=True, help='If True, Use SAM Bilinear')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum hyperparameter')
    parser.add_argument('--projector_dim', type=int, default=1024)
    parser.add_argument('--class_num', type=int, default=200)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--max_iter', type=float, default=30000)
    parser.add_argument('--test_interval', type=float, default=1000)
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
        help='Reduce noise by taking the first principle component'
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

    method_name = 'SAM_logConfid'
    method_name += '_qdim' + str(args.projector_dim)

    method_name += '_seed' + str(args.seed)

    logdir = os.path.join(args.logdir, method_name)
    logdir = f"{logdir}{model_name}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    model_path = os.path.join(logdir, "%s_best.pkl" % model_name)

    # Initialize model
    network, feature_dim = load_network(args.backbone)
    model = SAM(network=network, backbone=args.backbone, use_bilinear=args.use_bilinear,
                projector_dim=args.projector_dim,
                class_num=args.class_num, pretrained=args.pretrained, pretrained_path=args.pretrained_path).to(device)

    # TODO: Changed here
    if args.use_bilinear:
        classifier = Classifier(2048*1, args.class_num).to(device)  ##I have changed!!!
    else:
        classifier = Classifier(2048, args.class_num).to(device)

    print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
    print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))

    # Define Optimizer
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio},

    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000, 30000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # Train model
    train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=device, writer=writer,
          model_path=model_path)


if __name__ == '__main__':
    main()
