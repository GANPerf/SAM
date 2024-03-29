from models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101

from data.tranforms import TransformTrain
from data.tranforms import TransformTest
import data
from torch.utils.data import DataLoader
import os

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def load_data(args, num_workers: int = 0):
    batch_size_dict = {"train": args.batch_size, "test": 100}

    transform_train = TransformTrain()
    transform_test = TransformTest(mean=imagenet_mean, std=imagenet_std)

    if os.path.basename(args.root).find('CUB_200') > -1:
        dataset_name = 'CUB200'
    else:
        dataset_name = os.path.basename(args.root)
    dataset = data.__dict__[dataset_name]

    download = False
    datasets = {"train": dataset(root=args.root,
                                 split='train',
                                 label_ratio=args.label_ratio,
                                 download=download,
                                 transform=transform_train)}
    test_dataset = {
        'test' + str(i): dataset(root=args.root,
                                 split='test',
                                 label_ratio=100,
                                 download=download,
                                 transform=transform_test["test" + str(i)]) for i in range(10)
    }
    datasets.update(test_dataset)

    dataset_loaders = {x: DataLoader(datasets[x], batch_size=batch_size_dict[x], shuffle=True, num_workers=num_workers)
                       for x in ['train']}
    dataset_loaders.update({'test' + str(i): DataLoader(datasets["test" + str(i)],
                                                        batch_size=4, shuffle=False, num_workers=num_workers)
                            for i in range(10)})

    return dataset_loaders


def load_network(backbone):
    if 'resnet' in backbone:
        if backbone == 'resnet18':
            network = resnet18
            feature_dim = 512
        elif backbone == 'resnet34':
            network = resnet34
            feature_dim = 512
        elif backbone == 'resnet50':
            network = resnet50
            feature_dim = 2048
        elif backbone == 'resnet101':
            network = resnet101
            feature_dim = 2048
        elif backbone == 'resnet152':
            network = resnet152
            feature_dim = 2048
        else:
            raise ValueError(f'Unknown backbone: {backbone}')
    else:
        network = resnet50
        feature_dim = 2048

    return network, feature_dim
