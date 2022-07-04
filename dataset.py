import torch
import random
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def make_dataset(dataset):
    # data load
    batch_size  = 64
    random_seed = 555
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # data split
    train_idx, tmp_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=random_seed)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    tmp_dataset       = Subset(dataset, tmp_idx)

    val_idx, test_idx = train_test_split(list(range(len(tmp_dataset))), test_size=0.5, random_state=random_seed)
    datasets['valid'] = Subset(tmp_dataset, val_idx)
    datasets['test']  = Subset(tmp_dataset, test_idx)

    ## data loader 선언
    dataloaders, batch_num = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=4)
    dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'],
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=4)
    dataloaders['test']  = torch.utils.data.DataLoader(datasets['test'],
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=4)
    batch_num['train'], batch_num['valid'], batch_num['test'] = len(dataloaders['train']), len(dataloaders['valid']), len(dataloaders['test'])
    print('batch_size : %d,  tvt : %d / %d / %d' % (batch_size, batch_num['train'], batch_num['valid'], batch_num['test']))

    return dataloaders, batch_size


def load_data():
    data_path = './dataset/test/'
    dataset = datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize((224, 224)),   # EfficientNet은 ImageNet에 맞춰 학습되있음 --> ImageNet의 입력: 224x224 형식
            transforms.ToTensor(),           # 이미지 픽셀값의 범위 [0,1]
        ]))

    '''
    adversarial attack을 적용하기 위해 torchattacks 모듈 사용
    torchattacks의 각 공격은 'adv_images = torch.clamp(adv_images, min=0, max=1).detach()' 로 구현되어 있음.
    따라서 이미지 픽셀값의 범위는 [0,1]로 설정되어야 함.
    --> normalize를 수행 할 경우, 이미지 픽셀값의 범위가 변하기 때문에 transforms.ToTensor()까지만 사용
    
    * normalize는 train.py에서 Normalize class를 만들어 모델에 추가하여 수행
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(
        norm_layer,
        model
    )
    
    * The images have to be loaded in to a range of [0, 1] and then normalized using
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
    
    * ToTensor: Numpy 형태의 이미지 데이터를 Tensor형태로 바꿔주는 역할
    (Height x Width x Channels), 0~255의 값 --> torch.Float.Tensor 형태: (C x H x W) 순서와 0.0 부터 1.0사이의 값들로 변환
    '''

    dataloaders, batch_size = make_dataset(dataset)
    return dataloaders, batch_size, len(dataset.imgs)