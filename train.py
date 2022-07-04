import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import matplotlib.pyplot as plt

from dataset import load_data
from efficientnet_pytorch import EfficientNet



def train_model(device, dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    '''
    * state_dict: 각 layer 를 매개변수 텐서로 매핑하는 Python 사전(dict) 객체
    - layer; learnable parameters (convolutional layers, linear layers, etc.), registered buffers (batchnorm’s running_mean)
    - Optimizer objects (torch.optim)
    '''
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 epoch마다 training, validation phase 나눠줌.
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # training mode
            else:
                model.eval()   # evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # batch 별로 나눠진 데이터 불러오기
            for inputs, labels in dataloaders[phase]:
                # inputs, labels device('CPU' or 'GPU')로 보내서 계산
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 학습 가능한 가중치인 "optimizer 객체" 사용하여, 갱신할 변수들에 대한 모든 변화도 0으로 설정
                # backward() 호출시, 변화도가 buffer 에 덮어쓰지 않고 누적되기 때문.
                optimizer.zero_grad()

                # forward pass
                # gradient 계산하는 모드로, 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)             # h(x) 값, 모델의 예측 값
                    _, preds = torch.max(outputs, 1)    # dim = 1, output의 각 sample 결과값(row)에서 max값 1개만 뽑음.
                    loss = criterion(outputs, labels)   # h(x) 모델이 잘 예측했는지 판별하는 loss function


                    # training phase에서만 backward + optimize 수행
                    if phase == 'train':
                        loss.backward()     # gradient 계산
                        optimizer.step()    # parameter update

                # statistics
                running_loss += loss.item() * inputs.size(0)            # inputs.size(0) == batch size
                running_corrects += torch.sum(preds == labels.data)     # True == 1, False == 0, 총 정답 수
                num_cnt += len(labels)                                  # len(labels) == batch size

            if phase == 'train':
                scheduler.step()    # Learning Rate Scheduler

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #                 best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

    # load best model weights
    PATH = 'pytorch_model.pt'
    model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), PATH)
    torch.save(model, PATH)
    torch.save(model.state_dict(), f'C:/Users/mmclab1/.cache/torch/hub/checkpoints/{PATH}')
    print('model saved')

    # train, validation의 loss, acc 그래프로 나타내기
    plt.subplot(311)
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(313)
    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig('train_model_epoch20_adam_lr0.001.png')
    plt.show()

    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc, inputs



'''
adversarial attack을 위해 dataset불러오는 부분에서 하지 못했던 normalize를 model의 layer로 추가해 설정
normalize 클래스를 만들고 norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])를 통해 수행
'''
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def set_hyperparameters():

    '''
    #######################
    ###  network 설정  ####
    ######################
    '''
    model_name = 'efficientnet-b0'   # 모델 설정
    model = EfficientNet.from_pretrained(model_name, num_classes=2)
    # model = EfficientNet.from_name('efficientnet-b0') 모델 구조 가져오기
    # model = EfficientNet.from_pretrained('efficientnet-b0') 모델이 이미 학습한 weight 가져오기
    '''
    * ImageNet으로 이미 학습한 weight 가져오는 이유
    14 million 개의 이미지와 20,000개의 카테고리로 이루어진 ImageNet
    20,000개 중 1,000개의 class로 학습을 시켰을 때, 다양한 클래스이기 때문에 nn이 일반적인 성능을 내도록 유도됨.
    클래스에 특화되어있는 특정 feature를 뽑아내는 것을 넘어 , edge detection 처럼 일반적인 구성을 나타냄.
    '''

    model.set_swish(memory_efficient=False)

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(
        norm_layer,
        model
    )

    # TODO: 미리 학습된 weight고정 --> 뒷 부분 2단계만 학습함.
    # fc 제외하고 freeze
    # for n, p in model.named_parameters():
    #     if '_fc' not in n:
    #         p.requires_grad = False
    # model = torch.nn.parallel.DistributedDataParallel(model)


    '''
    #######################
    # training을 위한 설정 #
    ######################
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.05,
                          momentum=0.9,
                          weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    lmbda = lambda epoch: 0.98739
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    num_epochs = 20

    return criterion, device, model, optimizer, exp_lr_scheduler, num_epochs



def main():
    # hyper-paramters 받아오기
    criterion, device, model, optimizer, exp_lr_scheduler, num_epochs = set_hyperparameters()

    # dataset.py에서 dataloaders 불러오기
    dataloaders, _, _ = load_data()

    # 학습 돌리기
    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc, inputs = train_model(device, dataloaders, model,
                                                                                                  criterion,
                                                                                                  optimizer,
                                                                                                  exp_lr_scheduler,
                                                                                                  num_epochs=num_epochs)

    return inputs

if __name__ == '__main__':
    main()


