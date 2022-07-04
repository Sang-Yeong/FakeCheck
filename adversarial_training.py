import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import torchattacks
import torchvision.utils
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms as transforms
from dataset import load_data
from efficientnet_pytorch import EfficientNet


def train_model(device, dataloaders, batch_size, len_dataset, model, criterion, optimizer, scheduler, num_epochs=25):
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

            ratio_adv_ori = int((len_dataset // batch_size + 1) * 0.4)   # adversarial, original data 비율 정하기

            # batch 별로 나눠진 데이터 불러오기
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                # 설정한 비율에 따라 adversarial, original input으로 나누기
                if (phase == 'train' and (i < ratio_adv_ori)) or (phase == 'valid' and i % 2 == 0):
                    inputs = inputs.to(device)

                else:
                    # adversarial attack 정의
                    atks = [torchattacks.FGSM(model, eps=8 / 255),
                            torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=7),
                            torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7),
                            ]

                    inputs = atks[i % 3](inputs, labels).to(device)

                    # Image Processing Based Defense Methods --> tensor를 image로 변환하여 적용
                    for batch in range(inputs.shape[0]):
                        tensor2pil = transforms.ToPILImage()(inputs[batch]).convert('RGB')

                        # 1. Resizing
                        # Image.resize(size, resample=3, box=None, reducing_gap=None)
                        # resample(filter): PIL.Image.NEAREST, PIL.Image.BOX, PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC
                        tensor2pil.resize((74, 74))
                        tensor2pil.resize((224, 224))

                        # 다시 이미지를 tensor로 바꾸기
                        tensor_img = transforms.ToTensor()(tensor2pil)
                        inputs[batch] = tensor_img


                        # 2. jpeg compression
                        tensor2numpy = inputs[batch].cpu().numpy()
                        cv_img = np.transpose(tensor2numpy, (1, 2, 0))      # [w, h, c]
                        cv_img = cv_img * 255
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
                        result, encimg = cv2.imencode('.jpg', cv_img, encode_param)
                        if False == result:
                            print('could not encode image!')
                            quit()

                        # decode from jpeg format
                        jpeg_img = cv2.imdecode(encimg, 1)
                        jpeg2input = np.transpose(jpeg_img, (2, 0, 1)) / 255
                        inputs[batch] = torch.Tensor(jpeg2input).to(device)


                    # # save adversarial examples
                    # save_inputs = inputs.cpu().numpy()
                    # labels = labels.cpu().numpy()
                    # from matplotlib.pyplot import imsave
                    #
                    # for j in range(batch_size):
                    #     image = save_inputs[j, :, :, :]
                    #     label = labels[j]
                    #     if label == 0:
                    #         imsave(
                    #             f"C:/Users/mmclab1/Desktop/fakecheck/dataset/adv_img_examples/"
                    #             f"fake_adversarial_image_{j}.png",
                    #             np.transpose(image, (1, 2, 0)))
                    #     else:
                    #         imsave(
                    #             f"C:/Users/mmclab1/Desktop/fakecheck/dataset/adv_img_examples/"
                    #             f"real_adversarial_image_{j}.png",
                    #             np.transpose(image, (1, 2, 0)))



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
    PATH = 'pytorch_model_adv_epoch30_4_sgd_resize3_comp15.pt'
    model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), PATH)  # 모델 객체의 state_dict 저장
    torch.save(model, PATH)                 # 전체모델 저장
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

    plt.savefig('graph_adv_epoch30_4_sgd_resize3_comp15.png')
    plt.show()

    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc, inputs

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

def main():
    model_name = 'efficientnet-b0'   # 모델 설정
    model = EfficientNet.from_pretrained(model_name, num_classes=2)
    # model = EfficientNet.from_name('efficientnet-b0') 모델 구조 가져오기
    # model = EfficientNet.from_pretrained('efficientnet-b0') 모델이 이미 학습한 weight 가져오기

    # onnx로 모델 변환시 에러
    # RuntimeError: ONNX export failed: Couldn't export Python operator SwishImplementation
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

    # dataset.py에서 dataloaders 불러오기
    dataloaders, batch_size, len_dataset = load_data()

    # training을 위한 설정
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

    # 학습 돌리기
    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc, inputs = train_model(device, dataloaders, batch_size, len_dataset,
                                                                                                  model, criterion, optimizer, exp_lr_scheduler,
                                                                                                  num_epochs=50)
    return model, inputs

if __name__ == '__main__':
    _ = main()


