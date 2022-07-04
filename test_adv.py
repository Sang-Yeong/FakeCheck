import torch
import torchattacks
from torchvision import transforms as transforms
from train import set_hyperparameters, Normalize
from dataset import load_data
import cv2
import numpy as np


def test_model(model, phase='test'):
    # phase = 'train', 'valid', 'test'

    model.eval()    # evaluate mode; gradient 계산 안함.
    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    '''
    with torch.no_grad():   # memory save를 위해 gradient 저장하지 않음.
    보통 test를 할 때, gradient를 training 시키는 것이 아니기 때문에 위와 같은 코드를 추가한다.
    
    grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
    하지만, adversarial attack은 위와 같이 gradient를 토대로 data에 공격을 가하기 때문에 gradient가 필요하다.
    
    따라서 test_adv 에는 with torch.no_grad()를 제외해야 한다.
    '''

    for i, (inputs, labels) in enumerate(dataloaders[phase]):

        # adversarial attack 정의
        atks = [torchattacks.FGSM(model, eps=8 / 255),
                torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=7),
                torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=7),
                ]

        adv_images = atks[0](inputs, labels).to(device)

        # Image Processing Based Defense Methods --> tensor를 image로 변환하여 적용
        for batch in range(inputs.shape[0]):
            tensor2img = transforms.ToPILImage()(inputs[batch]).convert('RGB')

            # 1. Resizing
            # Image.resize(size, resample=3, box=None, reducing_gap=None)
            # resample(filter): PIL.Image.NEAREST, PIL.Image.BOX, PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC
            tensor2img.resize((74, 74))
            tensor2img.resize((224, 224))

            # 다시 이미지를 tensor로 바꾸기
            tensor_img = transforms.ToTensor()(tensor2img)
            inputs[batch] = tensor_img

            # 2. jpeg compression
            tensor2numpy = inputs[batch].cpu().numpy()
            cv_img = np.transpose(tensor2numpy, (1, 2, 0))  # [w, h, c]
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

        labels = labels.to(device)

        outputs = model(adv_images)         # forward pass
        _, preds = torch.max(outputs, 1)    # model이 가장 높은 확률로 예측한 label
        loss = criterion(outputs, labels)   # loss 계산

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        num_cnt += inputs.size(0)  # batch size

        test_loss = running_loss / num_cnt
        test_acc = running_corrects.double() / num_cnt
        print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc * 100))


if __name__ == '__main__':
    dataloaders, _, _ = load_data()                             # dataset 불러오기
    criterion, device, _, _, _, _ = set_hyperparameters()       # hyper-parameters 불러오기
    model = torch.load('pytorch_model_adv_epoch100_4_sgd_atk3.pt')                  # train에서 모델 저장했던 모델 불러오기

    test_model(model)
