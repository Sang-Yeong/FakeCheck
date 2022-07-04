import torch
from train import set_hyperparameters, Normalize
from dataset import load_data
from torchvision import transforms as transforms


def test_model(model, phase='train'):
    # phase = 'train', 'valid', 'test'

    model.eval()    # evaluate mode; gradient 계산 안함.
    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    with torch.no_grad():   # memory save를 위해 gradient 저장하지 않음.
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)

            # # Image Processing Based Defense Methods --> tensor를 image로 변환하여 적용
            # for batch in range(inputs.shape[0]):
            #     tensor2img = transforms.ToPILImage()(inputs[batch]).convert('RGB')
            #
            #     # 1. Resizing
            #     # Image.resize(size, resample=3, box=None, reducing_gap=None)
            #     # resample(filter): PIL.Image.NEAREST, PIL.Image.BOX, PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC
            #     tensor2img.resize((74, 74))
            #     tensor2img.resize((224, 224))
            #
            #     # 다시 이미지를 tensor로 바꾸기
            #     tensor_img = transforms.ToTensor()(tensor2img)
            #     inputs[batch] = tensor_img


            labels = labels.to(device)

            outputs = model(inputs)             # forward pass
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
