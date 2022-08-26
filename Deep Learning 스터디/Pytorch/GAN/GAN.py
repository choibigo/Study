import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

parameters = {
    "epoch" : 100,
    "save_epoch" : 5,
    "batch_size" : 128,
    "image_size" : 28,
    "lr" : 0.0002,
    "latent_vector_dim" : 100,
    "save_base_path" : "D:\\temp\\save_model\\GAN\\"
}

class Generator(nn.Module):
    """
    [Generator]
    - 생성자에 충분한 수의 매개 변수를 확보하기 위해, 4개의 선형 레이어를 쌓아서 생성자를 만든다.
    - 선형 레이어는 속해잇는 모든 뉴련이 이전 레이어의 모든 뉴련과 연결되어 있는 가장 단순한 레이어 이다.
    - 100차원의 랜덤 벡터를 받아 이를 256개의 뉴런을 가진 레이어로 보내고, 다시 레이어의 크기를 맞춰 512, 1024로 증가 시켰다. 
    - 마지막은 MNIST와 크기를 맞추기 위해 28X28 로 줄인다.

    - 각 레이어 홤수는 LeakyReLU를 사용 했다, LeakyReLU는 각 뉴런의 출력값이 0보다 높으면 그대로 두고, 0보다 작으면 정해진 작은 숫자를 곱하는 간단한 활성화 함수 이다.
    - 생성자 마지막 레이어는 출력값을 픽셀값의 범위인 -1과 1로 만들기 위해 Tanh를 사용 했다.
    - 레이어와 활성화 함수를 쌓은 덕분에 MNIST의 데이터 분포를 근사할 수 있는 충분판 표현력을 얻었다.
    - 더욱 복잡한 문제를 풀기 위해서는 더 깊은 레이어 구조와 더 많은 양의 매개 변수가 필요 하다.
    """
    def __init__(self, ouput_image_size, latent_vector_dim):
        super(Generator, self).__init__()
        self.ouput_image_size = ouput_image_size
        self.latent_vector_dim = latent_vector_dim
        self.main = nn.Sequential(
            nn.Linear(in_features=self.latent_vector_dim, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=self.ouput_image_size * self.ouput_image_size),
            nn.Tanh())
        
    def forward(self, inputs):
        out = self.main(inputs) # generator 결과
        out = out.view(-1, 1, self.ouput_image_size, self.ouput_image_size) # 결과를 이미지 사이즈 X 이미지 사이즈 크기로 만들어 준다.
        # TO-DO : Channel 은?
        return out

class Discriminator(nn.Module):
    """
    - 구분자는 이미지를 입력으로 받고 그 이미지가 진짜일 확률을 0과 1사이의 숫자로 출력한다.
    - 생성자와 마찬가지로 4개의 선형레이어를 쌓았으며 레이어마다 활성 함수를 LeakyReLU를 넣어 준다.
    - 이미지크기를 입력받은 뒤 1024, 512, 256 점차 줄어 든다.
    - 마지막에는 확률값을 나타내는 수자 하나를 출력 한다.
    - 레이어마다 들어간 Dropout은 학습 시에는 무작위로 절반의 뉴런을 사용하지 않도록 한다.
    - 이를 통해 모델의 과적합이 되는 것을 방지 할 수 있다.
    - 또한 구분자가 생성자보다 지나치게 빨리 학습되는 것을 막을 수 있다.
    - 출력 값은 0과 1사이로 만들기 위해 활성화 함수로 Sigmoid를 넣었다.
    """
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.main = nn.Sequential(
            nn.Linear(in_features=image_size*image_size, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Dropout(inplace=True),
            nn.Sigmoid())

    def forward(self, inputs):
        inputs = inputs.view(-1, self.image_size*self.image_size)
        out = self.main(inputs)
        return out

def latent_vector(batch_size, dim):
    """
    - z벡터가 존재하는 공간을 잠재 공간(Latent Space)라고 한다.
    - 여기에서는 잠재 공간의 크기를 임의의 100차원으로 뒀다.
    - 잠재 공간의 크기에는 제한이 없으나 나타내려고하는 대상의 정보를 충분히 담을 수 있을 만큼은 커야 한다.
    """
    z = torch.randn(batch_size, dim).cuda()

    return z

def save(model, path, epoch):
    model_script = torch.jit.script(model)
    torch.jit.save(model_script, f"{path}\\model_{epoch}.pth")

def train(parameter):

    # Data Preprocessing
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    # Data Loader
    mnist = datasets.MNIST(root='D:\\workspace\\data\\', download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=parameter['batch_size'], shuffle=True)


    genterator_model = Generator(parameter['image_size'], parameter['latent_vector_dim'])
    discriminator_model = Discriminator(parameter['image_size'])
    genterator_model.cuda()
    discriminator_model.cuda()

    criterion = nn.BCELoss()
    """
    - 구분자의 출력 값은 이미지가 진짜일 확률이고, 이 확률이 얼마나 정답과 가까운지를 측정하기 위해 Binary Cross Entropy Loss Function을 사용한다.
    - 이 함수는 예측한 확률값이 정답에 가까우면 낮아지고 정답에 멀면 높아진다. (정답에 가까우면 오차를 작게, 정답과 멀면 오차를 크게 만든다.)
    """
    G_optimizer = torch.optim.Adam(genterator_model.parameters(), lr = parameter['lr'], betas = (0.5, 0.999))
    D_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr = parameter['lr'], betas = (0.5, 0.999))

    # Train
    for epoch in range(1, parameter['epoch']+1):

        D_losses = []
        G_losses = []

        for i, (real_data, _) in enumerate(dataloader, 1): # real_data 는 배치 크기 만큼 데이터가 담겨 있다.
            batch_size = real_data.size(0) # 0번째 크기는 전체 데이터 크기 이다. 즉, 배치 크기이다.

            real_data = real_data.cuda()

            # region Discriminator Train
            target_real = torch.ones(batch_size, 1).cuda() # 진짜 이미지 정답은 1
            D_result_from_real = discriminator_model(real_data) # 진짜 이미지를 구분자에 넣는다.
            D_loss_real = criterion(D_result_from_real, target_real) # real 데이터의 오차를 구한다.

            z = latent_vector(batch_size,parameter['latent_vector_dim'])
            target_fake = torch.zeros(batch_size, 1).cuda() # 가짜 이미지 정답은 0
            fake_data = genterator_model(z) # 노이즈 벡터로 가짜 이미지를 생성한다.
            D_result_from_fake = discriminator_model(fake_data) # fake_data를 Discriminator로 구분한다.
            D_loss_fake = criterion(D_result_from_fake, target_fake) # 출력값이 정답인 0에서 멀수록 loss가 높아 진다.

            D_loss = D_loss_real + D_loss_fake # 최종 Loss는 Real/Fake 데이터 Discriminator의 합이다.

            discriminator_model.zero_grad() # 구분자의 매개 변수의 미분갑을 0으로 초기화 한다.
            D_loss.backward() # 역전파를 통해 매개 변수의 loss에 대한 미분값을 계산한다.
            D_optimizer.step() # 최적화 기법을 통해 매개변수를 업데이트 한다.
            # endregion

            # region Generator Train
            z = latent_vector(batch_size, parameter['latent_vector_dim'])
            fake_data = genterator_model(z) # 노이즈 벡터로 가짜 이미지를 생성한다.
            D_result_from_fake = discriminator_model(fake_data) # fake_data를 Discriminator로 구분한다.
            G_loss = criterion(D_result_from_fake, target_real) # 생성자의 입장에서 구분자의 출력값이 1에 가까울 수록 loss가 높아 진다.
            genterator_model.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            # endregion

            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            if i%20 == 0:
                print(f"Epoch : {epoch} - ({i : 5d} /{int(len(mnist)/batch_size) : 5d}) D Loss = {sum(D_losses)/len(D_losses) : 0.4f} // D Loss = {sum(G_losses)/len(G_losses) : 0.4f}")

        if epoch % parameter['save_epoch'] == 0:
            save(genterator_model, parameter['save_base_path'], epoch)
            print("Save Success")

def inference(model_path, image_count):

    load_model = torch.load(model_path)

    z = torch.randn(image_count, 100).cuda()

    output = load_model(z)
    output = output.view(image_count, 28, 28).cpu().data.numpy()
    output = (255*(output - np.min(output))/np.ptp(output)).astype(np.uint8)  

    return output

if __name__ =="__main__":
    train(parameters)

    # image_count = 10
    # image_list = inference(r"D:\temp\save_model\GAN\model_100.pth", image_count)
    
    # for i in range(image_count):
    #     image = image_list[i]
    #     cv2.imwrite(f"D:\\temp\\inference\\GAN\\result_{i}.png", image)
