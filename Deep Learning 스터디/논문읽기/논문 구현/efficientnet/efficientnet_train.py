import torch
import torchvision
import torchvision.transforms as transforms
import json
import time
from model.efficientnet import EfficientNet

# Parameters
paramters = {
    "epoch" : 50,
    "save_epoch" : 1,
    "batch_size" : 2,
    "lr" : 1e-3,
    "num_classes" : 10,
    "using_gpu" : False,
    "using_amp" : True,
    "model_save_path":"D:\\temp\\save_model\\GoogleNet\\"
}
def RecipeRun(parameter):
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Resize((512,512)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    label_info_json = {
        "label_count" : 10,
        "label_0" : {"code" : 5, "name" : "plane"},
        "label_1" : {"code" : 3, "name" : "automobile"},
        "label_2" : {"code" : 7, "name" : "bird"},
        "label_3" : {"code" : 9, "name" : "cat"},
        "label_4" : {"code" : 4, "name" : "deer"},
        "label_5" : {"code" : 1, "name" : "dog"},
        "label_6" : {"code" : 8, "name" : "frog"},
        "label_7" : {"code" : 6, "name" : "horse"},
        "label_8" : {"code" : 2, "name" : "ship"},
        "label_9" : {"code" : 0, "name" : "truck"}
        }
    
    extra_files = {
        'label_info': json.dumps(label_info_json)
    }

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=parameter['batch_size'],
                                            shuffle=True, num_workers=0, drop_last=True)

    model = EfficientNet.from_name('efficientnet-b0')

    if torch.cuda.is_available() and parameter['using_gpu']:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr = parameter['lr'])

    print("Train Start")
    for epoch in range(1, parameter['epoch']+1):
        epoch_loss = 0.0
        for i, batch in enumerate(trainloader, 1):
            iteration_start_time = time.time()
            inputs, labels = batch
            
            if torch.cuda.is_available() and parameter['using_gpu']:
                inputs = inputs.cuda()
                labels = labels.cuda()

            with torch.cuda.amp.autocast(enabled=paramters['using_amp']):
                output = model(inputs)
                loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iteration_elapsed_time = time.time() - iteration_start_time
            print(f"Epoch : {epoch} {i}/{int(len(trainset)/parameter['batch_size'])}, Loss : {loss}, Time : {iteration_elapsed_time}")

        print(f"Epoch : {epoch}, Loss : {epoch_loss / (len(trainset)/parameter['batch_size'])} ### ")

        if epoch % parameter['save_epoch'] == 0:
            with torch.no_grad():
                model_script = torch.jit.script(model)
                torch.jit.save(model_script, f"{parameter['model_save_path']}\\googlenet_{epoch}.pth", _extra_files = extra_files)


if __name__ =="__main__":
    RecipeRun(paramters)