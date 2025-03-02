import os
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch.optim as opt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from modules.nice import NICE
from modules.coupling import intercalate

#--------------------------------------------------------------
# Training Configuration
#--------------------------------------------------------------
MODEL_PATH: str = "./models/"
DATA_PATH : str = "../data"
SAVE_EPOCH: int = 5
START: int      = 0

EPOCHS: int = 50
BATCH_SIZE: int = 256

LR: float = 0.001

DEVICE: torch.DeviceObjType = torch.device('cpu')

#--------------------------------------------------------------
# Data Loading
#--------------------------------------------------------------
def rescale(tensor: torch.Tensor):
    ma, mi = tensor.max(), tensor.min()
    return (tensor - mi) / (ma - mi)

def add_noise(tensor: torch.Tensor, strengh: float = 1):
    noise = torch.distributions.Uniform(0., 1.).sample(tensor.shape)
    return (tensor * 255. + strengh * noise) / (255 + strengh)

transform = [
    transforms.ToTensor(), # create tensor from PIL Image [0, 1]
    transforms.Lambda(lambda x: x.view(-1)),   # flatten image (the initial model is an MLP)
    transforms.Lambda(lambda x: add_noise(x)), # add noise (it is not very perceptible to the idea, maybe increase strength?)
    transforms.Lambda(lambda x: rescale(x))    # rescale between  0 and 1
]
transform = transforms.Compose(transform)

trainset = datasets.MNIST(DATA_PATH, train =  True, transform=transform, download=True)
testset  = datasets.MNIST(DATA_PATH, train = False, transform=transform, download=True)

# create validation dataset
train_size, valid_size = int(len(trainset) * 0.85), int(len(trainset) * 0.15)
trainset, validset = torch.utils.data.random_split(trainset, [train_size, valid_size])

# create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset , batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

#--------------------------------------------------------------
# Create Model
#--------------------------------------------------------------
model = NICE(28 * 28, 1000, 5, 4)
# model.load_state_dict(torch.load('./models/nice-020.pt', weights_only=True))
model = model.to(DEVICE)

#--------------------------------------------------------------
# Training Procedure
#--------------------------------------------------------------
optimizer = opt.Adam(model.parameters(), LR)

for e in range(START, EPOCHS + 1):
    train_log: float = 0
    valid_log: float = 0

    model.train()
    # Training Phase
    for image, _ in tqdm(trainloader, total = len(trainloader), desc = "training phase", leave = False):
        optimizer.zero_grad()

        # add noise to image
        # image = image + torch.rand_like(image) / 256.
        # image = image.clamp(0, 1)
        image = image.to(DEVICE) 

        _, log = model(image)

        log = - log.mean()
        log.backward(); optimizer.step()
        train_log -= log.detach().item()

    model.eval()
    # Validation Phase
    with torch.no_grad():
        for image, _ in tqdm(validloader, total = len(validloader), desc = "validation phase", leave = False):

            # add noise to image
            # image = image + torch.rand_like(image) / 256.
            # image = image.clamp(0, 1)
            image = image.to(DEVICE) 

            _, log = model(image)

            log = - log.mean()
            valid_log -= log.detach().item()

    print(f"[epoch {e:>03}] train log-like: {train_log / len(trainloader):15.3f}  valid log-like: {valid_log / len(validloader):15.3f}", end = "")
    if e % SAVE_EPOCH == 0:
        save_path = os.path.join(MODEL_PATH, f"nice-{e:03d}.pt")
        torch.save(model.state_dict(), save_path)
        print(f" save model: {save_path}", end = "")
    print()

save_path = os.path.join(MODEL_PATH, f"nice-{e:03d}.pt")
torch.save(model.state_dict(), save_path)
print("final model path:", save_path)
