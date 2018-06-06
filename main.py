import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchsample.transforms
import torchvision.models as models
from utilities import get_xviews


TOTAL_VIEWS = 16
TEST_MODE = False
DEBUG_MODE = False
epochs = 75
state_dict = None
opt_dict = None
body_zones_flipped = dict([(1,3), (2,4), (3,1), (4,2), (5,5), (6,7), (7,6), (8,10), (9,9), (10,8), (11,12), (12,11), (13,14), (14,13), (15,16), (16,15), (17,17)])


class MVCNN(nn.Module):
    def __init__(self, num_classes=17, pretrained=True):
        super(MVCNN, self).__init__()

        self.cnn = resnet152(pretrained=pretrained)
        #self.cnn = resnet50(pretrained=pretrained)
        #self.cnn = resnet101(pretrained=pretrained)
        #self.cnn = alexnet(pretrained=pretrained)
        #self.cnn = vgg19(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=(5, 5), stride=3, padding=2, dilation=1, groups=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.avgpool1 = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(input_size=2048 + 128 * 4 * 3 + 256 * 5 * 4 + 512 * 10 * 8, hidden_size=768, num_layers=1,
                            bias=True, batch_first=False, dropout=0, bidirectional=False)

        self.attention = nn.Linear(768, 16)
        self.softmax = nn.Softmax()

        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = []
        for i in range(x.size()[1]):
            view = x[:, i]
            features = self.cnn(view)

            avg_pool = self.avgpool1(features).view(features.size(0), -1)
            attention = self.cnn_attention(avg_pool).unsqueeze(2)

            features = torch.mul(features, attention.unsqueeze(3).expand_as(features))

            features = torch.cat((self.avgpool1(features).view(features.size(0), -1),
                                  self.conv1(features).view(features.size(0), -1),
                                  self.conv2(features).view(features.size(0), -1),
                                  self.conv3(features).view(features.size(0), -1)), 1)
            outputs.append(features)

        outputs = torch.stack(outputs, dim=0)
        outputs, _ = self.lstm(outputs)
        attn_weights = self.softmax(
            self.attention(outputs[-1])
        )

        outputs = outputs.permute(1, 0, 2)
        attn_weights = torch.unsqueeze(attn_weights, 1)
        outputs = torch.bmm(attn_weights, outputs)
        outputs = torch.squeeze(outputs, 1)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)

        return outputs


def resnet50(pretrained=True, **kwargs):
    if pretrained:
        model = models.resnet50(pretrained=True)
    return model


def resnet101(pretrained=True, **kwargs):
    if pretrained:
        model = models.resnet101(pretrained=True)
    return model


def resnet152(pretrained=True, **kwargs):
    if pretrained:
        model = models.resnet152(pretrained=True)
    return model


def alexnet(pretrained=True, **kwargs):
    if pretrained:
        model = models.alexnet(pretrained=True)
    return model


def vgg19(pretrained=True, **kwargs):
    if pretrained:
        model = models.vgg19(pretrained=True)
    return model


class AvrgMetr(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AugData(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor, names, train):
        assert data_tensor.size(0) == target_tensor.size(0) and type(names[0]) is str
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.names = names
        self.train = train

    def __getitem__(self, index):
        np.random.seed()
        name = self.names[index]
        data = self.data_tensor[index]
        target = self.target_tensor[index]
        if self.train:
            data, target = transform_sample(data, target)
        return data, target

    def __len__(self):
        return self.data_tensor.size(0)


def invert_target(target):
    inverted_target = torch.Tensor(target.shape)
    for k, v in body_zones_flipped.items():
        inverted_target[k-1] = target[v-1]
    return inverted_target


def transform_sample(im, target=None):
    if target is None:
        invert = False
    else:
        invert = np.random.randint(2)
        if invert:
            target = invert_target(target)

    im = im.numpy()
    if invert:
        im = np.flip(im, 3).copy()
        im = np.flip(im, 0).copy()
        im = np.roll(im, 1, 0)

    rand_int = np.random.randint(TOTAL_VIEWS)
    im = np.roll(im, rand_int, 0)
    im = torch.from_numpy(im)
    im = im.view(-1, im.size(2), im.size(3))

    random_affine = torchsample.transforms.RandomAffine(
        translation_range=[-0.01, 0.01], rotation_range=15, zoom_range=[0.95, 1.05], interp='nearest'
    )
    im = random_affine(im)
    im = im.view(16, 1, im.size(1), im.size(2))
    if target is None:
        return im
    else:
        return im, target


def create_onehot(x):
    onehot = np.zeros(17)
    onehot[x-1] = 1
    return onehot


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    losses = AvrgMetr()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        input = input.repeat(1, 1, 3, 1, 1)

        if DEBUG_MODE and False:
            print(list(zip(target[0].cpu().tolist(), [x+1 for x in range(17)])))
            print(list(zip(target[1].cpu().tolist(), [x+1 for x in range(17)])))
            for j in range(TOTAL_VIEWS):
                plt.imshow(input.cpu().numpy()[0, j, 0])
                plt.show()

        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        output = model(input)
        loss = criterion(output, target)

        losses.update(loss.data[0], input.size(0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del input, target, output, loss

    scheduler.step()
    loss_tracker_train.append(losses.avg)

def validate(val_loader, model, criterion):
    losses = AvrgMetr()
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input = input.repeat(1, 1, 3, 1, 1)

        if DEBUG_MODE and False:
            print(list(zip(target[0].cpu().tolist(), [x+1 for x in range(17)])))
            print(list(zip(target[1].cpu().tolist(), [x+1 for x in range(17)])))
            for j in range(TOTAL_VIEWS):
                plt.imshow(input.cpu().numpy()[0, j, 0])
                plt.show()

        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        output = model(input)
        loss = criterion(output, target)

        losses.update(loss.data[0], input.size(0))
        
        del input, target, output, loss

    loss_tracker_val.append(losses.avg)


def name_to_array(name, directory):
    ext = "aps"
    arr = np.array(get_xviews("{}/{}.{}".format(directory, name, ext), x=TOTAL_VIEWS))
    arr = np.expand_dims(arr, 1)
    arr = np.pad(arr, ((0,0), (0,0), (0, 1), (0, 0)), mode="constant", constant_values=0)
    return arr


def predict(model, name):
    input = name_to_array(name, "test")
    input = np.expand_dims(input, 0)
    input = torch.Tensor(input)

    if DEBUG_MODE and False:
        for j in range(TOTAL_VIEWS):
            plt.imshow(input.numpy()[0, j, 0, :, :])
            plt.show()

    input = input.repeat(1,1,3,1,1)
    in_var = torch.autograd.Variable(input.cuda(), requires_grad=False)
    accum = None

    if type(model) != list:
        model = list(model)
    for m in model:
        output = torch.nn.Sigmoid()(m(in_var))
        output = output.data.cpu().numpy()[0]
        if accum is None:
            accum = output
        else:
            accum += output

    return accum / len(model)

def test_model(model, base_dir=None, epoch=0):
    time_str = str(int(time.time()))[2:]
    if base_dir == None:
        base_dir = "predictions/{}".format(time_str)
        os.mkdir(base_dir)
    outfile = open('{}/predictions_{}_{}.csv'.format(base_dir, time_str, epoch), 'w')
    print('Id,Probability', file=outfile)
    test_names = set([filename.split('.')[0] for filename in os.listdir('test/')])
    for name in test_names:
        print(name)
        for bodypart, prob in enumerate(predict(model, name)):
            print("{}_Zone{},{}".format(name, bodypart + 1, prob), file=outfile)





if TEST_MODE:
    models = []
    for name in state_dict:
        model = MVCNN(17, pretrained=True).cuda()
        model.load_state_dict(torch.load(name))
        model.eval()
        models.append(model)
        print("Added {}".format(name))
    test_model(models)
    exit()

model = MVCNN(17, pretrained=True).cuda()

train_file = open('stage1_labels.csv')
train_file.readline()
name_to_vector = {}
for line in train_file:
    name_zone, label = line.strip().split(',')
    name, zone = name_zone.split('_')
    zone_int = int(zone[4:])
    one_hot = create_onehot(zone_int)
    if name not in name_to_vector:
        name_to_vector[name] = np.zeros(17)
    if int(label) == 1:
        name_to_vector[name] += one_hot

sample_count = len(name_to_vector)
print(sample_count)

names = [None] * sample_count
train_in = np.empty((sample_count, TOTAL_VIEWS, 1, 660 + 1, 512 + 0), dtype=np.float32)
train_out = np.empty((sample_count, 17))
for i, (name, one_hot) in enumerate(name_to_vector.items()):
    in_tensor = name_to_array(name, "aps")
    train_in[i] = in_tensor
    train_out[i] = one_hot
    names[i] = name

train_split = int(len(train_in)) - 5
train_in, valid_in = train_in[0:train_split], train_in[train_split:]
train_out, valid_out = train_out[0:train_split], train_out[train_split:]

train_in = torch.Tensor(train_in)
train_out = torch.Tensor(train_out)
valid_in = torch.Tensor(valid_in)
valid_out = torch.Tensor(valid_out)

dataset = AugData(train_in, train_out, names, train=True)
valid_dataset = AugData(valid_in, valid_out, names, train=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, sampler=None, batch_sampler=None, num_workers=8, pin_memory=True, drop_last=True)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, pin_memory=True, drop_last=False)
crit = torch.nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4, last_epoch=-1)

if state_dict:
    model.load_state_dict(torch.load(state_dict))
    if opt_dict:
        optimizer.load_state_dict(torch.load(opt_dict))

torch.backends.cudnn.benchmark = False

time_str = str(int(time.time()))[2::]
base_dir = "predictions/{}".format(time_str)
loss_tracker_train = []
loss_tracker_val = []
best_loss = 0.010
this_loss = 1.0

for epoch in range(epochs):
    train(data_loader, model, crit, optimizer, scheduler, epoch)
    validate(valid_data_loader, model, crit)

    if epoch and epoch % 25 == 0:
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        torch.save(model.state_dict(), "{}/model_{}.torch".format(base_dir, epoch))
        torch.save(optimizer.state_dict(), "{}/opt_{}.torch".format(base_dir, epoch))

        plt.clf()
        plt.plot(loss_tracker_train[1:], label="Training loss")
        plt.plot(loss_tracker_val[1:], label="Validation loss")
        plt.legend(loc="upper left")
        plt.savefig("{}/predictions_{}.png".format(base_dir, epoch))

        test_model(model, base_dir, epoch)

    this_loss = loss_tracker_val[-1]
    print("Current loss: {}".format(this_loss))
    print("Best loss: {}".format(best_loss))

    if this_loss < best_loss + 0.0020:
        print("Found better model with {} loss (old loss was {})".format(this_loss, best_loss))
        best_loss = min(this_loss, best_loss)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        torch.save(model.state_dict(), "{}/best_model_{}_{:.4f}.torch".format(base_dir, epoch, this_loss))
