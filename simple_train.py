import torch
import torch.nn as nn
from model import TemperatureClassifier
from safetensors.torch import save_file



    
def split_data(dictionary):
    x  = []
    y = []

    x = list(dictionary.keys())
    y = list(dictionary.values())

    return x, y

def load_tensor(x, y, device):
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
    y_tensor = torch.tensor(y, dtype=torch.int64).to(device)

    return x_tensor, y_tensor

def validation(model, x_val_ten, y_val_ten):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    count = 0
    avg_val_loss = 0

    with torch.no_grad():
        for i in range(len(x_val_ten)):
            x_sm = x_val_ten[i].unsqueeze(0)
            y_sm = y_val_ten[i].unsqueeze(0)
            out = model(x_sm)
            loss = loss_fn(out, y_sm)
            avg_val_loss += loss.item()
            count += 1

    avg_val_loss = avg_val_loss / count
    print(f'Avg val loss: {avg_val_loss}')


def learning(model, x_train_ten, y_train_ten, x_test_ten, y_test_ten, x_val_ten, y_val_ten, epoch=8):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
   

    count = 0
    avg_train_loss = 0
    avg_test_loss = 0

    for i in range(epoch):
        model.train()
        print(f'Эпоха: {i}')
        optimizer.zero_grad()
        for j in range(len(x_train_ten)):
            x_sample = x_train_ten[j].unsqueeze(0)
            y_sample = y_train_ten[j].unsqueeze(0)

            out = model(x_sample)
            loss = loss_fn(out, y_sample)
            avg_train_loss += loss.item()
            count += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = avg_train_loss / count
        count = 0
        print(f'Avg train loss: {avg_train_loss}')
        avg_train_loss = 0
        
        model.eval()
        avg_test_loss = 0
        with torch.no_grad():
            for j in range(len(x_test_ten)):
                x_sm = x_test_ten[j].unsqueeze(0)
                y_sm = y_test_ten[j].unsqueeze(0)
                out = model(x_sm)
                loss = loss_fn(out, y_sm)
                avg_test_loss += loss.item()
                count += 1
        
        avg_test_loss = avg_test_loss / count
        print(f'Avg test loss: {avg_test_loss}')
        count = 0
        avg_train_loss = 0
        avg_test_loss = 0

    
    validation(model, x_val_ten, y_val_ten)

    save_file(model.state_dict(), 'temperature_classifier.safetensors')
    print('Модель temperature_classifier.safetensors сохранена')






train_data = {
    # ===== КЛАСС 0: ВСЕ ОТРИЦАТЕЛЬНЫЕ ЧИСЛА =====
    -1: 0, -5: 0, -10: 0, -0.5: 0, -12.5: 0, -30.5: 0,
    -15.2: 0, -11: 0, -0.05: 0, -0.15: 0, -100: 0,
    -35: 0, -0.02: 0, -0.01: 0, -34.4: 0, -44.5: 0,
    -1000: 0, -1110: 0, -5.5: 0, -4.5: 0, -11.4: 0,
    -345: 0, -222: 0, -145: 0, -289678: 0, -93949585: 0,
    -3456666: 0, -0.0045: 0, -0.99: 0, -0.0000000006: 0, -0.00000003: 0, -0.000034:0,  -0.0000000455: 0, -0.0000005666: 0, -0.0000056054: 0, -0.00000455: 0, -0.0000045: 0, -0.00000000432:0, -0.007: 0, -0.0000056: 0, -0.00000644: 0, -0.0000005444: 0,
    # ===== КЛАСС 2: ВСЕ ПОЛОЖИТЕЛЬНЫЕ ЧИСЛА =====
    123: 1, 23.4: 1, 56.6: 1, 0.05: 1, 0.01: 1,
    1: 1, 156: 1, 1111: 1, 566: 1, 13.5: 1,
    89.3: 1, 894: 1, 990: 1, 34.5: 1, 12.4: 1,
    134: 1, 90: 1, 543: 1, 15.7: 1, 10.5: 1,
    777: 1, 855: 1, 989789: 1, 344.1: 1, 4545454545: 1,
    0.034: 1, 0.89: 1, 0.99: 1, 0.78: 1, 0.17: 1,
    0.1455: 1, 0.035: 1, 0.00000005: 1, 0.000003:1, 0.0000007: 1, 0.0000002: 1, 0.0000000677: 1, 0.000000000455: 1, 0.0000000006775: 1, 0.000004554: 1, 0.000001344: 1, 0.000000345: 1, 0.000000340: 1, 0.000554: 1, 0.00000005565: 1, 0.000000456: 1, 0.00000005434: 1
}

test_data = {
    # ===== КЛАСС 0 =====
    -111: 0, -3450: 0, -9: 0, -789: 0,
    -12000: 0, -8984: 0, -1030: 0, -256897: 0,
    -0.1: 0, -0.002: 0, -0.007: 0,-0.00000034:0,-0.0000345: 0, -0.000000056777: 0,
    

    # ===== КЛАСС 2 =====
    9999: 1, 1456: 1, 12000: 1, 11000: 1,
    1060: 1, 456: 1, 888: 1, 111: 1, 100940: 1, 0.0000000345: 1, 0.00000000456: 1, 0.00000000545: 1
}

validation_data = {
    # ===== КЛАСС 0 =====
    -999: 0, -123345: 0, -9999999: 0, -3: 0, -67: 0,
    -77: 0, -456789: 0, -77777: 0, -88.56: 0, -567.78: 0,
    -55.33: 0, -980.3: 0, -134.23: 0, -989456: 0, -345566: 0,
    -1.1235: 0, -10: 0, -0.01: 0, -0.22: 0, -0.12: 0, -0.155: 0, -0.0000004: 0, -0.000000023: 0, -0.000000000504: 0, -0.000000056: 0,
    
    
    # ===== КЛАСС 2 =====
    1956933: 1, 898422: 1, 76: 1, 555: 1,
    22222: 1, 123456: 1, 777777: 1, 1: 1,
    3: 1, 4: 1, 5: 1, 19: 1, 19.5: 1,
    0.11: 1, 0.21: 1, 0.34: 1, 0.45: 1,
    0.111: 1, 1.111: 1, 9999999: 1, 955665455: 1,
    2345677: 1, 765456: 1, 0.0000034: 1, 0.0000000006:1, 0.00000000355: 1, 0.00000000006504: 1, 0.0000003455: 1
}


device = ('cuda' if torch.cuda.is_available() else 'cpu')



x_train = []
y_train = []

x_test = []
y_test = []

x_val = []
y_val = []

mean = 0
std = 0


model = TemperatureClassifier(1, 100, 2).to(device)

x_train, y_train = split_data(train_data)

x_test, y_test = split_data(test_data)

x_val, y_val = split_data(validation_data)

x_train_ten, y_train_ten = load_tensor(x_train, y_train, device)

x_test_ten, y_test_ten = load_tensor(x_test, y_test, device)

x_val_ten, y_val_ten = load_tensor(x_val, y_val, device)


learning(model, x_train_ten, y_train_ten, x_test_ten, y_test_ten, x_val_ten, y_val_ten, epoch=4000)