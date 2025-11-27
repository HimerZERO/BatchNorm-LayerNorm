import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms

from IPython.display import clear_output, display  # Для удобной работы в ipynb
import matplotlib.pyplot as plt
import numpy as np

import os
from torchvision import transforms, datasets

def get_dataloaders(dataset_name: str,
                    data_dir: str = "./data",
                    batch_size: int = None,
                    num_workers: int = 4,
                    pin_memory: bool = True,
                    validation_split: float = 0.1,
                    seed: int = 42):
    """
    Возвращает (train_loader, val_loader, test_loader, num_classes, image_size).
    Поддерживаемые dataset_name: 'CIFAR10', 'EMNIST', 'FashionMNIST', 'KMNIST', 'MNIST', 'STL10', 'SVHN'
    """
    # EMNIST не работает при указанной версии

    name = dataset_name.lower()
    # defaults
    if batch_size is None:
        batch_defaults = {
            'cifar10': 128,
            'svhn': 128,
            'stl10': 64,
            'mnist': 128,
            'fashionmnist': 128,
            'emnist': 128,
            'kmnist': 128
        }
        batch_size = batch_defaults.get(name, 64)

    # dataset-specific params: (torchvision class, kwargs, image_size, num_classes, mean, std, is_grayscale)
    dataset_map = {
        'cifar10': {
            'class': datasets.CIFAR10, 'root': data_dir, 'train_args': {'train':True},
            'image_size': 32, 'num_classes': 10,
            'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616],
            'is_grayscale': False
        },
        'svhn': {
            'class': datasets.SVHN, 'root': data_dir, 'train_args': {'split':'train'},
            'image_size': 32, 'num_classes': 10,
            'mean': [0.4377, 0.4438, 0.4728], 'std': [0.1980, 0.2010, 0.1970],
            'is_grayscale': False
        },
        'stl10': {
            'class': datasets.STL10, 'root': data_dir, 'train_args': {'split':'train'},
            'image_size': 96, 'num_classes': 10,
            'mean': [0.4467, 0.4398, 0.4066], 'std': [0.2241, 0.2215, 0.2239],
            'is_grayscale': False
        },
        # grayscale - будем конвертировать в RGB (3 канала) для совместимости с вашей Conv
        'mnist': {
            'class': datasets.MNIST, 'root': data_dir, 'train_args': {'train':True},
            'image_size': 28, 'num_classes': 10,
            'mean': [0.1307], 'std': [0.3081],
            'is_grayscale': True
        },
        'fashionmnist': {
            'class': datasets.FashionMNIST, 'root': data_dir, 'train_args': {'train':True},
            'image_size': 28, 'num_classes': 10,
            'mean': [0.2860], 'std': [0.3530],
            'is_grayscale': True
        },
        'emnist': {
            'class': datasets.EMNIST, 'root': data_dir, 'train_args': {'split':'balanced', 'train':True},
            'image_size': 28, 'num_classes': 47,
            'mean': [0.1307], 'std': [0.3081],
            'is_grayscale': True
        },
        'kmnist': {
            'class': datasets.KMNIST, 'root': data_dir, 'train_args': {'train':True},
            'image_size': 28, 'num_classes': 10,
            'mean': [0.1904], 'std': [0.3475],
            'is_grayscale': True
        },
    }

    key = name
    if key not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    info = dataset_map[key]
    image_size = info['image_size']
    num_classes = info['num_classes']
    mean = info['mean']
    std = info['std']
    is_gray = info['is_grayscale']


    if image_size < 32 and key in ('mnist','fashionmnist','emnist','kmnist'):
        train_resize = 32
    else:
        train_resize = image_size


    train_transform_list = []
    val_transform_list = []

    if is_gray:
        # Меняем на 3 канала
        train_transform_list.append(transforms.Lambda(lambda img: img.convert("RGB")))
        val_transform_list.append(transforms.Lambda(lambda img: img.convert("RGB")))

    if train_resize >= 64:
        train_transform_list += [
            transforms.RandomResizedCrop(train_resize, scale=(0.2,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    else:
        train_transform_list += [
            transforms.RandomCrop(train_resize, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]

    if len(mean) == 1:
        mean3 = [mean[0]] * 3
        std3 = [std[0]] * 3
    else:
        mean3 = mean
        std3 = std

    train_transform_list += [transforms.Normalize(mean=mean3, std=std3)]  # Нормализуем объекты
    train_transform = transforms.Compose(train_transform_list)


    val_transform_list += []
    if is_gray:
        val_transform_list.append(transforms.Lambda(lambda img: img.convert("RGB")))
    # Берём детерминированные трансформации
    if train_resize >= 64:
        val_transform_list += [
            transforms.Resize(int(train_resize * 1.15)),
            transforms.CenterCrop(train_resize),
            transforms.ToTensor()
        ]
    else:
        val_transform_list += [
            transforms.Resize(train_resize),
            transforms.ToTensor()
        ]
    val_transform_list += [transforms.Normalize(mean=mean3, std=std3)]
    val_transform = transforms.Compose(val_transform_list)


    # Загружаем датасет
    ds_class = info['class']
    root = os.path.join(info['root'], dataset_name)
    os.makedirs(root, exist_ok=True)

    if key == 'svhn':
        train_ds = ds_class(root=info['root'], split='train', transform=train_transform, download=True)
        test_ds = ds_class(root=info['root'], split='test', transform=val_transform, download=True)
        full_train_len = len(train_ds)
        val_len = int(full_train_len * validation_split)
        train_len = full_train_len - val_len
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    elif key == 'stl10':
        train_ds = ds_class(root=info['root'], split='train', transform=train_transform, download=True)
        test_ds = ds_class(root=info['root'], split='test', transform=val_transform, download=True)
        full_train_len = len(train_ds)
        val_len = int(full_train_len * validation_split)
        train_len = full_train_len - val_len
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    else:
        train_args = info.get('train_args', {})
        if key == 'emnist':
            train_ds = ds_class(root=info['root'], split=train_args.get('split','balanced'),
                                train=True, transform=train_transform, download=True)
            test_ds = ds_class(root=info['root'], split=train_args.get('split','balanced'),
                               train=False, transform=val_transform, download=True)
        else:
            train_ds = ds_class(root=info['root'], train=True, transform=train_transform, download=True)
            test_ds = ds_class(root=info['root'], train=False, transform=val_transform, download=True)
        full_train_len = len(train_ds)
        val_len = int(full_train_len * validation_split)
        train_len = full_train_len - val_len
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(seed))

    # Создаём загрузчики проверяя устройство
    device_has_cuda = torch.cuda.is_available()
    pin_memory_use = pin_memory and device_has_cuda

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=pin_memory_use)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=pin_memory_use)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=pin_memory_use)

    return train_loader, val_loader, test_loader, num_classes, train_resize, is_gray


def plot_live(step, train_losses, train_accuracies, val_losses, val_accuracies, model_name, name_db="default"):
    '''
    :param step: Номер эпохи
    :param model_name: Имя модели
    :param name_db: Название базы данных
    '''
    clear_output(wait=True)

    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.xlabel("Train")
    plt.ylabel("Loss")
    plt.plot(train_losses, label='Train Loss')
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.ylabel("Accuracy")
    plt.plot(train_accuracies, label='Train Accurac')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.xlabel("Val")
    plt.plot(val_losses, label='Val Loss')
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.plot(val_accuracies, label='Val Accurac')
    plt.grid()
    display(plt.gcf())

    if step == 100:  # TODO: Сделать гибче
        plt.savefig(f'graphs/{name_db}.{model_name}.png')
    plt.close()

def train(model, device, train_loader, val_loader, model_name, db_name, epochs=100):
    '''
    Docstring for train

    :param model: Модель
    :param device: Устройство
    :param model_name: Имя модели
    :param db_name: Название базы данных
    :param epochs: Количество эпох обучения
    '''

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch, (X, y) in enumerate(train_loader):
            if batch % 100 == 0:
                print(f"\n\tBatch {batch}")
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss += loss.item() * y.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)


        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                loss = criterion(pred, y)

                _, predicted = torch.max(pred.data, 1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()

                running_val_loss += loss.item() * y.size(0)

        avg_val_loss = running_val_loss / total_val
        val_accuracy = correct_val / total_val

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        if epoch + 1 % 10 == 0:  # Изменяемый параметр, сделить за обучением
            plot_live(epoch + 1, train_losses, train_accuracies, val_losses, val_accuracies, model_name, db_name)
        print(f"Its {model_name} Model on {db_name} Data Base")

    return train_losses, train_accuracies, val_losses, val_accuracies
