import numpy as np
from numpy.core.numeric import outer
import pandas as pd
import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import models
from torchvision import transforms
from torchvision.datasets.utils import download_url


def batch_cnn(image_dir, image_list, label_list):
    ''''''
    # Load pre-trained ResNet18 model to device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = models.resnet18(pretrained=True)
    resnet.to(DEVICE)
    extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    extractor.to(DEVICE)
    # Get labels
    download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".", "imagenet_class_index.json")
    with open("imagenet_class_index.json", "r") as h:
        labels = json.load(h)
    # Image preprocessing
    preprocess = transforms.Compose([transforms.Resize(943),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    # Load images, make predictions
    result_arr = np.zeros((len(image_list), 4), dtype="object")
    for ii in tqdm(range(len(image_list)), total=len(image_list)):
        fname = image_list[ii]
        img_data = {}
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        img_tensor = torch.unsqueeze(preprocess(img), 0).to(DEVICE)
        with torch.no_grad():
            resnet.eval()
            extractor.eval()
            preds = resnet(img_tensor)
            feats = extractor(img_tensor)
        _, idx = torch.max(preds, 1)
        result_arr[ii, 0] = fname
        result_arr[ii, 1] = label_list[ii]
        result_arr[ii, 2] = labels[str(idx.item())][1]
        result_arr[ii, 3] = feats.flatten().detach().cpu().numpy()
    return result_arr

