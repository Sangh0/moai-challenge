import sys
"""
Your $PATH: anaconda3/envs/PATH
"""
__path__ = "Users/PATH"
sys.path.append(__path__)
import argparse
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from model import UNet
from util.dataset import CustomDataset


def test(model, data_loader, weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    output_list, path_list = [], []
    with torch.no_grad():
        start = time.time()
        for batch, (image, path) in enumerate(data_loader):
            image = image.to(device)
            
            output = model(image)
            output_list.append(output.detach().cpu())
            
            path_list.append(path)
            
        end = time.time()

    print(f'time: {end-start:.3f}s')
        
    output = torch.cat(output_list, dim=0)
    output = torch.argmax(output, dim=1)
    return output, path_list


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def runs(outputs):
    outputs = np.array(outputs)
    preds_string=[]
    for i in tqdm(range(0, len(outputs))):
        sample = outputs[i].copy()
        for label_code in [1,2,3]:
            tmp=[]
            for s in sample:
                s = np.equal(s, label_code).flatten()*1
                tmp+=s.tolist()
            enc = rle_to_string(rle_encode(np.array(tmp)))
    
            preds_string.append(enc)
    return preds_string


def read_csv(submission_dir):
    submission = pd.read_csv(submission_dir)
    return submission


def save_csv(submission, save_dir, rle_encoded):
    submission['EncodedPixels'] = rle_encoded
    submission.to_csv(save_dir, index=False)


def main():
    parser = argparse.ArgumentParser(description='Test performance of model and save submission')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory of dataset')
    parser.add_argument('--weight_dir', type=str, required=True,
                        help='directory of a weight file of trained model')
    parser.add_argument('--submission_dir', type=str, required=True,
                        help='directory of submission file with csv format')
    parser.add_argument('--submission_save_dir', type=str, required=True,
                        help='directory to store submission file')
    parser.add_argument('--num_filters', type=int, default=32,
                        help='the number of channels in U-Net')
    args = parser.parse_args()

    test_set = CustomDataset(
        path=args.data_dir, 
        subset='test', 
        transforms_=None, 
        crop_size=None,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    model = UNet(num_filters=args.num_filters)
    result = test(model, test_loader, args.weight_dir)

    pred_string = runs(result[0])
    submission = read_csv(args.submission_dir)
    save_csv(submission, args.submission_save_dir, pred_string)
    print(f'save! {args.submission_save_dir}')

if __name__ == '__main__':
    main()
