import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import argparse
import copy

import sys



class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path












def test(dataset_name, model, device, test_loader, colour=0, colourmode=0):
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_score = []
    filepaths= []
    with torch.no_grad():  # replaced volatile=True in old version
        # for data, labels in test_loader:
        for data, labels, paths in test_loader:
            
            if colour > 0 and colour <= 3 and colourmode == 0:
                data = data[:,(colour-1),:,:]
                data = torch.unsqueeze(data,1)
                data = data.expand(-1, 3, -1, -1)
                
            if colour > 0 and colour <= 3 and colourmode == 1:
                for j in range(3):
                    if j != (colour-1):
                        data[:,j] = torch.zeros(data[:,j].size())
                
            
            data, labels = data.to(device), labels.to(device)  # replaced .cuda() in old version
            try:
                outputs, aux = model(data)
            except:
                outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().data)
            y_score.extend(predicted.cpu().data)
            filepaths.extend(paths)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    print('\n Accuracy of the inception_v3 on {}: {}/{} ({:.1f}%)\n'.format(
        dataset_name, correct, total,
        100. * correct / total))
    return y_true, y_score,filepaths


def bootstrap(y_tru, y_scr, n_bootstraps=2000, randomseed=245):
    """

    :param y_tru: true label stored as numpy array
    :param y_scr: predicted score stored as numpy array
    :param n_bootstraps:
    :param randomseed:
    :return:
    """

    # original_auc=roc_auc_score(y_tru, y_scr)
    if type(y_tru) != 'numpy.ndarray' or type(y_scr) != 'numpy.ndarray':
        try:
            y_tru, y_scr = np.array(y_tru), np.array(y_scr)
        except:
            print("unable to convert inputs to numpy arrays, check input")
            return
    total=len(y_tru)
    original_acc = (y_scr == y_tru).sum()/total
    print("Original Accuracy: {:0.1f}".format(original_acc*100))

    # n_bootstraps = 2000
    rng_seed = randomseed  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_scr) - 1, len(y_scr))
        # indices=np.random.choice( range(len(y_pred) - 1), len(y_pred), replace=True)  # this also with replacement
        if len(np.unique(y_tru[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        bootstrap_acc = ((y_tru[indices] == y_scr[indices]).sum())/total
        bootstrapped_scores.append(bootstrap_acc)
        # print("Bootstrap #{} ACC: {:0.2f}".format(i + 1, bootstrap_acc))

    # plot bootstrapping results

    plt.hist(bootstrapped_scores, bins=50)
    plt.title('Histogram of the bootstrapped accuracies on {} bootstrapping sample'.format(n_bootstraps))
    plt.savefig('acc_bootstrap_hist.png')

    # obtain the 95 % CI from the results
    sorted_accuracies = np.array(bootstrapped_scores)
    sorted_accuracies.sort()
    conf_low = sorted_accuracies[int(0.025 * len(sorted_accuracies))]
    conf_up = sorted_accuracies[int(0.975 * len(sorted_accuracies))]
    print('ACC with 95% confidence interval the ACC : {:.1f} ({:.1f} - {:.1f})'.format(original_acc*100, conf_low*100, conf_up*100))
    return original_acc, conf_low, conf_up


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="Path to model file")
    ap.add_argument("-i", "--img", required=True,
                    help="folder to test set images")
    ap.add_argument("-p", "--padding", required=True, default=False,
                    help="boolean value whether to pad (True) or crop (False) the image to square")
    ap.add_argument("-b", "--bts", required=False, default=80,
                    help="batch size")
    ap.add_argument("-n", "--name", required=False, default="norm",
                    help="name of the dataset")
    ap.add_argument("-c", "--crop", required=False, default=1536,
                    help="size after center crop")
    ap.add_argument("-ps", "--padSize", required=False, default=192,
                    help="padding size")
    ap.add_argument("-cl", "--colour", required=False, default=0,
                    help="Define the colour channel of the picture, 0=all, 1=red, 2=green, 3=blue")
    ap.add_argument("-cm", "--colourmode", required=False, default=0,
                    help="Mode in which single colour images are produced. 0=triplicate single channel, 1=one channel kept, others zeroed")


    args = vars(ap.parse_args())

    start_time = time.time()
    pad=args['padding']
    bat_size = int(args['bts'])
    name=args['name']
    crop_size=int(args['crop'])
    pad_size=int(args['padSize'])
    picture_colour=int(args['colour'])
    colour_mode=int(args['colourmode'])

    try:
        PATH = args['model']
    except:

        PATH = 'inceptionv3_raw_ukbb140219.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        test_data_dir = args['img']
    except:
        test_data_dir = 'data/raw/val'

    # process test set


    if pad==True:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Pad((0, pad_size)),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            'finetune': transforms.Compose([transforms.Pad((0, pad_size)), transforms.Resize((299, 299)),
                                            transforms.ToTensor()
                                            ]),
        }
    else:

        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            'finetune': transforms.Compose([transforms.CenterCrop(crop_size), transforms.Resize((299, 299)),
                                            transforms.ToTensor()
                                            ]),
        }



    # test_dataset = datasets.ImageFolder(test_data_dir, data_transforms['finetune'])
    test_dataset = ImageFolderWithPaths(test_data_dir, data_transforms['finetune'])


    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bat_size,
                                              shuffle=False, num_workers=4)

    model_ft = models.inception_v3(pretrained=False)

    class_names = ['female', 'male']

    num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(PATH, map_location=device))

    try:
        print('run script...')
        y_true, y_score ,filepaths= test('UK biobank test set with 34936 images', model_ft, device, test_loader,
                                         colour=picture_colour, colourmode=colour_mode)
        print('bootstrap...')
        bootstrap(y_true, y_score)
        with open('{}_true_label.txt'.format(name),'w+') as tt:
            for label in y_true:
                print(label,file=tt)
        with open('{}_predicted_label.txt'.format(name),'w+') as pp:
            for prediction in y_score:
                print(prediction,file=pp)
        with open('{}_file_paths.txt'.format(name),'w+') as fp:
            for path in filepaths:
                print(path,file=fp)

    except:

        print('debug test() or bootstrap()')
        pass

    run_time = time.time() - start_time

    print("Script run time: {:.0f}m {:.0f}s\n".format(run_time // 60, run_time % 60))
