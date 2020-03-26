# Local libraries

# Built-in libraries
import os
import time
from tqdm import tqdm
from copy import deepcopy

# External libraries
import torch
import matplotlib.pyplot as plt
import numpy as np



def train_model(model, data_loaders, criterion, optimizer, scheduler,
                num_epochs=25, model_save_path="",dataset_sizes=0,
               logger=None):
        
    # Setting torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Take the time
    since = time.time()

    # Copy model state and initialize accuracy value
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Initialize lists    
    acc_vec_train = list()  # average training loss of each epoch
    loss_vec_train = list()  # average training loss of each epoch
    acc_vec_val = list() # average test loss of each epoch
    loss_vec_val = list()  # average test loss of each epoch

    
    # Iterate over eache epoch
    for epoch in tqdm(range(num_epochs), disable=True):
        logger.info(f"Running epoch {epoch+1}/{num_epochs}")

        # Training and validation (finetune) per epoch
        for phase in ['train', 'finetune']:

            # Choosing between training and evaluating the model
            if phase == 'train':
                model.train()
                #lr_scheduler.step()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            
            # Iterate over data and send it to device
            for inputs, labels in tqdm(data_loaders[phase], desc=phase.capitalize()):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward propagation, track history if in training mode.
                with torch.set_grad_enabled(phase == 'train'):

                    # aux is an extra output for inception net i.e.
                    # the net outputs a tuple of 2 tensors!
                    try:
                        outputs, aux = model(inputs)
                    except:
                        outputs = model(inputs)

                    # ??????????????????????????????
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward and optimize in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # note loss.item() to get the python number instead
                # of loss.data[0] which now gives a tensor!
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
            
            
            # Getting epoch stats
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Printing epoch stats
            print(f"Loss: {epoch_loss:.4f} \nAccuracy: {epoch_acc:.4f}\n")

            # Add loss and accuracy depending on the phase
            if phase == 'train':
                loss_vec_train.append(epoch_loss)
                acc_vec_train.append(epoch_acc)

            if phase == 'finetune':
                loss_vec_val.append(epoch_loss)
                acc_vec_val.append(epoch_acc)

            # Deepcopy the model if the new model performs better by some margin
            if phase == 'finetune' and epoch >= 20 and epoch_acc - 0.005 >= best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)
                print(f"Epoch: {epoch} saving model to {model_save_path}")

            # Finetune and stop criteria
            if phase == 'finetune' and len(acc_vec_val) > 5:
                past_avg_val_acc = float(
                    sum(acc_vec_val[-5:]) / len(acc_vec_val[-5:]))
                if (epoch_acc - past_avg_val_acc) < 0.005:
                    print(f"Epoch {epoch}: val acc has not improved by 0.5% "
                          f"compared with average of last 5 epochs")

    # Get time for the whole training
    time_elapsed = time.time() - since
    logger.info(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logger.info(f"Best val Acc: {best_acc:4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)

    return model

def save_model_stats():
    # this function will take the statistics of the model and save them
    # TODO: this part of the function could be easily extracted out of this
    # part of the training and just outputting the stuff afterwards on main
    try:
        plot_prefix = os.path.splitext(model_save_path)[0]
        stat_file_name = plot_prefix + '_stat.txt'
        out_stat = np.array((loss_vec_train, loss_vec_val,
                             acc_vec_train, acc_vec_val))

        np.savetxt(stat_file_name, out_stat, delimiter='\t')
    except:
        print("Stat not saved")
        pass    

def stats_per_chunk(index, step):
    # Calculate statistics every 40 iterations
    if index % step == 0:
        print(f"Cross Entropy Loss: {loss.item() * 1000:.4f} "
        f"* 1e-03  [{index * 8}/"
        f"{len(data_loaders['train']) * 8} "
        f"({index/len(data_loaders['train'])*100:.0f}%)]")
        
        
def plot_model_stats():
    # Try to plot the model stats
    try:
        # Visualize loss change
        plot_prefix = os.path.splitext(model_save_path)[0]
        plt.plot(np.arange(1, epochs + 1), loss_vec_train, 'b')
        plt.plot(np.arange(1, epochs + 1), loss_vec_val, 'r')
        plt.title('Average loss of each epoch')
        loss_plot_name = plot_prefix + '_loss.png'
        plt.savefig(loss_plot_name, bbox_inches='tight')

        plt.plot(np.arange(1, epochs + 1), acc_vec_train, 'b')
        plt.plot(np.arange(1, epochs + 1), acc_vec_val, 'r')
        plt.title('Average accuracy of each epoch')
        acc_plot_name = plot_prefix + '_acc.png'
        plt.savefig(acc_plot_name, bbox_inches='tight')

    except:
        print("Figures not generated")
        pass
