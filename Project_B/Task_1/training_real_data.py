#import
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_network, get_dataset, Logger
from time import time 
from thop import profile
from MHIST import get_mhist_dataset
from sklearn.metrics import f1_score
import numpy as np
import sys

 ##### 
 
## Run parameters


dataset = 'MHIST'
batch_size = 32
  ## Batch size for evaluating whole dataset and synthetic dataset 
num_epochs = 20
lr = 0.01


out_log = 'results/%s_whole_dataset.txt'%(dataset) 
sys.stdout = Logger(out_log)

if dataset == 'MNIST':
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
                    get_dataset('MNIST', 'data')
    trainloader = DataLoader(dst_train, batch_size=batch_size, shuffle=True)      
elif dataset == 'MHIST':
    data_path = 'C:\\Users\Jackyy\OneDrive - University of Toronto\D\DIP\projectB_sh\DatasetCondensation-modified\mhist_dataset'
    ann_file = 'annotations.csv'
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_mhist_dataset(data_path, ann_file)
    
    trainloader = DataLoader(dst_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dst_test, batch_size=batch_size, shuffle=False)

#3. Initialize the Model, Loss Function, and Optimizer

if dataset == 'MNIST':
    model = get_network('ConvNet3', channel, num_classes, im_size=im_size)
elif dataset == 'MHIST':
    model = get_network('ConvNet7', channel, num_classes, im_size=im_size)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

#4. Define Cosine Annealing Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

#5. Training
start_time = time()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()  # Set the model to training mode

    pred_labels, true_labels = [], []
    for images, labels in trainloader:
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(images.to('cuda'))
        loss = criterion(outputs, labels.to('cuda'))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        if dataset == 'MNIST': # Calculate accuracy
            correct += (predicted.to('cpu') == labels).sum().item()
        elif dataset == 'MHIST':
            pred_labels.append(predicted.to('cpu').numpy())
            true_labels.append(labels.numpy())

    # Calculate and print average loss and accuracy for the epoch
    epoch_loss = running_loss / len(trainloader)
    
    # Calculate epoch accruacy for MNIST
    if dataset == 'MNIST':
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    elif dataset == 'MHIST':
        ## HP F1 Score
        hp_f1score = 100*f1_score(np.hstack(true_labels), np.hstack(pred_labels),
                                     pos_label = 0)
        ## SSA F1 Score
        ssa_f1score = 100*f1_score(np.hstack(true_labels), np.hstack(pred_labels),
                                     pos_label = 1)
        
        ## weighted F1 Score
        wf1score = 100*f1_score(np.hstack(true_labels), np.hstack(pred_labels),
                                     average = 'weighted')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, HP_F1Score: {hp_f1score:.2f}, SSA_F1Score: {ssa_f1score:.2f}, weighted_F1Score: {wf1score:.2f}%')
    
    scheduler.step()
end_time = time()
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")

    # Step the scheduler
   
# 6. Evaluate the Model and Calculate FLOPs
def calculate_flops(model, input_size):
    input = torch.randn(1, *input_size).to('cuda')
    flops, _ = profile(model, inputs=(input, ))
    return flops

# Calculate FLOPs
flops = calculate_flops(model, (channel, *im_size))  # MNIST images are 1x28x28

#7.Report Classification Accuracy and FLOPs
correct = 0
total = 0
with torch.no_grad():
    model.eval()
    pred_labels, true_labels = [], []
    for images, labels in testloader:
        outputs = model(images.to('cuda'))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if dataset == 'MNIST': # Calculate accuracy
            correct += (predicted.to('cpu') == labels).sum().item()
        elif dataset == 'MHIST':
            pred_labels.append(predicted.to('cpu').numpy())
            true_labels.append(labels.numpy())

if dataset == 'MNIST':
    test_accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {test_accuracy}%')
elif dataset == 'MHIST':
    ## HP F1 Score
    hp_f1score = 100*f1_score(np.hstack(true_labels), np.hstack(pred_labels),
                                 pos_label = 0)
    ## SSA F1 Score
    ssa_f1score = 100*f1_score(np.hstack(true_labels), np.hstack(pred_labels),
                                 pos_label = 1)
    
    ## weighted F1 Score
    wf1score = 100*f1_score(np.hstack(true_labels), np.hstack(pred_labels),
                                 average = 'weighted')
    
    print(f'Test Set: HP_F1score: {hp_f1score:.2f}%, SSA_F1score: {ssa_f1score:.2f}%, weighted_F1Score: {wf1score:.2f}%')
print(f'FLOPs for the model: {flops}')


sys.stdout.close()
