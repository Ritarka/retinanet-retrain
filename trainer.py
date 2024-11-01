import os
import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def evaluate_loss(model, criterion, dataloader, device):
    # Set model to eval mode
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Load inputs and labels to device
            inputs = inputs.to(device)
            labels = move_to_cuda(labels, device)  # Use the move_to_cuda function for labels
            
            # Call model and get outputs
            outputs = model(inputs, labels)
            
            # Calculate the loss using the loss function
            loss = outputs['bbox_regression']
            
            total_loss += loss.item()
            
    average_loss = total_loss / len(dataloader)
    return average_loss

def evaluate_accuracy(model, dataloader, device):
    # Set model to eval mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Load inputs and labels to device
            inputs = inputs.to(device)
            labels = move_to_cuda(labels, device)  # Use the move_to_cuda function for labels
            
            # Call model and get outputs
            outputs = model(inputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    return accuracy

def move_to_cuda(data, device):
    if isinstance(data, dict):
        return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_cuda(item, device) for item in data]
    else:
        return data.to(device)

def train(model, optimizer, criterion, trainloader, testloader, epochs, device):
    """
    Part 1.a: complete the training loop
    """
    train_losses = []  # For recording train losses
    test_losses = []  # For recording test losses
    
    # Move model to device here
    model.to(device)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    map = MeanAveragePrecision(box_format='xyxy', iou_type="bbox", extended_summary=True)

    for epoch in range(epochs):
        running_loss = 0.0
        # Set the model to train mode    
        model.train()
        num_iters = 0
        for i, data in enumerate(tqdm(trainloader)):
            inputs, labels = data

            # Load inputs and labels to device
            inputs = inputs.to(device, non_blocking=True)
            labels = [{'boxes': labels['boxes'][i], 'labels': labels['labels'][i]} for i in range(len(labels['boxes']))]
            labels = move_to_cuda(labels, device)
            
            outputs = model(inputs, labels)
            # Calculate the loss using the loss function
            loss = outputs['bbox_regression'] + outputs['classification']
            # print(outputs)
            # map.update(labels, outputs)
            # print(map.compute())
            
            # exit()
                                
            num_iters += 1
            
            if num_iters % 50 == 0:
                # Scale the loss and call backward on it
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}")

        # test_loss = evaluate_loss(model, criterion, testloader, device)
        # test_losses.append(test_loss)
        # print(f"Epoch {epoch+1}/{epochs} - Test Loss: {test_loss}")

        # test_accuracy = evaluate_accuracy(model, testloader, device)
        # print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_accuracy:.2f}%")
        
        # scheduler.step()

        if epoch % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': train_loss
            }
            torch.save(checkpoint, f"./checkpoints/model_checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch+1}")
            
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': train_loss
    }
    torch.save(checkpoint, f"./checkpoints/model_checkpoint_epoch_last.pth")
    print(f"Checkpoint saved for epoch last")


    return train_losses, test_losses
