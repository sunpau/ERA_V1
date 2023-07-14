import torch
from utils import GetCorrectPredCount

def train_model(model, device, train_loader, optimizer,criterion,scheduler):
  model.train()

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()  # zero the gradients- not to use perious gradients

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()   #updates the parameter - gradient descent
    scheduler.step()
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

  train_acc = 100*correct/processed
  train_loss = train_loss/len(train_loader)
  return train_acc, train_loss
  

def test_model(model, device, test_loader, criterion):
    model.eval() #set model in test (inference) mode

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    return test_acc, test_loss

