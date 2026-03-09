# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.
<img width="1067" height="342" alt="image" src="https://github.com/user-attachments/assets/b18c9947-810f-423e-aadd-d7b0449dadc6" />


## DESIGN STEPS
### STEP 1: 

Import necessary libraries.

### STEP 2: 

Load and preprocess the data.

### STEP 3: 

Create input-output sequences.

### STEP 4: 

Convert data to PyTorch tensors.

### STEP 5: 

Define the RNN model.

### STEP 6: 

Train the model using the training data.

### STEP 7:

Evaluate the model and plot predictions.

## PROGRAM

### Name: BAVYA SRI B

### Register Number:212224230034

```python
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size = 64 , num_layers = 2, output_size = 1):
      super(RNNModel, self).__init__()
      self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
      self.fc = nn.Linear(hidden_size,output_size)
    def forward(self,x):
      out,_ =self.rnn(x)
      out=self.fc(out[:,-1,:])
      return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

```
```PYTHON

# Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
          x_batch,y_batch = x_batch.to(device), y_batch.to(device)
          optimizer.zero_grad()
          outputs = model(x_batch)
          loss = criterion(outputs, y_batch)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")

     # Plot training loss
    print('Name: BAVYA SRI B ')
    print('Register Number: 212224230034')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()   

train_model(model,train_loader,criterion,optimizer)
```

### OUTPUT

## Training Loss Over Epochs Plot

True Stock Price, Predicted Stock Price vs time

<img width="446" height="446" alt="image" src="https://github.com/user-attachments/assets/394a5058-4035-4e6b-818a-c9d9e4ae6f70" />


## True Stock Price, Predicted Stock Price vs time

<img width="785" height="620" alt="image" src="https://github.com/user-attachments/assets/a9d2c117-e243-4971-86b0-209e1121f83d" />


### Predictions
<img width="1061" height="722" alt="image" src="https://github.com/user-attachments/assets/ae221ba3-f891-4bad-8816-e477615eddcd" />


## RESULT
The RNN model successfully predicts future stock prices based on historical closing prices. The predicted prices closely follow the actual prices, demonstrating the model's ability to capture temporal patterns. The performance of the model is evaluated by comparing the predicted and actual prices through visual plots.
