import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
print("pytorch version:",torch.__version__)

USE_GPU = True
# torch.cuda.set_device(6)
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    print('using device:', device, torch.cuda.current_device())
else:
    device = torch.device('cpu')
   
# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 50000
learning_rate = 0.001
lr = 0.001
# Toy dataset
x_train = np.array([[2], [3], [4]], dtype=np.float32)

y_train = np.array([[-152.061930066], [-152.122363082], [-152.137037926]], dtype=np.float32)


b1 = torch.randn(1, requires_grad = True)
w1 = torch.randn(1, requires_grad = True)
w2 = torch.randn(1, requires_grad = True)

# Linear regression model
# model = chmexp(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    # print(model.weight, model.bias)
    
    # Forward pass
    outputs = b1 + w1 * torch.exp(-w2 * inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    loss.backward()
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        b1 -= lr * b1.grad
        w1.grad.zero_()
        w2.grad.zero_()
        b1.grad.zero_()
    
    if (epoch+1) % 1000 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

print("b1:", b1, "w1:", w1, "w2:", w2)

w1 = w1.detach().numpy()
w2 = w2.detach().numpy()
b1 = b1.detach().numpy()

# Plot the graph
predicted = b1 + w1 * np.exp(-w2 * x_train) 
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, 'bo', label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
