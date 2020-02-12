import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 3)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.relu(output)
        return output

path = "https://raw.githubusercontent.com/jokecamp/FootballData/master/football-data.co.uk/england/2014-2015/Premier.csv"

df = pd.read_csv(path)
colunas = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
scouts = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
results = ["win", "draw", "def"]
cdf = df[colunas]

ved = np.zeros([df.shape[0], 1])
df['win'] = ved
df['draw'] = ved
df['def'] = ved

df['win'] = df.apply(lambda x: 1 if x['FTHG'] > x['FTAG'] else 0, axis=1)
df['draw'] = df.apply(lambda x: 1 if x['FTHG'] == x['FTAG'] else 0, axis=1)
df['def'] = df.apply(lambda x: 1 if x['FTHG'] < x['FTAG'] else 0, axis=1)


entrada = df[scouts].copy()
saida = df[results].copy()

scouts100 = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC"]
scouts10 = ["HY", "AY", "HR", "AR"]

entrada[scouts100]/=100
entrada[scouts10]/=10

#separar em entrada de treinamento e entrada de testes
entradaTreinamento = entrada.iloc[:-20]
entradaTeste = entrada.iloc[-20:]

#separar em saida de treinamento e saida de testes
saidaTreinamento = saida.iloc[:-20]
saidaTeste = saida.iloc[-20:]

#converter para o tensor
training_input = torch.FloatTensor(entradaTreinamento.values)
training_output = torch.FloatTensor(saidaTreinamento.values)
test_input = torch.FloatTensor(entradaTeste.values)
test_output = torch.FloatTensor(saidaTeste.values)

#print(training_input)
#print(training_output)
#print(test_input)
#print(test_output)

inputSize = training_input.size()[1]
hiddenSize = 30

model = Net(inputSize, hiddenSize)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.3)

model.train()
epochs = 1000
erro = []
errosMSE = []
errosRMSE = []

model.eval()
y_pred = model(training_input)

loss = (y_pred - training_output)

loss1 = criterion(y_pred.squeeze(), training_output)

loss2 = torch.sqrt(loss1)

for epochs in range(epochs):
#while(loss.item() > min_erro):
    optimizer.zero_grad()

    #feed forward
    y_pred = model(training_input)
    #calc error
    loss = torch.abs(torch.mean(y_pred - training_output))
    loss1 = criterion(y_pred.squeeze(), training_output)
    loss2 = torch.sqrt(criterion(y_pred.squeeze(), training_output))
    #print("Erro: ", format(loss.item()))
    erro.append(loss)
    errosMSE.append(loss1.item())
    errosRMSE.append(loss2.item())

    #backpropagation
    loss1.backward()
   # loss2.backward()
    optimizer.step()

erro = np.array(erro)
y_pred = model(test_input)
n_digits = 3
y_pred = torch.round(y_pred * 10 ** n_digits) / (10**n_digits)

loss1 = criterion(y_pred.squeeze(), test_output)
loss2 = torch.sqrt(loss1)

print(y_pred.T)
print(test_output.T)
