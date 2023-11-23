import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import r2_score
import pickle
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def data_load(sname = "AAL",file = "all_stocks_5yr.csv",index_start = 150,index_end = 780,plot = True ):
    data_AAL = pd.read_csv(file)
    data_AAL.columns = data_AAL.columns.str.capitalize()
    data_AAL.set_index("Date", inplace=True)
    data_AAL.index = pd.to_datetime(data_AAL.index)

    data_AAL = data_AAL.iloc[index_start:index_end,:]

    if plot == True:
        data_AAL['Open'].plot(label = "open")
        data_AAL['Close'].plot(label = "Close")
        data_AAL['High'].plot(label = "High")
        data_AAL['Low'].plot(label = "Low")
        plt.ylabel("stock_price")
        plt.title(f"{sname}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        data_AAL["Volume"].plot()
        plt.ylabel("Volume")
        plt.title(f"{sname}")
        plt.xticks(rotation=45)
        plt.show()
    return data_AAL

# Feature Engineering
def feature_engineering(data,window = [5,10,20], d_window = 3,plot = True):
    sel_col = ['Open', 'High', 'Low', 'Close','Volume']
    data = data[sel_col]
    # Moving Average SMA = (Close_1 + Close_2 + ... + Close_n) / n

    window_size = window  # windows size

    for i in window_size:
        data[f'SMA_{i}'] = data['Close'].rolling(window=i).mean()


    # Relative Strength Indicator (RSI)
    # RS = (Avg Gain) / (Avg Loss)
    # RSI = 100 - (100 / (1 + RS))
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    for i in window_size:
        avg_gain = gain.rolling(window=i).mean()
        avg_loss = loss.rolling(window=i).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data[f'RSI_{i}'] = rsi


    # stochastic oscillation index (SOI)
    k_window = window_size  # %K 窗口大小
    d_window = d_window  # %D 窗口大小

    for i in k_window:
        lowest_low = data['Low'].rolling(window=i).min()
        highest_high = data['High'].rolling(window=i).max()

        k = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
        d = k.rolling(window=d_window).mean() # d is the moving average for k, windows was set by fiexed number, 5 or 3

        data[f'K_{i}'] = k
        data[f'D_{i}'] = d

    data["P_Diff"] = data["Close"] - data["Open"]
    data["P_Range"] = data["High"] - data["Low"]

    data["ratio"] = (data["P_Diff"]/data["Open"])

    # volume change
    data['Vol_Change'] = data['Volume'].pct_change() * 100

    for i in window_size:
        data[f'Vol_MA_{i}'] = data['Volume'].rolling(window=i).mean()
        
    if plot == True:
        sel_col = ["SMA_5","SMA_10","SMA_20"]#,'P_Diff','P_Range','Vol_Change']
        data[sel_col].plot()
        plt.ylabel("Stock Close Price")
        plt.title("(a)")
        plt.xticks(rotation=45)
        plt.show()

        sel_col = ['P_Diff','P_Range']
        data[sel_col].plot()
        plt.ylabel("Price Difference")
        plt.title("(b)")
        plt.xticks(rotation=45)
        plt.show()

        sel_col = ['RSI_5','RSI_10','RSI_20','K_5','D_5','K_10','D_10','K_20','D_20']#,'P_Diff','P_Range','Vol_Change']
        data[sel_col].plot()
        plt.ylabel("Value")
        plt.title("(c)")
        plt.xticks(rotation=45)
        plt.show()

        sel_col = ['Vol_MA_5','Vol_MA_10','Vol_MA_20']
        data[sel_col].plot()
        plt.ylabel("Volume")
        plt.title("d")
        plt.xticks(rotation=45)
        plt.show()

        data[["ratio"]].plot()
        plt.ylabel("ratio")
        plt.title("AAL")
        plt.xticks(rotation=45)
        plt.show()

        # columns_to_drop = ["Open", "High", "Low", "Close", "Volume"]
        # data = data.drop(columns=columns_to_drop)
        correlation_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.show()
    return data


# build target variable and normalization for features
def Minmax(data):
    scalers = {} #creat dictionary to store variable
    data['target'] = data['ratio'].shift(-1) 
    data = data.dropna() 
    print("data size = ",data.shape)
    # Loop through each column and scale the data
    for col in data.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        scalers[col] = scaler
            
    df_main = data.astype(np.float32)  
    
    # df_main["ratio"][401:451].plot(label = "ratio")
    # df_main["target"][400:450].plot(label = "target")
    # plt.title('normalization of ratio and target')
    # plt.legend()
    # plt.show()
    
    # a = scalers["ratio"].inverse_transform(df_main["ratio"][401:451].values.reshape(-1, 1))
    # b = scalers["target"].inverse_transform(df_main["target"][400:450].values.reshape(-1, 1))
    # plt.plot(a,label = "ratio")
    # plt.plot(b,label = "target")
    # plt.title('original of ratio and target')
    # plt.legend()
    # plt.show()
    return df_main, scalers


def PCA_plus(data,n_components=10):

    X = data.drop('target', axis=1)  # feature variable
    y = data['target']  # target variable

    # # standardisation, df_main has been normalization, so in here we don't further standardlization
    # scaler_pca = StandardScaler()
    # X_scaled = scaler_pca.fit_transform(X)
    pca = PCA(n_components=n_components)  # set the numbder of principal components
    X_pca = pca.fit_transform(X)

    # check the proportion of principal components
    explained_variance = pca.explained_variance_ratio_
    print("Proportion of principal components:", explained_variance)

    # cumulative variance
    cumulative_variance = np.cumsum(explained_variance)

    # print pca information
    for i, ev in enumerate(explained_variance):
        print(f"principal components {i + 1}: Proportion of variance = {ev:.4f}, Cumulative proportion of variance = {cumulative_variance[i]:.4f}")

    # print(X_pca.shape)

    y = y.to_numpy().reshape(-1, 1)

    PCA_df = np.concatenate((X_pca,y), axis=1)
    columns = [f'PCA{i+1}' for i in range(n_components)] + ['target']
    PCA_out = pd.DataFrame(PCA_df, columns=columns)
    print("PCA size = ",PCA_df.shape)
    print(PCA_out.head(5))
    return PCA_out


def train_test_set(data,input_dim = 10,seq = 10,test_set_size = 30):
    data_feat, data_target = [],[]
    df_main = data
    for index in range(df_main.shape[0]-1,seq-1,-1):
        data_feat.append(df_main.drop(columns=['target'])[index-seq: index])
        data_target.append(df_main['target'][index-seq: index])

    data_feat.reverse()
    data_target.reverse()

    data_feat = np.array(data_feat)
    data_target = np.array(data_target)

    print("data_feat shape:",data_feat.shape)
    print("data_target shape",data_target.shape)


    # predict stock price within 30 trading day
    #test_set_size = int(np.round(0.1*df_main.shape[0]))  # np.round(1)
    train_size = data_feat.shape[0] - (test_set_size) 
    print("testset length",test_set_size)  
    print("trainset length",train_size)     

    trainX = torch.from_numpy(data_feat[:train_size].reshape(-1,seq,input_dim)).type(torch.Tensor) #bitch_size*sqe*input_feature  
    testX  = torch.from_numpy(data_feat[train_size:].reshape(-1,seq,input_dim)).type(torch.Tensor) #bitch_size*sqe*input_feature 
    trainY = torch.from_numpy(data_target[:train_size].reshape(-1,seq,1)).type(torch.Tensor) #bitch_size*sqe*1
    testY  = torch.from_numpy(data_target[train_size:].reshape(-1,seq,1)).type(torch.Tensor) #bitch_size*sqe*1
    #538*10*1
    print("trainX size:",trainX.shape) 
    print("testX size:",testX.shape) 
    print("trainY size:",trainY.shape) 
    print("testY size:",testY.shape)   

    # #we can use this to set any size of batch-size， this project put all together
    # batch_size=2395
    # train = torch.utils.data.TensorDataset(trainX,trainY)
    # test = torch.utils.data.TensorDataset(testX,testY)
    # train_loader = torch.utils.data.DataLoader(dataset=train, 
    #                                            batch_size=batch_size, 
    #                                            shuffle=False)

    # test_loader = torch.utils.data.DataLoader(dataset=test, 
    #                                           batch_size=batch_size, 
    #                                           shuffle=False)
    return trainX, testX, trainY, testY

class LSTM_GRU_RNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim,bidirectional=False,model = "LSTM"):
            super(LSTM_GRU_RNN, self).__init__()
            self.model = model
            # Hidden dimensions
            self.hidden_dim = hidden_dim
            self.bidirectional = bidirectional
            # Number of hidden layers
            self.num_layers = num_layers
            self.dropout = nn.Dropout(p=0.5)  # Dropout layer，p was drop probability,
            self.relu = nn.ReLU()  # Add a ReLU activation
            # Building your LSTM
            # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,bidirectional=bidirectional)
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

            # fc layer after LSTM，this project is a regression problem，
            # so it cann't add activation function after fc2, but you can add it after fc1
            if self.bidirectional == False:
                self.fc1 = nn.Linear(hidden_dim, hidden_dim//2) 
            else:
                self.fc1 = nn.Linear(hidden_dim*2, hidden_dim//2) 
                
            self.fc2 = nn.Linear( hidden_dim//2, output_dim) 
        def forward(self, x):
            if self.bidirectional == False:
                # Initialize hidden state with zeros   
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_() 
                # x.size(0) is batch_size
                # Initialize cell state
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            else:
                h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).requires_grad_() # Multiply num_layers by 2 for bidirectional
                c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

            # One time step
            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            # GRU don't need cell state
            if self.model == "LSTM":
                out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            elif self.model == "GRU":
                out, hn = self.gru(x, h0.detach())
            elif self.model == "RNN":
                out, hn = self.rnn(x, h0.detach())
            else:
                # Handle unknown model name
                raise ValueError("Unknown model name" )
            out = self.dropout(out)
            out = self.fc1(out) 
            out = self.relu(out)  # Apply ReLU activation
            out = self.fc2(out) 
            return out

def run_train_test(trainX,trainY,testX,
              input_dim = 10, hidden_dim = 34, num_layers = 2, 
              output_dim = 1,seq = 10,num_epochs = 1000,means = 5,
              model_name = "LSTM",bidirectional=True,name="AAL"):
    

    train_pred_lst = []
    test_pred_lst=[]
    train_loss_lst=[]
    test_loss_lst=[]
    train_mae_lst=[]
    test_mae_lst=[]
    train_r2_lst=[]
    test_r2_lst = []
    
    # store multipal model parameter
    model_parameters = []

    for i in range(means):
        model = LSTM_GRU_RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                        bidirectional=bidirectional, model = model_name)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0) 
        loss_fn = torch.nn.MSELoss(reduction='mean')           
        model.train()
        # hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            # model.hidden = model.init_hidden()
            # Forward pass
            y_train_pred_inter = model(trainX) # y_train_pred_inter and trainY were bitch_size*sqe*1

            loss = loss_fn(y_train_pred_inter, trainY)
            if t % 20 == 0 and t !=0:               
                print("Epoch ", t, "MSE: ", loss.item())
            # hist[t] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()
            
            # Update parameters
            optimiser.step()
        
        
        model_parameters.append(model.state_dict())
        # Set the model to evaluation mode
        model.eval()
        # Train Predict Value
        # Predict the values using the model on the training data
        y_train_pred = model(trainX)
        print("y_train_pred size=",y_train_pred.shape )  #  538*10*1
        train_pred = y_train_pred.detach().numpy()[:,-1,0]  # Extract the final predicted values
        train_pred_lst.append(train_pred)                   # 每个长度的最后一个值就是估算的结果，共计batch_size个

        y_test_pred = model(testX)
        test_pred = y_test_pred.detach().numpy()[:,-1,0]  # Extract the final predicted values
        test_pred_lst.append(test_pred)
    # 计算模型参数的平均值
    # average_model_parameters = {key: torch.stack([param[key] for param in model_parameters]).mean(dim=0) 
    #                          for key in model_parameters[0]}

    # # 创建一个新的模型实例
    # average_model = LSTM_GRU_RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
    #                             num_layers=num_layers, bidirectional=bidirectional, model=model_name)

    # # 加载平均模型参数
    # average_model.load_state_dict(average_model_parameters)

    # # 保存平均模型参数
    # torch.save(average_model.state_dict(), 'average_model_parameters.pth')

    with open(f'{name}_{model_name}_train_pred_ind{input_dim}_hd{hidden_dim}_nl{num_layers}_outd{output_dim}_seq{seq}_ne{num_epochs}_means{means}.pkl', 'wb') as f:
        pickle.dump(train_pred_lst, f)
    with open(f'{name}_{model_name}_test_pred_ind{input_dim}_hd{hidden_dim}_nl{num_layers}_outd{output_dim}_seq{seq}_ne{num_epochs}_means{means}.pkl', 'wb') as f:
        pickle.dump(test_pred_lst, f)
   
    torch.save(model.state_dict(), 'model_w_parameters.pth')

    return train_pred_lst,test_pred_lst

def train_matrics(train_pred_file,data,trainY,seq,scalers,test_set_size = 30,plot=True):
    with open(train_pred_file, 'rb') as f:
        train_pred_lst = pickle.load(f)
    train_true_value = trainY.detach().numpy()[:,-1,0] # train_true_value is target value
                                                       # it ahead of ratio one step
    train_mean_pred = np.mean(train_pred_lst,axis = 0) # mean of all the prediction
    
    if plot == True:
        plt.plot(train_mean_pred, label="train_Preds_mean") 
        plt.plot(train_true_value, label="train_Real_Data")    # true price change rate
        plt.legend()
        plt.title("Real and Predicted Percentage Price Change in Trainset ")
        plt.ylabel("Ratio (Normolization)")
        plt.xlabel("Trading Day")
        plt.show()

        plt.plot(train_mean_pred[400:-1], label="train_Preds_mean") 
        plt.plot(train_true_value[400:-1], label="train_Real_Data")
        plt.title("Real and Predicted Percentage Price Change in Trainset (Part) ")
        plt.ylabel("Ratio (Normolization)")
        plt.xlabel("Trading Day")    
        plt.legend()
        plt.show()
        
    train_target_inverse = scalers["target"].inverse_transform(train_mean_pred.reshape(-1, 1))
    train_true_target_inverse = scalers["target"].inverse_transform(train_true_value.reshape(-1, 1))
   
    if plot == True:
        plt.plot(train_target_inverse, label="train_target_inverse") 
        plt.plot(train_true_target_inverse, label="train_true_target_inverse") 
        plt.title("Real and Predicted Percentage Price Change in Trainset")
        plt.ylabel("Ratio (De-Normolization)")
        plt.xlabel("Trading Day")    
        plt.legend()
        plt.show()

        plt.plot(train_target_inverse[400:-1], label="train_target_inverse") 
        plt.plot(train_true_target_inverse[400:-1], label="train_true_target_inverse")    
        plt.title("Real and Predicted Percentage Price Change in Trainset (Part) ")
        plt.ylabel("Ratio (De-Normolization)")
        plt.xlabel("Trading Day")    
        plt.legend()
        plt.show()
    
    train_ori = data.iloc[seq:data.shape[0]-test_set_size]
    #print(train_ori.shape)
    train_Open_ori = scalers["Open"].inverse_transform(train_ori["Open"].values.reshape(-1, 1))
    train_Close_ori = scalers["Close"].inverse_transform(train_ori["Close"].values.reshape(-1, 1))
    train_ratio_ori = scalers["ratio"].inverse_transform(train_ori["ratio"].values.reshape(-1, 1))
    
    train_pred_price = train_target_inverse[0:-1] * train_Open_ori[1::] + train_Open_ori[1::] #the predict is the close price of the next day
    train_true_price = train_true_target_inverse[0:-1] * train_Open_ori[1::] + train_Open_ori[1::] 
    
    if plot == True:
        plt.plot(train_true_price, label="train_true_price") 
        plt.plot(train_pred_price, label="train_pred_price") 
        plt.title("Stock Price Prediciton for Trainset")
        plt.ylabel("Stock Price")
        plt.xlabel("Trading Day")  
        plt.legend()    
        plt.show()

        plt.plot(train_true_price[400:-1], label="train_true_price") 
        plt.plot(train_pred_price[400:-1], label="train_pred_price") 
        plt.title("Stock Price Prediciton for Trainset (Part)")
        plt.ylabel("Stock Price")
        plt.xlabel("Trading Day")  
        plt.legend()    
        plt.show()
    
    # Assuming train_pred_price and train_true_close are NumPy arrays or lists
    train_mse = mean_squared_error(train_true_price, train_pred_price)
    print(f"train_MSE: {train_mse}")
    train_rmse = np.sqrt(train_mse)
    print(f"train_RMSE: {train_rmse}")

    value_range = np.max(train_true_price) - np.min(train_true_price)
    train_rrmse = train_rmse / value_range
    print(f"train_rRMSE: {train_rrmse}")

    # Assuming train_pred_price and train_true_close are NumPy arrays or lists
    train_mae = mean_absolute_error(train_true_price, train_pred_price)
    print(f"train_MAE: {train_mae}")
    # Assuming train_pred_price and train_true_close are NumPy arrays or lists
    train_r_squared = r2_score(train_true_price, train_pred_price)
    print(f"train_R-squared (R²): {train_r_squared}")

    wins = 0
    losses = 0
    prev_real_price = train_true_price[0]
    prev_pred_price = train_pred_price[0]

    for real_price, pred_price in zip(train_true_price[1:], train_pred_price[1:]):
        if real_price > prev_real_price and pred_price > prev_pred_price:
            wins += 1
        elif real_price < prev_real_price and pred_price < prev_pred_price:
            wins += 1
        else:
            losses += 1
        prev_real_price = real_price
        prev_pred_price = pred_price

    # caculate Win-Loss Ratio
    if losses != 0:
        win_loss_ratio = wins / losses
    else:
        win_loss_ratio = "inf"
    print("train Win-Loss Ratio:", win_loss_ratio)
    print("#######################################")
    return train_pred_price, train_true_price

    
def test_matrics(test_pred_file,data,testY,scalers,test_set_size = 30,plot=True):
    with open(test_pred_file, 'rb') as f:
        test_pred_lst = pickle.load(f)
        
    test_mean_pred = np.mean(test_pred_lst,axis = 0)
    test_ori = data.tail(test_set_size)
    test_true_value = testY.detach().numpy()[:,-1,0]
    
    test_Open_ori = scalers["Open"].inverse_transform(test_ori["Open"].values.reshape(-1, 1))
    test_Close_ori = scalers["Close"].inverse_transform(test_ori["Close"].values.reshape(-1, 1))
    test_ratio_ori = scalers["ratio"].inverse_transform(test_ori["ratio"].values.reshape(-1, 1))
    
    test_target_inverse = scalers["target"].inverse_transform(test_mean_pred.reshape(-1, 1))
    test_true_target_inverse = scalers["target"].inverse_transform(test_true_value.reshape(-1, 1))
    if plot == True:
        plt.plot(test_target_inverse, label="validation_target_inverse") 
        plt.plot(test_true_target_inverse, label="validation_true_target_inverse") 
        plt.title("Real and Predicted Percentage Price Change in Validation Set")
        plt.ylabel("Ratio (De-Normolization)")
        plt.xlabel("Trading Day")    
        plt.legend()
        plt.show()
        
    test_pred_price = test_target_inverse[0:-1] * test_Open_ori[1::]+test_Open_ori[1::] #the predict is the close price of the next day
    test_true_price = test_true_target_inverse[0:-1] * test_Open_ori[1::] + test_Open_ori[1::] 

    if plot == True:
        plt.plot(test_pred_price, label="test_pred_price") 
        plt.plot(test_true_price, label="test_true_price")
        plt.title("Stock Price Prediciton for Validation set")
        plt.ylabel("Stock Price")
        plt.xlabel("Trading Day")  
        plt.legend()    
        plt.show()
        
    test_mse = mean_squared_error(test_true_price, test_pred_price)
    print(f"test_MSE: {test_mse}")
    test_rmse = np.sqrt(test_mse)
    print(f"test_RMSE: {test_rmse}")

    value_range = np.max(test_true_price) - np.min(test_true_price)
    test_rrmse = test_rmse / value_range
    print(f"test_rRMSE: {test_rrmse}")

    # Assuming train_pred_price and train_true_close are NumPy arrays or lists
    test_mae = mean_absolute_error(test_true_price, test_pred_price)
    print(f"test_MAE: {test_mae}")
    # Assuming train_pred_price and train_true_close are NumPy arrays or lists
    test_r_squared = r2_score(test_true_price, test_pred_price)
    print(f"test_R-squared (R²): {test_r_squared}")

    wins = 0
    losses = 0
    prev_real_price = test_true_price[0]
    prev_pred_price = test_pred_price[0]

    for real_price, pred_price in zip(test_true_price[1:], test_pred_price[1:]):
        if real_price > prev_real_price and pred_price > prev_pred_price:
            wins += 1
        elif real_price < prev_real_price and pred_price < prev_pred_price:
            wins += 1
        else:
            losses += 1
        prev_real_price = real_price
        prev_pred_price = pred_price

    # caculate Win-Loss Ratio
    if losses != 0:
        win_loss_ratio = wins / losses
    else:
        win_loss_ratio = "inf"
    print("test_Win-Loss Ratio:", win_loss_ratio)
    print("#######################################")
    return test_pred_price, test_true_price


def run_test2(trainX,trainY,*testX,
              input_dim = 10, hidden_dim = 128, num_layers = 3, 
              output_dim = 1,seq = 10,num_epochs = 500,means = 2,
              model_name = "LSTM",bidirectional=True, 
              name=["AAL","DAL","UAL","LUV"]):
    

    train_pred_lst = []
    test_pred_lst=[]
    train_loss_lst=[]
    test_loss_lst=[]
    train_mae_lst=[]
    test_mae_lst=[]
    train_r2_lst=[]
    test_r2_lst = []
    
    test_pred_AAL_lst=[]
    test_pred_UAL_lst=[]
    test_pred_DAL_lst=[]
    test_pred_LUV_lst=[]
    for i in range(means):
        model = LSTM_GRU_RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                        bidirectional=bidirectional, model = model_name)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0) 
        loss_fn = torch.nn.MSELoss(reduction='mean')           
        model.train()
        # hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            # model.hidden = model.init_hidden()
            # Forward pass
            y_train_pred_inter = model(trainX) # y_train_pred_inter and trainY were bitch_size*sqe*1

            loss = loss_fn(y_train_pred_inter, trainY)
            if t % 20 == 0 and t !=0:               
                print("Epoch ", t, "MSE: ", loss.item())
            # hist[t] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()
            
            # Update parameters
            optimiser.step()
        
        # Set the model to evaluation mode
        model.eval()
        # # Train Predict Value
        # # Predict the values using the model on the training data
        # y_train_pred = model(trainX)
        # #print("y_train_pred size=",y_train_pred.shape )  #  538*10*1
        # train_pred = y_train_pred.detach().numpy()[:,-1,0]  # Extract the final predicted values
        # train_pred_lst.append(train_pred)                   # 每个长度的最后一个值就是估算的结果，共计batch_size个

        y_test_pred_AAL = model(testX[0])
        test_pred_AAL = y_test_pred_AAL.detach().numpy()[:,-1,0]  # Extract the final predicted values
        test_pred_AAL_lst.append(test_pred_AAL)

        y_test_pred_DAL = model(testX[1])
        test_pred_DAL = y_test_pred_DAL.detach().numpy()[:,-1,0]  # Extract the final predicted values
        test_pred_DAL_lst.append(test_pred_DAL)
        
        y_test_pred_UAL = model(testX[2])
        test_pred_UAL = y_test_pred_UAL .detach().numpy()[:,-1,0]  # Extract the final predicted values
        test_pred_UAL_lst.append(test_pred_UAL)
        
        y_test_pred_LUV = model(testX[3])
        test_pred_LUV = y_test_pred_LUV .detach().numpy()[:,-1,0]  # Extract the final predicted values
        test_pred_LUV_lst.append(test_pred_LUV)
    # with open(f'{name}_{model_name}_train_pred_ind{input_dim}_hd{hidden_dim}_nl{num_layers}_outd{output_dim}_seq{seq}_ne{num_epochs}_means{means}.pkl', 'wb') as f:
    #     pickle.dump(train_pred_lst, f)
    with open(f'{name[0]}_{model_name}_test2_pred_ind{input_dim}_hd{hidden_dim}_nl{num_layers}_outd{output_dim}_seq{seq}_ne{num_epochs}_means{means}.pkl', 'wb') as f:
        pickle.dump(y_test_pred_AAL, f)
    with open(f'{name[1]}_{model_name}_test2_pred_ind{input_dim}_hd{hidden_dim}_nl{num_layers}_outd{output_dim}_seq{seq}_ne{num_epochs}_means{means}.pkl', 'wb') as f:
        pickle.dump(y_test_pred_DAL, f)
    with open(f'{name[2]}_{model_name}_test2_pred_ind{input_dim}_hd{hidden_dim}_nl{num_layers}_outd{output_dim}_seq{seq}_ne{num_epochs}_means{means}.pkl', 'wb') as f:
        pickle.dump(y_test_pred_UAL, f)
    with open(f'{name[3]}_{model_name}_test2_pred_ind{input_dim}_hd{hidden_dim}_nl{num_layers}_outd{output_dim}_seq{seq}_ne{num_epochs}_means{means}.pkl', 'wb') as f:
        pickle.dump(y_test_pred_LUV, f)
   
    #torch.save(model.state_dict(), 'model_w_parameters.pth')

    return test_pred_AAL_lst, test_pred_DAL_lst, test_pred_UAL_lst, test_pred_LUV_lst


def test_matrics2(data,test_pred_lst,testY,scalers,test_set_size = 30,plot=True,stock_name = "AAL"):
    # loaded_model = LSTM_GRU_RNN(input_dim=10, hidden_dim=128, output_dim=1, num_layers=3,
    #                     bidirectional=True, model = "LSTM")
    # loaded_model.load_state_dict(torch.load('model_w_parameters.pth'))
    
    # test_pred_lst = []
    # y_pred = loaded_model(testX)
    # test_pred = y_pred.detach().numpy()[:,-1,0]  # Extract the final predicted values
    # test_pred_lst.append(test_pred)
    test_mean_pred = np.mean(test_pred_lst,axis = 0)
    test_ori = data.tail(test_set_size)
    test_true_value = testY.detach().numpy()[:,-1,0]
    
    test_Open_ori = scalers["Open"].inverse_transform(test_ori["Open"].values.reshape(-1, 1))
    test_Close_ori = scalers["Close"].inverse_transform(test_ori["Close"].values.reshape(-1, 1))
    test_ratio_ori = scalers["ratio"].inverse_transform(test_ori["ratio"].values.reshape(-1, 1))
    
    test_target_inverse = scalers["target"].inverse_transform(test_mean_pred.reshape(-1, 1))
    test_true_target_inverse = scalers["target"].inverse_transform(test_true_value.reshape(-1, 1))
    if plot == True:
        plt.plot(test_target_inverse, label=f"test_target_inverse_{stock_name}") 
        plt.plot(test_true_target_inverse, label=f"test_true_target_inverse_{stock_name}") 
        plt.title(f"Real and Predicted Percentage Price Change in Test Set {stock_name}")
        plt.ylabel("Ratio (De-Normolization)")
        plt.xlabel("Trading Day")    
        plt.legend()
        plt.show()
        
    test_pred_price = test_target_inverse[0:-1] * test_Open_ori[1::]+test_Open_ori[1::] #the predict is the close price of the next day
    test_true_price = test_true_target_inverse[0:-1] * test_Open_ori[1::] + test_Open_ori[1::] 

    if plot == True:
        plt.plot(test_pred_price, label=f"test_pred_price_{stock_name}") 
        plt.plot(test_true_price, label=f"test_true_price_{stock_name}")
        plt.title(f"Stock Price Prediciton for Test set {stock_name}")
        plt.ylabel("Stock Price")
        plt.xlabel("Trading Day")  
        plt.legend()    
        plt.show()
        
    test_mse = mean_squared_error(test_true_price, test_pred_price)
    print(f"test_MSE_{stock_name}: {test_mse}")
    test_rmse = np.sqrt(test_mse)
    print(f"test_RMSE_{stock_name}: {test_rmse}")

    value_range = np.max(test_true_price) - np.min(test_true_price)
    test_rrmse = test_rmse / value_range
    print(f"test_rRMSE_{stock_name}: {test_rrmse}")

    # Assuming train_pred_price and train_true_close are NumPy arrays or lists
    test_mae = mean_absolute_error(test_true_price, test_pred_price)
    print(f"test_MAE_{stock_name}: {test_mae}")
    # Assuming train_pred_price and train_true_close are NumPy arrays or lists
    test_r_squared = r2_score(test_true_price, test_pred_price)
    print(f"test_R-squared(R²)_{stock_name}: {test_r_squared}")

    wins = 0
    losses = 0
    prev_real_price = test_true_price[0]
    prev_pred_price = test_pred_price[0]

    for real_price, pred_price in zip(test_true_price[1:], test_pred_price[1:]):
        if real_price > prev_real_price and pred_price > prev_pred_price:
            wins += 1
        elif real_price < prev_real_price and pred_price < prev_pred_price:
            wins += 1
        else:
            losses += 1
        prev_real_price = real_price
        prev_pred_price = pred_price

    # caculate Win-Loss Ratio
    if losses != 0:
        win_loss_ratio = wins / losses
    else:
        win_loss_ratio = "inf"
    print(f"test_Win-Loss-Ratio_{stock_name}:", win_loss_ratio)
    print("#######################################")
    return test_pred_price, test_true_price