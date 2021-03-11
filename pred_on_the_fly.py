import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
from novograd import Novograd
import argparse

class COVID(nn.Module):
    def __init__(self):
        super(COVID, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2)

# dataset downloaded from https://github.com/iiscleap/Coswara-Data
class On_the_fly_Coughing(Dataset):
#rapper for the Cough dataset
    # Argument List
    #  path to the Cough csv file    
    def __init__(self, csv_path , label_to_retrieve):
        csvData = pd.read_csv(csv_path)
        #initialize lists to hold file names, labels, and folder numbers
        self.labels = []
        self.file_path = []
        self.indexes = []
        self.shirnked_labels={0: 'healthy', 1: 'unknow_rep_illness', 2: 'covid_positive', 3: 'asymptotic'}
        #loop through the csv entries and only add entries from folders in the folder list
        for i in range(0,len(csvData)):
            if csvData.iloc[i,2] in label_to_retrieve:
                self.file_path.append(csvData.iloc[i, -2])
                self.labels.append(self.__shirnk_labels__(csvData.iloc[i, 2]))
                self.indexes.append(i)   
    def __shirnk_labels__(self,cur_label):
        if cur_label==1 or cur_label==2:
            cur_label=1
        elif cur_label==3 or cur_label==4 or cur_label==5 or cur_label==6 :
            cur_label=2             
        elif cur_label==0:
            cur_label=0
        else:
            print("the allowed label is 0:healthy ,1:unknown_rep_illlness, 2: covid_positive, 3: asymptotic , but got = ", cur_label)
        return cur_label
    def __getitem__(self, index):
        path = self.file_path[index]
        print("wav file location " , path)
        print("label ", self.shirnked_labels[self.labels[index]])
        #sound = torchaudio.load(path, out = None, normalization = True)
        sound = torchaudio.load(path, out = None, normalization = True)
        soundData = sound[0]
        #load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        #downsample the audio to ~8kHz
        tempData = torch.zeros([160000]) #tempData accounts for audio clips that are too short
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[0,:]
        else:
            tempData[:] = soundData[0,:160000]
        
        soundData = tempData
        soundFormatted = torch.zeros([32000])
        soundFormatted[:32000] = soundData[::5] #take every fifth sample of soundData
        soundFormatted=torch.unsqueeze(soundFormatted,0)
        return path, soundFormatted , self.labels[index]
    
    def __len__(self):
        return len(self.file_path)

def visualize_spectrogram(audio, lb):
    # Get spectrogram using Librosa's Short-Time Fourier Transform (stft)
    spec = np.abs(librosa.stft(audio))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)  # Decibels

    # Use log scale to view frequencies
    librosa.display.specshow(spec_db, y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Audio Spectrogram **{}** Example'.format(num2labels[lb.item()]))
    plt.show()

def get_on_the_fly_prediction(input_data, loaded_model,lb):
    out= loaded_model(input_data)
    out = out.permute(1, 0, 2)
    pred = out.max(2)[1].item()
    assert type(pred)==int
    return num2labels[pred]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--path",
                        default=None,
                        type=str,
                        required=True,
                        help="path to the csv file containing the testing data")
    parser.add_argument("--saved_model", default='./saved_model/convid.pt', type=str, required=True,
                        help="path to the saved model path, default to './saved_model/convid.pt' ")
    parser.add_argument("--use_label",
                        default='asym',
                        type=str,
                        required=True,
                        help=" healthy, asyn =has covid and is asymptomatic ,covid_pos= tested positive with covid but excluding asymptomatic")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shirnked_labels={0: 'healthy', 1: 'unknow_rep_illness', 2: 'covid_positive', 3: 'asymptotic'}
    print("usin device", device)
    # load pre-trained model 
    loaded_model = COVID()
    loaded_model.to(device)
    opt = Novograd(loaded_model.parameters(), lr = 0.01, weight_decay = 0.0001)
    sch = optim.lr_scheduler.StepLR(opt, step_size = 20, gamma = 0.1)
    PATH=args.saved_model
    checkpoint = torch.load(PATH)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loaded_model.eval()
    ### loading test dataset 
    csv_path = 'test.csv'   
    num2labels={0: 'healthy', 1: 'resp_illness_not_identified', 2: 'no_resp_illness_exposed', 3: 'recovered_full', 4: 'positive_mild', 5: 'positive_asymp', 6: 'positive_moderate'}    
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu
    if args.use_label=='asym': 
        asymptomatic = On_the_fly_Coughing(csv_path,[5])
        asymptomatic_loader = torch.utils.data.DataLoader(asymptomatic, batch_size = 1, shuffle = True, **kwargs)
        asyn_wav_loc, asyn_in_data, asyn_lb= iter(asymptomatic_loader).next()
        pred=get_on_the_fly_prediction(asyn_in_data.cuda(), loaded_model, asyn_lb )
        print("prediction", pred , "| true label ", num2labels[asyn_lb.item()])
    elif args.use_label=='covid_pos':
        positive_with_covid=On_the_fly_Coughing(csv_path,[3,4,6])
        covid_excl_asy_loader = torch.utils.data.DataLoader(positive_with_covid, batch_size = 1, shuffle = True, **kwargs)
        covid_pos_wav_loc, covid_pos_in_data, covid_lb= iter(covid_excl_asy_loader).next()
        pred=get_on_the_fly_prediction(covid_pos_in_data.cuda(),loaded_model, covid_lb)
        print("prediction", pred, "| true label " , num2labels[covid_lb.item()])
 
    elif args.use_label=='healthy':
        healthy=On_the_fly_Coughing(csv_path,[0])
        healthy_loader = torch.utils.data.DataLoader(healthy, batch_size = 1, shuffle = True, **kwargs)
        healthy_wav_loc, healthy_pos_in_data, healthy_lb= iter(healthy_loader).next()
        pred=get_on_the_fly_prediction(healthy_pos_in_data.cuda(), loaded_model,healthy_lb )
        print("prediction", pred, "| true label ", num2labels[healthy_lb.item()])
    else:
        print("please use one of the following : healthy , asym , covid_pos")


