import torch
import cv2
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from transformers import AdamW, AutoModel, AutoTokenizer, ViTForImageClassification, ElectraModel, ElectraForSequenceClassification, ElectraTokenizer, BertForSequenceClassification, BertTokenizerFast, ViTConfig
from transformers import AutoModel,ViTModel,ViTFeatureExtractor
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import glob
import gc
import pandas as pd
import numpy as np
import os
from eda import *
from transformers import AdamW
from sklearn.metrics import f1_score

def make_id(x):
    return (int(x[6:]))

def export_txt(tokenizer, row, aug, max_length) :
    text = row[0]
    if (aug > 0) and (aug <=1) : 
        text = EDA(text, alpha_sr=aug, alpha_ri=aug, alpha_rs=aug, p_rd=aug, num_aug=1)[0]
    else :
        pass
    inputs = tokenizer(
        text, 
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        pad_to_max_length=True,
        add_special_tokens=True
        )
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]
    return input_ids, attention_mask

class txt_Dataset(Dataset) :
    
    def __init__(self, df, tokenizer, encoders, max_length = 512, aug = 0):
        self.max_length = max_length
        self.dataset = df.dropna(axis=0)
        self.dataset['label_0'] =  encoders[0].transform(self.dataset['유형'])
        self.dataset['label_1'] =  encoders[1].transform(self.dataset['극성'])
        self.dataset['label_2'] =  encoders[2].transform(self.dataset['시제'])
        self.dataset['label_3'] =  encoders[3].transform(self.dataset['확실성'])
        self.dataset = self.dataset.rename(columns = {'문장' : 'document'})
        self.dataset['ID'] = self.dataset['ID'].apply(make_id)
        self.dataset = self.dataset[['ID', 'document', 'label_0', 'label_1', 'label_2', 'label_3']]
        self.tokenizer = tokenizer
        self.aug = aug
        print('found {} data'.format(len(self.dataset)))
        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 1:].values
        input_ids, attention_mask = export_txt(self.tokenizer, row, self.aug, self.max_length) 
        y0, y1, y2, y3 = row[1], row[2], row[3], row[4]
        return {
            'input_ids':  input_ids,
            'attention_mask': attention_mask,
            'y0':y0, 'y1':y1, 'y2':y2, 'y3':y3
        }

def infer(model, test_loader, device, mode) :
    model.eval()
    correct = 0
    batches = 0
    total = 0
    total_loss = 0.0
    type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
    type_labels, polarity_labels, tense_labels, certainty_labels = [], [], [], []
    with torch.no_grad() :
        for i, batch in enumerate(tqdm(test_loader)):
            if mode == 'text' or mode == 'both' :
                input_ids_batch = batch["input_ids"].to(device)
                attention_masks_batch = batch["attention_mask"].to(device)
            if mode == 'img' or mode == 'both' :
                pixel_values = batch['pixel_values'].to(device)
            y_batch_0 = batch["y0"].to(device, dtype=torch.int64)
            y_batch_1 = batch["y1"].to(device, dtype=torch.int64)
            y_batch_2 = batch["y2"].to(device, dtype=torch.int64)
            y_batch_3 = batch["y3"].to(device, dtype=torch.int64)

            
            if mode == 'text' :
                y_pred_0, y_pred_1, y_pred_2, y_pred_3 = model(input_ids_batch, attention_masks_batch)

            type_preds += y_pred_0.argmax(1).detach().cpu().numpy().tolist()
            type_labels += y_batch_0.detach().cpu().numpy().tolist()

            polarity_preds += y_pred_1.argmax(1).detach().cpu().numpy().tolist()
            polarity_labels += y_batch_1.detach().cpu().numpy().tolist()

            tense_preds += y_pred_2.argmax(1).detach().cpu().numpy().tolist()
            tense_labels += y_batch_2.detach().cpu().numpy().tolist()

            certainty_preds += y_pred_3.argmax(1).detach().cpu().numpy().tolist()
            certainty_labels += y_batch_3.detach().cpu().numpy().tolist()

            act_vec_0 = torch.nn.Softmax(1)(y_pred_0)
            act_vec_1 = torch.nn.Softmax(1)(y_pred_1)
            act_vec_2 = torch.nn.Softmax(1)(y_pred_2)
            act_vec_3 = torch.nn.Softmax(1)(y_pred_3)
            
            batches += 1
            if i == 0 :
                act_vec_whole_0 = act_vec_0.cpu()
                act_vec_whole_1 = act_vec_1.cpu()
                act_vec_whole_2 = act_vec_2.cpu()
                act_vec_whole_3 = act_vec_3.cpu()
            else :
                act_vec_whole_0 = torch.cat((act_vec_whole_0, act_vec_0.cpu()))
                act_vec_whole_1 = torch.cat((act_vec_whole_1, act_vec_1.cpu()))
                act_vec_whole_2 = torch.cat((act_vec_whole_2, act_vec_2.cpu()))
                act_vec_whole_3 = torch.cat((act_vec_whole_3, act_vec_3.cpu()))
                
    return [type_preds, polarity_preds, tense_preds, certainty_preds], [act_vec_whole_0.numpy(), act_vec_whole_1.numpy(), act_vec_whole_2.numpy(), act_vec_whole_3.numpy()]
#
# def train_one_epoch(self, train_loader, train_mode) :
#     if train_mode == True :
#         self.model.train()
#     else :
#         self.model.eval()
#     correct = 0
#     batches = 0
#     total = 0
#     type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
#     for batch in tqdm(train_loader):
#         self.optimizer.zero_grad()
#         if self.mode == 'text' or self.mode == 'both' :
#             input_ids_batch = batch["input_ids"].to(self.device)
#             attention_masks_batch = batch["attention_mask"].to(self.device)
#         if self.mode == 'img' or self.mode == 'both' :
#             pixel_values = batch['pixel_values'].to(self.device)
#         y_batch_0 = batch["y0"].to(self.device, dtype=torch.int64)
#         y_batch_1 = batch["y1"].to(self.device, dtype=torch.int64)
#         y_batch_2 = batch["y2"].to(self.device, dtype=torch.int64)
#         y_batch_3 = batch["y3"].to(self.device, dtype=torch.int64)
#
#         if self.mode == 'text' :
#             y_pred_0, y_pred_1, y_pred_2, y_pred_3 = self.model(input_ids_batch, attention_masks_batch)
#
#         type_preds += y_pred_0.argmax(1).detach().cpu().numpy().tolist()
#         polarity_preds += y_pred_1.argmax(1).detach().cpu().numpy().tolist()
#         tense_preds += y_pred_2.argmax(1).detach().cpu().numpy().tolist()
#         certainty_preds += y_pred_3.argmax(1).detach().cpu().numpy().tolist()
#
#     print("End_Loss:", total_loss, "type_f1:", type_f1, "polarity_f1:", polarity_f1,
#           "tense_f1:", tense_f1, "certainty_f1:", certainty_f1 )
#     #accuracy = correct.float() / len(train_loader.dataset)
#     return [type_f1,  polarity_f1, tense_f1,  certainty_f1, weighted_f1]

def train_test_split_k(df, k) :
    df_train = df[df['kfold']!=k]
    df_val = df[df['kfold']==k]
    return df_train, df_val
    
def load_dataset(path, k, random_state, reset_k = True) :
    df = pd.read_csv(path)
    encoder_0 = LabelEncoder()
    encoder_1 = LabelEncoder()
    encoder_2 = LabelEncoder()
    encoder_3 = LabelEncoder()
    encoder_0.fit(sorted((pd.read_csv(path))['유형'].unique()))
    encoder_1.fit(sorted((pd.read_csv(path))['극성'].unique()))
    encoder_2.fit(sorted((pd.read_csv(path))['시제'].unique()))
    encoder_3.fit(sorted((pd.read_csv(path))['확실성'].unique()))

    
    if reset_k == True :
        folds = StratifiedKFold(n_splits=k, random_state=random_state, shuffle=True)
        df['kfold'] = -1
        for i in range(k):
            df_idx, valid_idx = list(folds.split(df.values, df['유형']))[i]
            valid = df.iloc[valid_idx]
            df.loc[df[df.ID.isin(valid.ID) == True].index.to_list(), 'kfold'] = i
        df.to_csv('Data/train_k.csv')
    else :
        pass
    return df, [encoder_0, encoder_1, encoder_2, encoder_3]
                

def define_model_and_tokenizer(md, device) : 
    if md == 'klue_bert_base' : 
        model = TxtClassifier("klue/bert-base", device, n_classes = [4, 3, 3, 2])
        tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
    return model, tokenizer


class TxtClassifier(nn.Module):
    def __init__(self, text_model_name, device, n_classes):
        super(TxtClassifier, self).__init__()
        self.device = device
        self.text_model = AutoModel.from_pretrained(text_model_name).to(self.device)
        #self.text_model = text_model
        self.text_model.gradient_checkpointing_enable()  
        self.drop = nn.Dropout(p=0.1)

        def get_cls(target_size):
            return nn.Sequential(
              nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size).to(self.device),
              nn.LayerNorm(self.text_model.config.hidden_size).to(self.device),
              nn.Dropout(p = 0.1),
              nn.ReLU().to(self.device),
              nn.Linear(self.text_model.config.hidden_size, target_size).to(self.device),
            )  
        self.cls0 = get_cls(n_classes[0])
        self.cls1 = get_cls(n_classes[1])
        self.cls2 = get_cls(n_classes[2])
        self.cls3 = get_cls(n_classes[3])
        
    
    def forward(self, input_ids, attention_mask):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8).to(self.device)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(self.device)
        outputs = transformer_encoder(text_output.last_hidden_state)
        outputs = outputs[:,0]
        output = self.drop(outputs)
        out0 = self.cls0(output)
        out1 = self.cls1(output)
        out2 = self.cls2(output)
        out3 = self.cls3(output)
        
        return out0, out1, out2, out3

class fine_tuning : 
    def __init__(self, img_root, exp, file, condition, cv_k, model_txt, model_img, batch_size,  lr, max_length, decay, save_interval,
                 train_path, batch_size_test = 16, max_epoch = 20, early_stopping =None, aug_num = 0, aug = 0, mode = 'text') :

        self.exp = exp
        self.file = file
        self.condition = condition
        self.cv_k = cv_k
        self.file_path = os.path.join(self.file, self.condition)
        self.file_k_path = os.path.join(self.file_path, str(self.cv_k))
        os.makedirs(self.file_k_path, exist_ok=True)
        self.max_epoch = max_epoch
        self.md_txt = model_txt
        self.md_img = model_img
        self.aug = aug
        self.aug_num = 0
        self.ml = max_length
        self.b = batch_size
        self.d = decay
        self.lr = lr
        self.save_interval = save_interval
        self.mode = mode

        df, self.encoders = load_dataset(train_path, 4, 42, True)
        self.df_train, self.df_val = train_test_split_k(df, cv_k) # 0~3 까지의 값

        if torch.cuda.is_available() == True :
            self.device = torch.device("cuda")
        else :
            self.device = torch.device("cpu")
            
        if self.mode == 'text' :
            self.model, self.tokenizer = define_model_and_tokenizer(self.md_txt, self.device)
            train_dataset = txt_Dataset(self.df_train, self.tokenizer, self.encoders, int(self.ml), aug)
            val_dataset = txt_Dataset(self.df_val, self.tokenizer, self.encoders, int(self.ml), False)
            self.extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch32-384')

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.d, correct_bias=True) # AdamW에서 bias correction 과정만 생략해주시면 BERTAdam이 됩니다!

        self.train_loader = DataLoader(train_dataset, batch_size=self.b, shuffle=True, num_workers = 0)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, num_workers = 0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10, threshold=5e-8)
        
    def train(self) :
        self.history_list = []
        best_val_f1 = 0
        for i in range(self.max_epoch):
            history = dict()
            loss_train, f1_scores_train = self.train_one_epoch(self.train_loader, True)
            print("train_Loss:", loss_train, "weighted_f1:",  f1_scores_train[4], "whole_f1:",  f1_scores_train[5],  "type_f1:",  f1_scores_train[0], "polarity_f:",  f1_scores_train[1], "tense_f1:",  f1_scores_train[2], "certainty_f1:",  f1_scores_train[3])
            with torch.no_grad() :
                loss_val, f1_scores_val = self.train_one_epoch(self.val_loader, False)
            print("val_Loss:", loss_val,"val_weighted_f1:",  f1_scores_val[4], "val_whole_f1:",  f1_scores_val[5], "val_type_f1:",  f1_scores_val[0], "val_polarity_f:",  f1_scores_val[1], "val_tense_f1:",  f1_scores_val[2], "val_certainty_f1:",  f1_scores_val[3])
            history['loss'] =  loss_train
            history['type_f1']  = f1_scores_train[0]
            history['polarity_f1'] = f1_scores_train[1]
            history['tense_f1'] = f1_scores_train[2]
            history['certainty_f1'] = f1_scores_train[3]
            history['weighted_f1'] = f1_scores_train[4]
            history['whole_f1'] = f1_scores_train[5]
            history['val_loss'] = loss_val
            history['type_f1_val']  = f1_scores_val[0]
            history['polarity_f1_val'] = f1_scores_val[1]
            history['tense_f1_val'] = f1_scores_val[2]
            history['certainty_f1_val'] = f1_scores_val[3]
            history['weighted_f1_val'] = f1_scores_val[4]
            history['whole_f1_val'] = f1_scores_val[5]
            self.scheduler.step(f1_scores_val[5])
            self.history_list.append(history)
            if i % self.save_interval == 0:
                if f1_scores_val[5] > best_val_f1 :
                    print('val_f1 improved!! saving model...')
                    targetPattern = r"{}/*.pt".format(self.file_k_path)
                    h5_list = glob.glob(targetPattern)
                    if len(h5_list) >= 1 :
                        print('previous model deleted')
                        os.remove(h5_list[0])
                    checkpoint_path = os.path.join(self.file_k_path,
                                                   self.condition + '_{:02d}-{:.3f}-{:.3f}-{:.3f}.pt'.format(i, loss_val, f1_scores_train[5], f1_scores_val[5]))
                    torch.save(self.model.state_dict(), checkpoint_path)
                    best_val_f1 = f1_scores_val[5]

                else :
                    print('val_accuracy is not improved...')
        self.history = pd.DataFrame(self.history_list)

    def train_one_epoch(self, train_loader, train_mode) :
        if train_mode == True :
            self.model.train()
        else :
            self.model.eval()
        correct = 0
        batches = 0
        total = 0
        total_loss = 0.0
        type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
        type_labels, polarity_labels, tense_labels, certainty_labels = [], [], [], []
        for batch in tqdm(train_loader):
            self.optimizer.zero_grad()
            if self.mode == 'text' or self.mode == 'both' :
                input_ids_batch = batch["input_ids"].to(self.device)
                attention_masks_batch = batch["attention_mask"].to(self.device)
            if self.mode == 'img' or self.mode == 'both' :
                pixel_values = batch['pixel_values'].to(self.device)
            y_batch_0 = batch["y0"].to(self.device, dtype=torch.int64)
            y_batch_1 = batch["y1"].to(self.device, dtype=torch.int64)
            y_batch_2 = batch["y2"].to(self.device, dtype=torch.int64)
            y_batch_3 = batch["y3"].to(self.device, dtype=torch.int64)

            if self.mode == 'text' :
                y_pred_0, y_pred_1, y_pred_2, y_pred_3 = self.model(input_ids_batch, attention_masks_batch)#[0]

            loss_0 = F.cross_entropy(y_pred_0, y_batch_0)
            loss_1 = F.cross_entropy(y_pred_1, y_batch_1)
            loss_2 = F.cross_entropy(y_pred_2, y_batch_2)
            loss_3 = F.cross_entropy(y_pred_3, y_batch_3)
            loss = 0.25 * loss_0 + 0.25 * loss_1 + 0.25 * loss_2 + 0.25 * loss_3
            
            if train_mode == True :
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * len(y_batch_0)
            type_preds += y_pred_0.argmax(1).detach().cpu().numpy().tolist()
            type_labels += y_batch_0.detach().cpu().numpy().tolist()
            polarity_preds += y_pred_1.argmax(1).detach().cpu().numpy().tolist()
            polarity_labels += y_batch_1.detach().cpu().numpy().tolist()
            tense_preds += y_pred_2.argmax(1).detach().cpu().numpy().tolist()
            tense_labels += y_batch_2.detach().cpu().numpy().tolist()
            certainty_preds += y_pred_3.argmax(1).detach().cpu().numpy().tolist()
            certainty_labels += y_batch_3.detach().cpu().numpy().tolist()

            total += len(y_batch_0)
            batches += 1
            if (batches % 100 == 0) and (train_mode == True) :
                type_f1 = f1_score(type_labels, type_preds, average='macro')
                polarity_f1 = f1_score(polarity_labels, polarity_preds, average='macro')
                tense_f1 = f1_score(tense_labels, tense_preds, average='macro')
                certainty_f1 = f1_score(certainty_labels, certainty_preds, average='macro')
                print("Loss:", loss.item(), "type_f1:", type_f1, "polarity_f1:", polarity_f1,
                      "tense_f1:", tense_f1, "certainty_f1:", certainty_f1)
        total_loss = total_loss / len(train_loader.dataset)
        
        label_whole = list(np.array(type_labels) * 4**0 + np.array(polarity_labels) * 4**1 + np.array(tense_labels) * 4**2 + np.array(certainty_labels) * 4**3)
        preds_whole = list(np.array(type_preds) * 4**0 + np.array(polarity_preds) * 4**1 + np.array(tense_preds) * 4**2 + np.array(certainty_preds) * 4**3)
        
        type_f1 = f1_score(type_labels, type_preds, average='macro')
        polarity_f1 = f1_score(polarity_labels, polarity_preds, average='macro')
        tense_f1 = f1_score(tense_labels, tense_preds, average='macro')
        certainty_f1 = f1_score(certainty_labels, certainty_preds, average='macro')
        weighted_f1 = (type_f1 + polarity_f1 + tense_f1 + certainty_f1)/4
        whole_f1 = f1_score(label_whole, preds_whole, average='weighted')
        print("End_Loss:", total_loss, "type_f1:", type_f1, "polarity_f1:", polarity_f1,
              "tense_f1:", tense_f1, "certainty_f1:", certainty_f1, "whole_f1:",whole_f1  )

        return total_loss, [type_f1,  polarity_f1, tense_f1,  certainty_f1, weighted_f1, whole_f1]

    def save_history(self) :
        self.show_history()
        self.save_summary()
        
    def show_history(self) :
        fig = plt.figure(figsize=(13,6))
        ax1 = fig.add_subplot(1, 2, 1)
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        ax2 = fig.add_subplot(1, 2, 2)
        plt.plot(self.history['whole_f1'])
        plt.plot(self.history['whole_f1_val'])
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.savefig(os.path.join(self.file_k_path, 'Loss-f1_plot.png'))
        self.history.to_csv(os.path.join(self.file_k_path,'history.csv'))
        
        
    def save_summary(self) :
        A = np.array(self.history.index+1)
        index = list(A[A%self.save_interval==0]-1)
        Result = self.history[['loss','val_loss','weighted_f1', 'weighted_f1_val','whole_f1', 'whole_f1_val', 'type_f1',
                               'polarity_f1', 'tense_f1','certainty_f1',
                               'type_f1_val', 'polarity_f1_val', 'tense_f1_val','certainty_f1_val']].iloc[index ,:].reset_index()
        Result['index'] = Result['index']+1
        Result.rename(columns = {'index' : 'epoch'}, inplace = True)
        Result['index']=self.exp
        Result['cv_k']=self.cv_k
        Result['model_txt'] = self.md_txt
        Result['model_img'] = self.md_img
        Result['batch_size']=self.b
        Result['max_length']=self.ml
        Result['learning_rate']=self.lr
        Result['L2_decay']=self.d
        Result['aug']=self.aug 
        Result['aug_num'] = self.aug_num
        Result = Result[['index','cv_k', 'model_txt', 'model_img','batch_size','max_length', 'learning_rate','L2_decay','aug', 'aug_num', 'epoch','loss', 'val_loss', 'weighted_f1', 'weighted_f1_val', 'whole_f1', 'whole_f1_val', 'type_f1', 'polarity_f1', 'tense_f1','certainty_f1', 'type_f1_val', 'polarity_f1_val', 'tense_f1_val','certainty_f1_val']]
        
        if not ('Result.csv' in os.listdir(self.file)):
            Result.to_csv(os.path.join(self.file,'Result.csv'))
        else:
            Result_0 = pd.read_csv(os.path.join(self.file,'Result.csv'), index_col=0)
            Result = pd.concat([Result_0, Result],axis=0)
            Result.to_csv(os.path.join(self.file,'Result.csv'))
                    
    def find_best_epoch(self) : 
        targetPattern = r"{}/*.pt".format(self.file_k_path)
        h5_list = glob.glob(targetPattern)
        f1_list = []
        for h5 in h5_list:
            f1_list.append(float(h5.split('-')[-1][:5])) # validation 기준
        self.best_model_path = h5_list[(np.argmax(np.array(f1_list)))]
        self.max_epoch = int((self.best_model_path.split('-')[-4].split('_'))[-1])
        return np.max(np.array(f1_list))       
    

class inference : 
    def __init__(self, model, tokenizer, extractor, encoders, file_path, cv_k, ml, test_path, val_dataloader, model_path, max_epoch, mode, img_root) :
        self.mode = mode
        self.encoders = encoders
        self.cv_k = cv_k
        self.dataframe_test = pd.read_csv(test_path).iloc[:,1:]
        self.file_path = file_path
        self.file_k_path = os.path.join(self.file_path, str(cv_k))
        self.model_path = model_path
        self.max_epoch  = max_epoch
        self.test_df = pd.read_csv(test_path)
        self.target_list = ['유형', '극성', '시제', '확실성']
        fake_list = ['사실형', '긍정', '현재', '확실']
        for target, fake in zip(self.target_list, fake_list) :
            self.test_df[target] = fake
            
        self.sub_df = pd.read_csv('Data/sample_submission.csv')
        
        if torch.cuda.is_available() == True :
            self.device = torch.device("cuda")
        else :
            self.device = torch.device("cpu")

        self.model = model
        self.tokenizer = tokenizer

        self.model.load_state_dict(torch.load(self.model_path)) 
            
        if self.mode == 'text' :
            test_dataset = txt_Dataset(self.test_df, tokenizer, encoders, ml, False)

        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers = 0)
        self.predict()

    def predict(self) :
        pred_tests, self.act_vec_tests = infer(self.model, self.test_loader, self.device, self.mode) 
        pred_lists = []
        for i in range(4) :
            #pred_list[i] = self.encoders[i].inverse_transform(pred_tests[i].cpu().numpy())
            self.test_df[self.target_list[i]] = self.encoders[i].inverse_transform(pred_tests[i]) 
            #pred_lists.append(pred_list)
        self.test_df.to_csv(os.path.join(self.file_path, 'test_cv_{}.csv'.format(self.cv_k)), index = False)
        self.test_df['label'] = self.test_df.apply(lambda x : x['유형']+'-'+x['극성']+'-'+x['시제']+'-'+x['확실성'], axis =1)
        
        self.sub_df['label'] =self.test_df['label']
        self.sub_df.to_csv(os.path.join(self.file_path, 'sub_cv_{}.csv'.format(self.cv_k)), index = False)
        
    def pred_val(self, val_df, val_dataloader) :
        _, pred_val, _ = infer(self.model, val_dataloader, self.device, self.mode) 
        pred_val_list = self.encoder.inverse_transform(pred_val.cpu().numpy())
        val_df['cat3_pred'] = pred_val_list
        val_df = val_df[val_df['cat3'] !=val_df['cat3_pred']]
        return val_df
        

        
class grid_search :
    def __init__(self, train_path, test_path, img_root, file, cv_k_list, model_txt_list, model_img_list, batch_size_list, lr_list, 
                 max_length_list, decay_list, aug_list, max_epoch, interval, train, initial_exp, start_exp, mode) :
        self.img_root = img_root
        self.mode = mode
        self.train_path = train_path
        self.test_path =test_path
        self.file = file
        self.cv_k_list = cv_k_list
        self.model_txt_list = model_txt_list
        self.model_img_list = model_img_list
        self.batch_size_list = batch_size_list
        self.lr_list = lr_list
        self.max_length_list = max_length_list 
        self.decay_list = decay_list
        self.aug_list = aug_list

        self.max_epoch = max_epoch
        self.save_interval = interval
        self.train = train
        self.exp = initial_exp
        self.start_exp = start_exp
        self.run()

    def run(self) :
        for md_txt in self.model_txt_list :
            for md_img in self.model_img_list :
                for b in self.batch_size_list :
                    for lr in self.lr_list : 
                        for ml in self.max_length_list :
                            for d in self.decay_list :
                                for aug in self.aug_list :

                                    self.Result = dict()
                                    self.Result_list = []
                                    self.Result['index']=self.exp
                                    self.Result['mode'] = self.mode
                                    self.Result['model_txt'] = md_txt
                                    self.Result['model_img'] = md_img
                                    self.Result['batch_size']=b
                                    self.Result['max_length']=ml
                                    self.Result['learning_rate']=lr
                                    self.Result['L2_decay']=d
                                    self.Result['aug']=aug

                                    condition = '{}_md_txt({})md_img({})b({})L({})ml({})d({})a({})'.format(self.exp, md_txt, md_img, b, lr, ml, d, aug)
                                    file_path = os.path.join(self.file, condition)
                                    if self.exp < self.start_exp :
                                        print('{} passed'.format(condition))
                                        self.exp = self.exp + 1
                                        pass
                                    else :
                                        if b * ml > 81920000000000 :
                                            print('To large... {} passed'.format(condition))
                                            self.exp = self.exp + 1
                                            pass
                                        else :

                                            for cv_k in self.cv_k_list :

#                                                     try :
                                                print(condition)
                                                self.my_bert = fine_tuning (self.img_root, self.exp, self.file, condition, cv_k,
                                                                            md_txt, md_img, b, lr, ml, d, self.save_interval,
                                                                            self.train_path, batch_size_test = 16,
                                                                            max_epoch = self.max_epoch, early_stopping =None, aug_num = 0, aug = aug, mode = self.mode)

                                                if self.train == True :
                                                    if 'history.csv' in os.listdir(self.my_bert.file_k_path) :
                                                        print('Already trained, pass!!!')
                                                    else:
                                                        self.my_bert.train()
                                                        self.my_bert.save_history()
                                                else :
                                                    pass
                                                max_f1 = self.my_bert.find_best_epoch()
                                                self.Result['f1_cv_{}'.format(cv_k)] = max_f1



                                                self.my_bert_inf = inference(self.my_bert.model, self.my_bert.tokenizer, self.my_bert.extractor, self.my_bert.encoders, file_path, cv_k, ml, self.test_path,
                                                                             self.my_bert.val_loader, self.my_bert.best_model_path,
                                                                             self.my_bert.max_epoch, self.mode, self.img_root)


                                                if cv_k == 0 :
                                                    act_vec_ensses = self.my_bert_inf.act_vec_tests
                                                else :
                                                    for ens_i in range(len(act_vec_ensses)) :
                                                        act_vec_ensses[ens_i] += self.my_bert_inf.act_vec_tests[ens_i]


                                            for ens_i in range(len(act_vec_ensses)) :
                                                act_vec_ensses[ens_i] = act_vec_ensses[ens_i] / 4
                                                pred_enss = np.argmax(act_vec_ensses[ens_i], 1)
                                                self.my_bert_inf.test_df[self.my_bert_inf.target_list[ens_i]] = self.my_bert_inf.encoders[ens_i].inverse_transform(pred_enss)
                                            self.my_bert_inf.sub_df['label'] = self.my_bert_inf.test_df.apply(lambda x : x['유형']+'-'+x['극성']+'-'+x['시제']+'-'+x['확실성'], axis =1)

                                            self.my_bert_inf.sub_df.to_csv(os.path.join(self.my_bert_inf.file_path, 'sub_cv_enss.csv'.format(cv_k)), index = False)

                                            self.Summary = pd.DataFrame([self.Result])
                                            if 'Summary_cv.xlsx' in os.listdir(self.file) :
                                                Summary_0 = pd.read_excel(os.path.join(self.file, 'Summary_cv.xlsx'), engine='openpyxl')
                                                self.Summary = pd.concat((Summary_0, self.Summary))
                                                self.Summary = self.Summary.drop_duplicates('index', keep='last')
                                                self.Summary.to_excel(os.path.join(self.file, 'Summary_cv.xlsx'), index = False)
                                            else :
                                                self.Summary.to_excel(os.path.join(self.file, 'Summary_cv.xlsx'), index = False)



                                            self.exp = self.exp + 1

                                            gc.collect()
                                            torch.cuda.reset_max_memory_allocated(device=torch.device("cuda"))
                                            torch.cuda.empty_cache()


