import os
from src.logger import logger
from src.entity import ModelTrainerConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.components.model.transformer import Transformer
from src.components.preprocessing import Tokenizer,Loader




class Cnn_Dataset(Dataset):
    def __init__(self, X_path, y_path, limit=500):
        self.X_path = X_path[:limit]
        self.y_path = y_path[:limit]

    def __len__(self):
        return len(self.X_path)

    def __getitem__(self, idx):
        return self.X_path[idx], self.X_path[idx].split("\\")[-1][0:-5]+"mpg"


tokenizer=Tokenizer()
vocab_len=len(tokenizer.vocab)



class Model_Trainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config= config
        self.device= "cuda" if torch.cuda.is_available() else "cpu"

        self.transformer = Transformer(
            # emb_dim=self.config.emb_dim,
            ffn_hidden=self.config.ffn_hidden,
            num_heads=self.config.num_heads,
            # drop_prob=self.config.drop_prob,
            num_layers=self.config.num_layers,
            max_length=self.config.max_length,
            vocab_size=vocab_len
        ).to(self.device)
        model_path=os.path.join(self.config.root_dir,"transformer_model.pth")
        if os.path.exists(model_path):
           
            self.transformer.load_state_dict(torch.load(model_path, map_location=self.device))

        
            

    def train(self):
        alignment_path=os.path.join(self.config.alignment_data_path)
        speaker_path=os.path.join(self.config.speaker_data_path)

        X_paths=os.listdir(alignment_path)
        y_paths=os.listdir(speaker_path)

        ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        loader=Loader()

        dataset = Cnn_Dataset(X_paths, y_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,  
            shuffle=True
        )

        optimizer = torch.optim.AdamW(self.transformer.parameters(),lr=3e-4,weight_decay=1e-2)

        for epoch in range(self.config.epochs):
            print("Running epoch no : ",epoch+1)
            self.transformer.train()
            total_loss = 0.0
            batch_no=0
            for X_batch, y_batch in dataloader:
                batch_no=batch_no+1
                rem=(500//self.config.batch_size)-batch_no+1
                print(f"Batch no : {batch_no}, Total batch : {500/self.config.batch_size}, Remaining batch :{rem}" )
                X,y=loader.load_data(X_batch, y_batch, self.config.alignment_data_path,self.config.speaker_data_path,self.config.landmark_model_path)
                
                X = X.to(self.device)
                y = [t.to(self.device) for t in y]

                optimizer.zero_grad()

                y_pred=self.transformer(X)
                y_pred = y_pred.permute(1, 0, 2)
                y_pred = y_pred.log_softmax(dim=-1)

                input_lengths = torch.full(
                    size=(y_pred.size(1),),fill_value=y_pred.size(0),dtype=torch.long
                )

                target_lengths = torch.tensor(
                    [len(t) for t in y],dtype=torch.long
                )

                targets = torch.cat(y)   

                loss = ctc_loss(y_pred,targets,input_lengths,target_lengths)

            
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {total_loss: .4f}")

        os.makedirs(self.config.root_dir, exist_ok=True)

        model_save_path = os.path.join(self.config.root_dir, "transformer_model.pth")

        torch.save(self.transformer.state_dict(), model_save_path)

        logger.info(f"Model saved at: {model_save_path}")


    