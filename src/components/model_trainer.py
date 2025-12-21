import os
from src.logger import logger
from src.entity import ModelTrainerConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.components.model.transformer import Transformer
from src.components.preprocessing import Tokenizer,Loader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    videos, labels = zip(*batch)
    
    videos_padded = pad_sequence(videos, batch_first=True, padding_value=0.0)

    input_lengths = torch.tensor([v.size(0) for v in videos], dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    
    return videos_padded, list(labels), input_lengths, target_lengths

class Cnn_Dataset(Dataset):
    def __init__(self, X_path, y_path,  config,limit=500):
        self.X_path = X_path[:limit]
        self.y_path = y_path[:limit]
        self.config = config
        self.loader = Loader()
    def __len__(self):
        return len(self.X_path)

    def __getitem__(self, idx):
        y_file = self.y_path[idx]
        name_no_ext = os.path.splitext(y_file)[0] 
        x_file = f"{name_no_ext}.mpg"
        

        # print(x_file)
        # print(y_file)
        X, y = self.loader.load_data(
            x_file, y_file, 
            self.config.alignment_data_path, 
            self.config.speaker_data_path,
            self.config.landmark_model_path
        )
        return X, y

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
        

        
            

    def train(self):
        alignment_path = os.path.abspath(self.config.alignment_data_path)
        speaker_path = os.path.abspath(self.config.speaker_data_path)

        model_path = os.path.join(self.config.root_dir, "transformer_model.pth")
        checkpoint_path = os.path.join(self.config.root_dir, "checkpoint.pth")

        y_paths=os.listdir(alignment_path)
        X_paths=os.listdir(speaker_path)

        ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        # loader=Loader()

        dataset = Cnn_Dataset(X_paths, y_paths,self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,  
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )

        optimizer = torch.optim.AdamW(self.transformer.parameters(),lr=1e-4,weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.1,total_iters=500)


        start_epoch = 0
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.transformer.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1

            print(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, start_epoch + self.config.epochs):
            print("Running epoch no... ",epoch+1)
            self.transformer.train()
            total_loss = 0.0
            batch_no=0
            for X_batch, y_batch, input_lengths, target_lengths in dataloader:
                total_batches = len(dataloader)
                batch_no=batch_no+1
                rem=int(total_batches) - batch_no
                print(f"Batch no : {batch_no}, Total batch : {total_batches}, Remaining batch :{rem}" )
                
                X = X_batch.to(self.device).permute(0, 2, 1, 3, 4)
                y = [t.to(self.device) for t in y_batch]

                optimizer.zero_grad()

                y_pred=self.transformer(X)
                y_pred = y_pred.log_softmax(dim=-1)
                y_pred = y_pred.permute(1, 0, 2)

                # input_lengths = torch.full(
                #     size=(y_pred.size(1),),fill_value=y_pred.size(0),dtype=torch.long
                # )

                # target_lengths = torch.tensor(
                #     [len(t) for t in y],dtype=torch.long
                # )

                targets = torch.cat(y)   

                loss = ctc_loss(y_pred,targets,input_lengths,target_lengths)

            
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()


            torch.save({
                "epoch": epoch,
                "model_state": self.transformer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()
            }, checkpoint_path)
            print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {total_loss: .4f}")

        os.makedirs(self.config.root_dir, exist_ok=True)

        
       

        torch.save(self.transformer.state_dict(), model_path)


        logger.info(f"Model saved at: {model_path}")


    