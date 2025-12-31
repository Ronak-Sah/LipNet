import os
from src.logger import logger
from src.entity import ModelEvaluationConfig
import torch
import torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
from src.components.model.transformer import Transformer
from src.components.preprocessing import Tokenizer,Loader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    videos, labels = zip(*batch)
    
    videos_padded = pad_sequence(videos, batch_first=True, padding_value=0.0)

    # input_lengths = torch.tensor([v.size(0) for v in videos], dtype=torch.long)
    # target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    
    # eps = 1e-5
    # mean = videos_padded.mean(dim=(1, 2, 3, 4), keepdim=True)
    # std = videos_padded.std(dim=(1, 2, 3, 4), keepdim=True)
    # videos_padded = (videos_padded - mean) / (std + eps)

    return videos_padded, list(labels)
    

class Cnn_Dataset(Dataset):
    def __init__(self, X_path, y_path,  config,limit=100):
        self.X_path = X_path[500:limit+500]
        self.y_path = y_path[500:limit+500]
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
        # if isinstance(X, torch.Tensor):
        #     X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)
        return X, y




tokenizer=Tokenizer()
vocab_len=len(tokenizer.idx_to_char)


import torch

def ctc_greedy_decode(logits, blank=0):

    preds = torch.argmax(logits, dim=-1)  # (B, T)

    decoded = []
    for seq in preds:
        prev = blank
        out = []
        for token in seq:
            token = token.item()
            if token != prev and token != blank:
                out.append(token)
            prev = token
        decoded.append(out)

    return decoded


def word_error_rate(ref, hyp):
    """
    ref: ground truth string
    hyp: predicted string
    """
    ref_words = ref.split()
    hyp_words = hyp.split()

    n = len(ref_words)
    m = len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,    # deletion
                    dp[i][j - 1] + 1,    # insertion
                    dp[i - 1][j - 1] + 1 # substitution
                )

    return dp[n][m] / max(1, n)



class Model_Evaluation:
    def __init__(self,config: ModelEvaluationConfig,):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Transformer(
            ffn_hidden=self.config.ffn_hidden,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_length=self.config.max_length,
            vocab_size=vocab_len
        ).to(self.device)

        model_path = config.model_path
        checkpoint_path = os.path.join("artifacts\model_trainer\checkpoint.pth")
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])

        logger.info("Model loaded succesfully for evaluation")

    
    def evaluate(self):
        
        self.model.eval()
        total_wer = 0.0
        count = 0
        batch_no=0

        alignment_path=os.path.join(self.config.alignment_data_path)
        speaker_path=os.path.join(self.config.speaker_data_path)

        X_paths=os.listdir(speaker_path)
        y_paths=os.listdir(alignment_path)

        scripted_model = torch.jit.script(self.model)
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        scripted_model.save(os.path.join(self.config.root_dir,"lip_reading.pt"))


        dataset = Cnn_Dataset(X_paths, y_paths,self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,  
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
            # persistent_workers=True
        )

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                total_batches = len(dataloader)
                batch_no=batch_no+1
                rem=int(total_batches) - batch_no
                print(f"Batch no : {batch_no}, Total batch : {total_batches}, Remaining batch :{rem}" )
                # X,y=loader.load_data(X_batch, y_batch, self.config.alignment_data_path,self.config.speaker_data_path,self.config.landmark_model_path)
                
                X = X_batch.to(self.device).permute(0, 2, 1, 3, 4)
                y = [t.to(self.device) for t in y_batch]


                y_pred=self.model(X)
                # y_pred = y_pred.permute(1, 0, 2)
 

                decoded_tokens = ctc_greedy_decode(y_pred)

                for i in range(len(decoded_tokens)):
                    pred_text = tokenizer.labels_to_text(decoded_tokens[i])
                    
                    gt_text = tokenizer.labels_to_text(y[i].tolist())
                    total_wer += word_error_rate(gt_text, pred_text)
                    count += 1
        print("Word error rate is :",total_wer / count)
        return total_wer / count
    


