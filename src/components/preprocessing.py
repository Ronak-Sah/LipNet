import os
import contextlib
import warnings
warnings.filterwarnings('ignore')

@contextlib.contextmanager
def deep_suppress():
    """Redirects stdout/stderr at the OS level (file descriptor)."""
    # Open devnull
    devnull = os.open(os.devnull, os.O_RDWR)
    # Duplicate existing stdout/stderr so we can restore them later
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    try:
        # Replace stdout/stderr with devnull
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Restore original descriptors
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        # Close duplicates
        os.close(devnull)
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)


import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch 


class Tokenizer():
    def __init__(self):
        self.vocab = list("abcdefghijklmnopqrstuvwxyz1234567890!ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.vocab)} 
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
    def text_to_labels(self,text):
        output=[]
        for c in text:
            output.append(self.char_to_idx[c])
        return output
    
    def labels_to_text(self,tokens):
        words=[]
        for t in tokens:
            words.append(self.idx_to_char[t])

        return "".join(words)



def decode_align(path):
    words=[]
    with open(path,'r') as fobj:
        for line in fobj:
            _,_,word=line.strip().split()
            if word!= "sil":
                words.append(word)
    return " ".join(words)


class Frames():
    def __init__(self,model_path):
        base_options = python.BaseOptions(
            model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1
        )
        with deep_suppress():
            self.detector = vision.FaceLandmarker.create_from_options(options)


        # MediaPipe lip landmark indices
        self.LIPS = [
            61, 146, 91, 181, 84, 17, 314, 405,
            321, 375, 291, 308, 324, 318,
            402, 317, 14, 87, 178, 88
        ]
        self.FEAT_DIM = len(self.LIPS) * 3 

    def extract_mouth_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        H=60
        W=100
        mouth_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
        
            result = self.detector.detect(mp_image)
        
            if not result.face_landmarks:
                continue
        
            face_landmarks = result.face_landmarks[0]

            
            xs, ys = [], []
            for idx in self.LIPS:
                lm = face_landmarks[idx]
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))
    
            x_min, x_max = max(min(xs)-5,0), min(max(xs)+5, w)
            y_min, y_max = max(min(ys)-5,0), min(max(ys)+5, h)

            mouth_crop = frame[y_min:y_max, x_min:x_max]
            mouth_crop = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
            mouth_crop = cv2.resize(mouth_crop, (W, H))

    
            mouth_tensor = torch.tensor(mouth_crop, dtype=torch.float32) / 255.0
            # mouth_tensor = mouth_tensor.unsqueeze(0)  

            mouth_frames.append(mouth_tensor)
            
        
        cap.release()
        

        if len(mouth_frames) == 0:
            return None
        
        # print(type(mouth_frames))            
        # print(type(mouth_frames[0]))         
        # print(type(mouth_frames[0][0]))      
        # print(mouth_frames[0][0].shape)      

        return torch.stack(mouth_frames)



class Loader():
    def __init__(self):
        pass
    def load_data(self,X_path,y_path, alignment_data_path, speaker_data_path,landmark_path):

        X_all = []
        y_all = []

        for align_file, video_file in zip(X_path, y_path):
            alignment_path=os.path.join(alignment_data_path,align_file)
            speaker_path=os.path.join(speaker_data_path,video_file)
            
            frames=Frames(landmark_path)
            tokeniser=Tokenizer()

            text=decode_align(alignment_path)
            decode_text=tokeniser.text_to_labels(text)

                    
            frames=frames.extract_mouth_frames(speaker_path)
            if frames is not None:
                frames = torch.tensor(frames, dtype=torch.float32)
                
                X_all.append(frames)
                y_all.append(torch.tensor(decode_text, dtype=torch.long))   
        
        # print("Type of X_all:", type(X_all))
        # print("Length of batch:", len(X_all))

        # print("Type of first element:", type(X_all[0]))
        # print("Shape of first element:", X_all[0].shape)
        video_tensors = torch.nn.utils.rnn.pad_sequence(
            X_all, batch_first=True
        )   
        video_tensors = video_tensors.unsqueeze(1)
                    
        return video_tensors, y_all
        