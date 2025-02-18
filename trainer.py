import torch
import sys

#sys.path.append('/home/beto/ISC/YOLO/TRACK/track/ultralytics-main')

from ultralytics import YOLO



def train_model():
    # Load the YOLO model with pretrained weights
    model = YOLO('./yolo11m-seg.pt')

    # Train the model with updated parameters for small datasets
    output = model.train(
        cfg='hyp.yaml',
        data='data.yaml',  
        epochs=30,         
        batch=64,           
        device="0,1",          
        imgsz=640,         
        pretrained=True,   
        patience=5,        
        augment=True,       
    )
    return output


if __name__ == '__main__':
    output = train_model()
