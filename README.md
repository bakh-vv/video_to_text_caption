# Video to text caption
Generating text captions based on a video. 

## Model architecture:            
Two models are stacked on top of each other. The first model is a pre-trained EfficientNet that extracts features from every frame of the video. 1 frame per second is kept.         

The second model is a sequence to sequence model that takes in the CNN features and outputs the captions. Second model architecture:              
<img src="media/model.png"/>

## Examples:                 
<img src="media/ezgif.com-gif-maker(17).gif"/>
<img src="media/ezgif.com-gif-maker(18).gif"/>
<img src="media/ezgif.com-gif-maker(19).gif"/>
<img src="media/ezgif.com-gif-maker(20).gif"/>
<img src="media/ezgif.com-gif-maker(15).gif"/>
<img src="media/ezgif.com-gif-maker(16).gif"/>

Based on the [Sequence to Sequence -- Video to Text (Venugopalan et al.)](https://arxiv.org/abs/1505.00487) paper with following modifications:
- Added pre-trained GloVe embeddings
- Used a newer pre-trained CNN model ([EfficientNet](https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1))
- Implemented beam search


## Data:
MSVD dataset


