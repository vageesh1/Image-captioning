# Image-captioning
It is a project that I completed for a DL course project in my undergraduate.
The motive of this project is to make an end-to-end project that takes the input as an image and returns the image's caption. 
It is deployed on HuggingFace Spaces using Streamlit.

## Models Used
-- **(Inception-LSTM)**- The Most basic model we have used is the Inception Encoder and LSTM decoder, and then their seq2seq model is created, which returns the generated caption in the seq2seq model.
-- **(Attention Decoder)** - The second model we used improved the first one in which we applied an attention decoder with resnet as the image encoder, first encoding the given image. Then, attention is
applied to the decoder, and a seq2seq framework is made, which returns the generated caption.
-- **(Clip Captioning)**- CLIP Prefix for Image Captioning model method produces a prefix for each caption by applying a mapping network over the CLIP embedding. It is trained over many images and textual
descriptions using a contrastive loss, correlating visual and textual representations well. We noticed that the other models had pretty high bias related to gender. Hence, we tried to calculate the misclassification rate for this model in order to understand bias mitigation.
-- **(VIT-GPT2)** - This is the fourth model, which consists of VIT from hugging face transformers for the image feature extraction, and then GPT-2 is used for the text encoding and processing of the sequential input of image and text. The final layers of these models are fine-tuned on the A3DS dataset. 

## Methodology Steps- 
-- The models used for applying the image captioning are Attention Decoder, The CLIP captioning framework, and VIT+GPT2 captioner.
-- The Attention decoder model is trained from scratch using the Flickr 8k dataset, the CLIP captioning model is pre-trained, and the VIT+GPT2 model is fine-tuned on the A3DS dataset.
-- For the UI development, we used the streamlit Python framework, which uses Python code to create interactive applications.
-- And for deployment purposes, we have hugging face spaces that provide CPU-free CPU.
-- In the UI, we have made an image uploader that takes images as the input, applies all the necessary functions for the images passed through their respective models, and displays the input image with the predicted caption. The inference time of all this process is around 30 seconds- 1min.

## Results Comparision
-- Pre Trained Model(CLIP results)
![Alt Text](https://github.com/vageesh1/Image-captioning/blob/main/Pre%20Trained%20Result.png)
-- Fine Tuned Model(VIT+GPT-2)
![Alt Text](https://github.com/vageesh1/Image-captioning/blob/main/Fine%20Tuned%20Result.png)
-- Trained Model(Attention Decoder)
![Alt Text](https://github.com/vageesh1/Image-captioning/blob/main/Trained%20Result.png)

## Observation and Analysis
-- We got the best results for the pre-trained model based on the CLIP models. The VIT+GPT2 model didn't perform well on our image data.
-- Attention decoder Model also performed well on the dataset as it was trained on a generalized dataset of flickr8k with attention applied with the decoder.
-- The CLIP model worked pretty well on the test images as it uses a contrastive loss-making visual and textual representations well correlated
-- We successfully hosted all the models using the hugging face spaces by using the framework Streamlit
-- We can see clear instances of bias in both model 1 and model 2. We confirmed the same using the metric of misclassification rate and observed that the bias was significantly reduced in model 3(misclassification rate of 0.03 as compared to 0.37 and 0.43 in model 2 and model 1 respectively)

## Files Description
-- **(Neural Net)**- The Neural Net Folder contains all the necessary files for training, dataset loading, and inferencing for our Attention Decoder Model. 
-- **(NoteBooks)**- This folder contains the Jupyter notebooks, which consists of end-to-end pipelines for different models like CLIP, VIT+GPT-2, and Attention Decoder. The training loops, experimentations, loading and saving of models, and inference functions.  
-- **(Test examples)**- This folder consists of sample images on which we tested our model. 
-- **(Model.py)**- The file for loading different models used while deploying on hugging face
-- **(app.py)**- Python file to make the final streamlit interface

## InterFace- 
-- The interface is made using the streamlit and deployed on hugging face spaces
-- Here is the link to deployment-![Deployment Link](https://huggingface.co/spaces/Vageesh1/clip_gpt2)
-- The screenshot from Deployment- 
![Alt Text](https://github.com/vageesh1/Image-captioning/blob/main/Deployment%20image.png)

## Conclusion And Future Works
-- At a high level, our work more closely unifies tasks involving visual and linguistic
I understand with recent progress in object detection. While this suggests
several directions for future research, the immediate benefits of our approach
maybe captured by simply replacing pre-trained CNN features with pre trained
bottom-up attention features.
-- As an initial attempt to approach the potential gender bias in captioning systems, we emphasize that utilizing gender prediction accuracy to quantify gender
bias is not enough; evaluation metrics that can effectively lead to high caption
quality and low gender bias are still lacking.
-- The other direction in which we could possibly work is incorporating pre-trained
models, to other challenging tasks, such as visual question answering or image-to-dD translation, through mapping networks.







