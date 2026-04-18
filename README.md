Names         : Edvard Vindenes Steenslid and Morten Kvamme. 

Title         : Diagnose cardiovascular disease from electrocardiography (ECG) measurements

Description   : We want to develop a multi-label classification model that diagnones different types of cardiovascular disesase. We plan to use Grad-CAM to identify which part of the ECG was important for the classification. The metric for succsess are classification accuracy and recall with f1-score. 

Data          : The PTB-XL dataset, https://physionet.org/content/ptb-xl/1.0.3/. Suitable since it has 22k 10-second snippets of ECG. 

Models        : We will investigate a 1D-CNN and a LSTM

Demo website: https://huggingface.co/spaces/Steenslid/ecg-ptbxl-demo


This model was trained using the PTB-XL dataset (version 1.0.3), accessed via PhysioNet. The original dataset is licensed under the Creative Commons Attribution 4.0 International Public License (CC BY 4.0). Find the original dataset here: https://physionet.org/content/ptb-xl/1.0.3/ and the license here: https://physionet.org/content/ptb-xl/view-license/1.0.3/
