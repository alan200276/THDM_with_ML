# THDM_with_ML


# Workflow 
* Data Generating and Preprocessing  
    
    > MC data comes from Madgraph5. 
    * process card can be found in [Cards](https://github.com/alan200276/THDM_with_ML/tree/main/Cards)

    > You can use the same environment by this docker image. 
    * Docker Image: [HEPtools](https://hub.docker.com/repository/docker/alan200276/ubuntu)

    > Preprocessing files are in [Preprocess](https://github.com/alan200276/THDM_with_ML/tree/main/Preprocess)
    1. create event list and reduce data size from ROOT format to h5  
        * create event list: [Event_List.py](https://github.com/alan200276/THDM_with_ML/blob/main/Preprocess/Event_List.py)  
        * reduce data size: [downsize.py](https://github.com/alan200276/THDM_with_ML/blob/main/Preprocess/downsize.py)
    2. preprocessing precedure: using downsized data to calculate high-level features and make images
        * [preprocess.py](https://github.com/alan200276/THDM_with_ML/blob/main/Preprocess/preprocess.py)  
        *This file relies on [JSS.py](https://github.com/alan200276/THDM_with_ML/blob/main/Preprocess/JSS.py) for jet substructures, [jet_trimming.py](https://github.com/alan200276/THDM_with_ML/blob/main/Preprocess/jet_trimming.py) for jet trimming, [make_jet_image.py](https://github.com/alan200276/THDM_with_ML/blob/main/Preprocess/make_jet_image.py) for making jet images and [make_event_image.py](https://github.com/alan200276/THDM_with_ML/blob/main/Preprocess/make_event_image.py) for making full-event images.

* Analysis(in [Notebook](https://github.com/alan200276/THDM_with_ML/tree/main/Notebook))
    * High-level feature distrubitions and cut-base method: [high_level_feature.py](https://github.com/alan200276/THDM_with_ML/blob/main/Notebook/high_level_feature.py)
    * BDT Method: [BDT_classifier.py](https://github.com/alan200276/THDM_with_ML/blob/main/Notebook/BDT_classifier.py)
    * 3CNN Method: [3CNN_training.py](https://github.com/alan200276/THDM_with_ML/blob/main/Notebook/3CNN_training.py)

* Figures
    * Low-level features images: [Jet_Event_Images.ipynb](https://github.com/alan200276/THDM_with_ML/blob/main/Notebook/Jet_Event_Images.ipynb)

* Trained Models (in [Model](https://github.com/alan200276/THDM_with_ML/tree/main/Model))
    * [BDT trained model](https://github.com/alan200276/THDM_with_ML/blob/main/Model/clf_GBDT_2000.h5)
    * [3CNN trained model](https://github.com/alan200276/THDM_with_ML/blob/main/Model/model_3cnn_500_norm.h5)
    * Prediction values are in [prediction](https://github.com/alan200276/THDM_with_ML/tree/main/Model/prediction)