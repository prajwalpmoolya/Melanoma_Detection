# Melanoma-Detection
> In this assignment, I have build a multiclass classification model using a custom convolutional neural network in tensorflow.


## Table of Contents
* [Problem Statement](#problem-statement)
* [General Info](#general-information)
* [Technologies Used](#technologies-used)


<!-- You can include any other section that is pertinent to your problem -->

# Problem Statement
> - To build a multiclass classification model using a custom convolutional neural network in TensorFlow. To build a CNN based model which can accurately detect melanoma a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths.

## General Information
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

The data set contains the following diseases:
- Squamous cell carcinoma
- Dermatofibroma
- Melanoma
- Actinic keratosis
- Nevus
- Seborrheic keratosis
- Pigmented benign keratosis
- Vascular lesion
- Basal cell carcinoma

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- TensorFlow
- Keras
- Python 3
- Pandas 
- Numpy
- Matplotlib

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

# Model 1
Trained for 20 epochs with available data
> Insights:
> - The model clearly overfits.
> - The training accuracy is continuously increasing while the validation accuracy is around 50%.
> - The loss on training set decreases after each epoch but in case of the validation loss, its increasing after decreasing for few initial epochs.

# Model 2 with Data Augumentation
> Insights:
> - The application of data augmentation and dropout layer reduced overfitting.
> - Results on training and validation datasets are much closer.
> - Overall accuracy is not high.

# Model 3
> Insights:
> - Included normalization. 
> - Rectified class imbalances present in the training dataset with Augmentor library.

## References
- https://towardsdatascience.com/
- https://www.tensorflow.org/tutorials/images/data_augmentation
- https://augmentor.readthedocs.io/en/stable/
- https://keras.io/guides/
- https://www.analyticsvidhya.com/blog/2021/03/image-augmentation-techniques-for-training-deep-learning-models/


## Contact
Created by [@prajwalpmoolya] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
