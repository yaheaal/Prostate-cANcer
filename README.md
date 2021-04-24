# Files guid

# Table of contents
* [Overview](#overview)
* [Techniques I used](#techniques-i-used)
* [Tips to improve the accuracy](#tips-to-improve-the-accuracy)
* [Note about the score in this parameters.](#note-about-the-score-in-this-parameters)
* [Reference](#reference)

# Overview
### Problem Statement
With more than 1 million new diagnoses reported every year, prostate cancer (PCa) is the second most common cancer among males worldwide that results in more than 350,000 deaths annually. We will develop model for detecting PCa on images of prostate tissue samples, and estimate severity of the disease using the most extensive multi-center dataset on Gleason grading yet available.

For more [information](https://www.kaggle.com/c/prostate-cancer-grade-assessment).

### Problem Type & Data
The challenge in this competition is to **Classify** the severity of prostate cancer from microscopy scans of prostate biopsy samples. There are two unusual twists to this problem relative to most competitions:

* Each individual image is quite large. We're excited to see what strategies you come up with for efficiently locating areas of concern to zoom in on.
* The labels are imperfect. This is a challenging area of pathology and even experts in the field with years of experience do not always agree on how to interpret a slide. This will make training models more difficult, but increases the potential medical value of having a strong model to provide consistent ratings. All of the private test set images and most of the public test set images were graded by multiple pathologists, but this was not feasible for the training set. You can find additional details about how consistently the pathologist's labels matched [here](https://zenodo.org/record/3715938#.XpTU3PJKiUl).
### Problem solution
I used Pytorch framework to build a convolutional neural network, where I used **transfer learning** techniques.
Using some **image augmentation** and **TAA** (test-time augmentation) I was able to reach more than 90% accuracy.
You can find more details in my main [code](https://translate.google.com/?sl=en&tl=ar&op=translate) and in [Techniques I used](https://translate.google.com/?sl=en&tl=ar&op=translate).

### Evaluation
Submissions are scored based on the [Quadratic Weighted Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa), which measures the agreement between two outcomes. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement than expected by chance, the metric may go below 0.
# Techniques I used
### Tiles :
Using tiling method based on this [notebook](https://www.kaggle.com/iafoss/panda-16x128x128-tiles).
Where I used `tile_size = 256` `n_tiles = 16`.

### TTA :
Test-time augmentation, or TTA for short, is an application of data augmentation to the test dataset.
Specifically, it involves creating multiple augmented copies of each image in the test set, having the model make a prediction for each, then returning an ensemble of those predictions.

[Here](https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification) you can see how to use Test-Time Augmentation to Make Better Predictions.
### Ensemble learning :
Briefly Ensemble is training multiple models, where I traind 5 models on 5 different parts of the data, then for each test data there was 5 predictions wich I Merged them
together to get the last and the best prediction.
### EarlyStopping
I trained each fold for 5 epochs, so here we are just using early stopping the save the best parameters from these epochs.
### ReduceLROnPlateau
Simply we are just reducing the learning rate whenever the model is not improving.
# Tips to improve the accuracy
Note that I didn't use this tips due to **lack of time** and **computing power**.
### Reduce the image size :
.......
### Reduce the batch size :
.......
### Train for longer time (more epochs) : 
.......
### Find the best mix between the 5 models :
.......
### Use another technique to reduce the LR :
.......
### Try another augmention techniques : 
.......
 for example adding some noise to the images.

# Note about the score in this parameters.
# Reference
