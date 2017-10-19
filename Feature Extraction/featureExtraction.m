%% Feature Extraction Using AlexNet
% This example shows how to extract learned features from a pretrained
% convolutional neural network, and use those features to train an image
% classifier. Feature extraction is the easiest and fastest way use the
% representational power of pretrained deep networks. For example, you can
% train a support vector machine (SVM) using |fitcecoc| (Statistics and
% Machine Learning Toolbox(TM)) on the extracted features.

%% Load Data
% Unzip and load the sample images as an image datastore. |imageDatastore|
% automatically labels the images based on folder names and stores the data
% as an |ImageDatastore| object. An image datastore lets you store large
% image data, including data that does not fit in memory. Split the data
% into 70% training and 30% test data.
unzip('MerchData.zip');
images = imageDatastore('MerchData',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

[trainingImages,testImages] = splitEachLabel(images,0.7,'randomized');

%%
% There are now 55 training images and 20 validation images in this very
% small data set. Display some sample images.
numTrainImages = numel(trainingImages.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainingImages,idx(i));
    imshow(I)
end

%% Load Pretrained Network
% Load a pretrained AlexNet network. If Neural Network Toolbox Model _for
% AlexNet Network_ is not installed, then the software provides a download
% link. AlexNet is trained on more than a million images and can classify
% images into 1000 object categories. For example, keyboard, mouse, pencil,
% and many animals. As a result, the model has learned rich feature
% representations for a wide range of images.
net = alexnet;

%%
% Display the network architecture. The network has five convolutional
% layers and three fully connected layers.
net.Layers

%%
% If the training images differ in size from the image input layer, then
% you must resize or crop the image data. In this example, the images are
% the same size as the input size of AlexNet, so you do not need to resize
% or crop the images.

%% Extract Image Features
% The network constructs a hierarchical representation of input images.
% Deeper layers contain higher-level features, constructed using the
% lower-level features of earlier layers. To get the feature
% representations of the training and test images, use |activations| on the
% fully connected layer |'fc7'|. To get a lower-level representation of the
% images, use an earlier layer in the network.
layer = 'fc7';
trainingFeatures = activations(net,trainingImages,layer);
testFeatures = activations(net,testImages,layer);

%%
% Extract the class labels from the training and test data.
trainingLabels = trainingImages.Labels;
testLabels = testImages.Labels;

%% Fit Image Classifier
% Use the features extracted from the training images as predictor
% variables and fit a multiclass support vector machine (SVM) using
% |fitcecoc| (Statistics and Machine Learning Toolbox).
classifier = fitcecoc(trainingFeatures,trainingLabels);

%% Classify Test Images
% Classify the test images using the trained SVM model the features
% extracted from the test images.
predictedLabels = predict(classifier,testFeatures);

%%
% Display four sample test images with their predicted labels.
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(testImages,idx(i));
    label = predictedLabels(idx(i));
    imshow(I)
    title(char(label))
end

%%
% Calculate the classification accuracy on the test set. Accuracy is the
% fraction of labels that the network predicts correctly.
accuracy = mean(predictedLabels == testLabels)

%%
% This SVM has high accuracy. If the accuracy is not high enough using
% feature extraction, then try transfer learning instead.