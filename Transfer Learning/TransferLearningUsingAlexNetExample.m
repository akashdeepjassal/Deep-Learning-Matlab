%% Transfer Learning Using AlexNet
% This example shows how to fine-tune a pretrained AlexNet convolutional
% neural network to perform classification on a new collection of images.
%
% Transfer learning is commonly used in deep learning applications. You can
% take a pretrained network and use it as a starting point to learn a new
% task. Fine-tuning a network with transfer learning is usually much faster
% and easier than training a network with randomly initialized weights from
% scratch. You can quickly transfer learned features to a new task using a
% smaller number of training images.

%% Load Data
% Unzip and load the new images as an image datastore. |imageDatastore|
% automatically labels the images based on folder names and stores the data
% as an |ImageDatastore| object. An image datastore enables you to store
% large image data, including data that does not fit in memory, and
% efficiently read batches of images during training of a convolutional
% neural network.
unzip('MerchData.zip');
images = imageDatastore('MerchData',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%%
% Divide the data into training and validation data sets. Use 70% of the
% images for training and 30% for validation. |splitEachLabel| splits the
% |images| datastore into two new datastores.
[trainingImages,validationImages] = splitEachLabel(images,0.7,'randomized');

%%
% This very small data set now contains 55 training images and 20
% validation images. Display some sample images.
numTrainImages = numel(trainingImages.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainingImages,idx(i));
    imshow(I)
end

%% Load Pretrained Network
% Load the pretrained AlexNet neural network. If Neural Network Toolbox(TM)
% Model _for AlexNet Network_ is not installed, then the software provides
% a download link. AlexNet is trained on more than one million images and
% can classify images into 1000 object categories, such as keyboard, mouse,
% pencil, and many animals. As a result, the model has learned rich feature
% representations for a wide range of images.
net = alexnet;

%%
% Display the network architecture. The network has five convolutional
% layers and three fully connected layers.
net.Layers



%% Transfer Layers to New Network
% The last three layers of the pretrained network |net| are configured for
% 1000 classes. These three layers must be fine-tuned for the new
% classification problem. Extract all layers, except the last three, from
% the pretrained network.
layersTransfer = net.Layers(1:end-3);

%%
% Transfer the layers to the new classification task by replacing the last
% three layers with a fully connected layer, a softmax layer, and a
% classification output layer. Specify the options of the new fully
% connected layer according to the new data. Set the fully connected layer
% to have the same size as the number of classes in the new data. To learn
% faster in the new layers than in the transferred layers, increase the
% |WeightLearnRateFactor| and |BiasLearnRateFactor| values of the fully
% connected layer.
numClasses = numel(categories(trainingImages.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%%
% If the training images differ in size from the image input layer, then
% you must resize or crop the image data. In this example, the images are
% the same size as the input size of AlexNet, so you do not need to resize
% or crop the images.

%% Train Network
% Specify the training options. For transfer learning, keep the features
% from the early layers of the pretrained network (the transferred layer
% weights). Set |InitialLearnRate| to a small value to slow down learning
% in the transferred layers. In the previous step, you increased the
% learning rate factors for the fully connected layer to speed up learning
% in the new final layers. This combination of learning rate settings
% results in fast learning only in the new layers and slower learning in
% the other layers. When performing transfer learning, you do not need to
% train for as many epochs. An epoch is a full training cycle on the entire
% training data set. Specify the mini-batch size and validation data. The
% software validates the network every |ValidationFrequency| iterations
% during training, and automatically stops training if the validation loss
% stops improving. Validate the network once per epoch.
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch);

%%
% Train the network that consists of the transferred and new layers. By
% default, |trainNetwork| uses a GPU if one is available (requires Parallel
% Computing Toolbox(TM) and a CUDA-enabled GPU with compute capability 3.0
% or higher). Otherwise, it uses a CPU. You can also specify the execution
% environment by using the |'ExecutionEnvironment'| name-value pair
% argument of |trainingOptions|.
netTransfer = trainNetwork(trainingImages,layers,options);


%% Classify Validation Images
% Classify the validation images using the fine-tuned network.
predictedLabels = classify(netTransfer,validationImages);

%%
% Display four sample validation images with their predicted labels.
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(validationImages,idx(i));
    label = predictedLabels(idx(i));
    imshow(I)
    title(char(label))
end

%%
% Calculate the classification accuracy on the validation set. Accuracy is
% the fraction of labels that the network predicts correctly.
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels)

%%
% This trained network has high accuracy. If the accuracy is not high
% enough using transfer learning, then try feature extraction instead.