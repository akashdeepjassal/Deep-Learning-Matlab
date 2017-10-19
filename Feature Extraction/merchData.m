function [dataTrain,dataTest] = merchData()

unzip(fullfile(matlabroot,'examples','nnet','MerchData.zip'));

data = imageDatastore('MerchData',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

[dataTrain,dataTest] = splitEachLabel(data,0.8);

dataTrain = shuffle(dataTrain);

end