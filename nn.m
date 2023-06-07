function [predictions] = nn(train_image_path, trainLabels, test_image_path, width, train_fraction, batch_size, learning_rate)

% % Resize the training images and save them 
% Create a directory to store the resized images
resizedDir_training = 'resized_images_training';
if ~exist(resizedDir_training, 'dir')
    mkdir(resizedDir_training);
end
resizedDir_validating = 'resized_images_validating';
if ~exist(resizedDir_validating, 'dir')
    mkdir(resizedDir_validating);
end
resizedDir_testing = 'resized_images_testing';
if ~exist(resizedDir_testing, 'dir')
    mkdir(resizedDir_testing);
end


% % --------- DELETE LATER -----------
% Split the dataset into training and validation sets
    categories = unique(trainLabels);
    num_categories = numel(categories);

    train_idx = [];
    val_idx = [];

    for i = 1:num_categories
        indices = find(strcmp(trainLabels, categories(i)));
        rand_idx = randperm(numel(indices));
        res = indices(rand_idx(1:floor(numel(indices)*train_fraction)));
        train_idx = [train_idx; res'];
        val_idx = [val_idx, setdiff(indices, train_idx)];
    end
    trainIdx = reshape(train_idx', [], 1);
    valIdx = reshape(val_idx, [], 1);

    train_feats = train_image_path(trainIdx, :);
    train_labels_ = trainLabels(trainIdx);
    val_feats = train_image_path(valIdx, :);
    val_labels = trainLabels(valIdx);

% Initialsie the cell array for the resized image paths
resized_image_paths_training = cell(numel(train_feats), 1);
resized_image_paths_validating = cell(numel(val_feats), 1);
resized_image_paths_testing = cell(numel(test_image_path), 1);
% Training
for i = 1:numel(train_feats)
    img = imread(train_feats{i});
    img = imresize(img, [width, width]);
    % Convert the image to double data type
    img = im2double(img);
    
    % Find the minimum and maximum pixel values
    minValue = min(img(:));
    maxValue = max(img(:));
    
    % Normalize the image
    normalizedImg = (img - minValue) / (maxValue - minValue);

    % Generate a unique filename for each resized image
    [~, imageName, extension] = fileparts(train_feats{i});
    resizedFilename = fullfile(resizedDir_training, [imageName, '_resized', extension]);
%     imgGray = rgb2gray(img);
   
    % Save the resized image
    imwrite(normalizedImg, resizedFilename);
    
    % Store the path of the resized image
    resized_image_paths_training{i} = resizedFilename;
end

% Validating
for i = 1:numel(val_feats)
    img = imread(val_feats{i});
    img = imresize(img, [width, width]);
    % Convert the image to double data type
    img = im2double(img);
    
    % Find the minimum and maximum pixel values
    minValue = min(img(:));
    maxValue = max(img(:));
    
    % Normalize the image
    normalizedImg = (img - minValue) / (maxValue - minValue);

    % Generate a unique filename for each resized image
    [~, imageName, extension] = fileparts(val_feats{i});
    resizedFilename = fullfile(resizedDir_validating, [imageName, '_resized', extension]);
%     imgGray = rgb2gray(img);
   
    % Save the resized image
    imwrite(normalizedImg, resizedFilename);
    
    % Store the path of the resized image
    resized_image_paths_validating{i} = resizedFilename;
end


% Testing
for i = 1:numel(test_image_path)
    img = imread(test_image_path{i});
    img = imresize(img, [width, width]);
    % Convert the image to double data type
    img = im2double(img);
    
    % Find the minimum and maximum pixel values
    minValue = min(img(:));
    maxValue = max(img(:));
    
    % Normalize the image
    normalizedImg = (img - minValue) / (maxValue - minValue);

    % Generate a unique filename for each resized image
    [~, imageName, extension] = fileparts(test_image_path{i});
    resizedFilename = fullfile(resizedDir_testing, [imageName, '_resized', extension]);
%     imgGray = rgb2gray(img);
   
    % Save the resized image
    imwrite(normalizedImg, resizedFilename);
    
    % Store the path of the resized image
    resized_image_paths_testing{i} = resizedFilename;
end

% Train data DataStore
trainData = imageDatastore(resized_image_paths_training);
trainData.Labels = categorical(train_labels_);
% Validate data DataStore
validateData = imageDatastore(resized_image_paths_validating);
validateData.Labels = categorical(val_labels);
% Test data DataStore
testData = imageDatastore(resized_image_paths_testing);

numFilters=75;
layers = [
    % image input layer whose input size is 32-32-1. "3" represents one
    % color channel (RGB image)
    imageInputLayer([width width 3]) 
    % convolutional layer whose filter size is 3 by 3 
    convolution2dLayer(3,numFilters,'Padding','same')
    % batch normalization layer
    batchNormalizationLayer
    % activation layer (relu)
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,numFilters*2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,numFilters*4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(500)
    fullyConnectedLayer(100)
    fullyConnectedLayer(15)
    softmaxLayer
    classificationLayer];

% Set up the training options
options = trainingOptions('adam', ...
    'MiniBatchSize',batch_size, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',learning_rate, ...
    'Shuffle','every-epoch', ...
    'L2Regularization',0.1, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ValidationData',validateData);


% Train the network
net = trainNetwork(trainData, layers, options);

% Evaluate the trained network
predictions = classify(net, testData);
predictions = cellstr(predictions);
% Compute accuracy
% accuracy = sum(predictions == trainLabels) / numel(trainLabels);
% fprintf('Test accuracy: %.2f%%\n', accuracy * 100);


% % % % ----------- Vary:- Batch size, learing rate, losses

end








