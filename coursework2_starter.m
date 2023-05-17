%% Step 0: Set up parameters, vlfeat, category list, and image paths.

step = 4;
size_ = 7;

vocab_size = 100; % you need to test the influence of this parameter

FEATURE = 'bag of sift grayscale';
% FEATURE = 'bag of sift colour';
% FEATURE = 'spatial pyramids sift grayscale'
% FEATURE = 'spatial pyramids sift colour'

% CLASSIFIER = 'nearest neighbor';
CLASSIFIER = 'support vector machine';

LAMBDA = 0.000001;

data_path = '../data/';

%% getting categories and path

%This is the list of categories / directories to use. The categories are
%somewhat sorted by similarity so that the confusion matrix looks more
%structured (indoor and then urban and then rural).
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};
   
%This list of shortened category names is used later for visualization.
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};
    
%number of training examples per category to use. Max is 100. For
%simplicity, we assume this is the number of test cases per category, as
%well.
num_train_per_cat = 100; 

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);
%   train_image_paths  1500x1   cell      
%   test_image_paths   1500x1   cell           
%   train_labels       1500x1   cell         
%   test_labels        1500x1   cell          

%% Step 1: Represent each image with the appropriate feature
% Each function to construct features should return an N x d matrix, where
% N is the number of paths passed to the function and d is the 
% dimensionality of each image representation. See the starter code for
% each function for more details.

fprintf('Using %s representation for images\n', FEATURE)

switch lower(FEATURE)
    
    case 'tiny image'     
        train_image_feats = get_tiny_images(train_image_paths);
        test_image_feats  = get_tiny_images(test_image_paths);

    case 'colour histogram'
        train_image_feats = get_colour_histograms(train_image_paths);
        test_image_feats  = get_colour_histograms(test_image_paths);

    case 'bag of sift grayscale'
        % build vocab
        if exist('vocab.mat', 'file')
            load('vocab.mat')
        else
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocab_grayscale(train_image_paths, vocab_size, step, size_); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end
        % bags of sifts
        if exist('image_feats.mat', 'file')
            load('image_feats.mat')
        else
            train_image_feats = get_bag_of_sifts_grayscale(train_image_paths, step, size_); %Allow for different sift parameters
            test_image_feats  = get_bag_of_sifts_grayscale(test_image_paths, step, size_); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end
    
    case 'bag of sift colour'
        % build vocab
        if exist('vocab.mat', 'file')
            load('vocab.mat')
        else
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab_size = 50; % you need to test the influence of this parameter
            vocab = build_vocab_colour(train_image_paths, vocab_size, step, size_); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end 
        % bags of sifts
        if exist('image_feats.mat', 'file')
            load('image_feats.mat')
        else
            train_image_feats = get_bag_of_sifts_colour(train_image_paths, step, size_); %Allow for different sift parameters
            test_image_feats  = get_bag_of_sifts_colour(test_image_paths, step, size_); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end
    
    case 'spatial pyramids grayscale'
        % build vocab
        if exist('vocab.mat', 'file')
            load('vocab.mat')
        else
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab_size = 50; % you need to test the influence of this parameter
            vocab = build_vocabulary(train_image_paths, vocab_size, step, size_); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end 
        % bags of sifts
        if exist('image_feats.mat', 'file')
            load('image_feats.mat')
        else
            train_image_feats = get_spatial_pyramid_grayscale(train_image_paths, step, size_); %Allow for different sift parameters
            test_image_feats  = get_spatial_pyramid_grayscale(test_image_paths, step, size_); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end

    case 'spatial pyramids colour'
       % build vocab
        if exist('vocab.mat', 'file')
            load('vocab.mat')
        else
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab_size = 50; % you need to test the influence of this parameter
            vocab = build_vocabulary(train_image_paths, vocab_size, step, size_); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end 
        % YOU CODE get_bags_of_sifts.m
        if exist('image_feats.mat', 'file')
            load('vocab.mat')
        else
            train_image_feats = get_spatial_pyramid_colour(train_image_paths, step, size_); %Allow for different sift parameters
            test_image_feats  = get_spatial_pyramid_colour(test_image_paths, step, size_); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end

end
%% Step 2: Classify each test image by training and using the appropriate classifier
% Each function to classify test features will return an N x 1 cell array,
% where N is the number of test cases and each entry is a string indicating
% the predicted category for each test image. Each entry in
% 'predicted_categories' must be one of the 15 strings in 'categories',
% 'train_labels', and 'test_labels'. See the starter code for each function
% for more details.

fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER)
switch lower(CLASSIFIER)    
    case 'nearest neighbor'
        predicted_categories = knn_classifier(1, train_image_feats, train_labels, test_image_feats);
    case 'support vector machine'
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, LAMBDA);
%         predicted_categories = svm_classify_modified(train_image_feats, train_labels, test_image_feats, LAMBDA);
end

%% Step 3: Build a confusion matrix and score the recognition system
% You do not need to code anything in this section. 

% This function will recreate results_webpage/index.html and various image
% thumbnails each time it is called. View the webpage to help interpret
% your classifier performance. Where is it making mistakes? Are the
% confusions reasonable?
create_results_webpage( train_image_paths, ...
                        test_image_paths, ...
                        train_labels, ...
                        test_labels, ...
                        categories, ...
                        abbr_categories, ...
                        predicted_categories)
