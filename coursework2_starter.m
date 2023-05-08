%% Step 0: Set up parameters, vlfeat, category list, and image paths.

FEATURE = 'bag of sift grayscale';
% FEATURE = 'bag of sift colour';
step = 3;
size = 8;

vocab_size = 1000; % you need to test the influence of this parameter

% FEATURE = 'spatial pyramids sift grayscale'
% FEATURE = 'spatial pyramids sift colour'

CLASSIFIER = 'nearest neighbor';
% CLASSIFIER = 'support vector machine';

data_path = 'data/';

categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};

abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};
    
num_train_per_cat = 100; 

fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);
%   train_image_paths  1500x1   cell      
%   test_image_paths   1500x1   cell           
%   train_labels       1500x1   cell         
%   test_labels        1500x1   cell          

%% Step 1: Represent each image with the appropriate feature

fprintf('Using %s representation for images\n', FEATURE)

switch lower(FEATURE)    
    case 'bag of sift grayscale'
        % YOU CODE build_vocabulary.m
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocabulary(train_image_paths, vocab_size); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end 
        % YOU CODE get_bags_of_sifts.m
        if ~exist('image_feats.mat', 'file')
            train_image_feats = get_bag_of_sifts_grayscale(train_image_paths, step, size); %Allow for different sift parameters
            test_image_feats  = get_bag_of_sifts_grayscale(test_image_paths, step, size); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end
    
    case 'bag of sift colour'
        % YOU CODE build_vocabulary.m
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab_size = 50; % you need to test the influence of this parameter
            vocab = build_vocabulary(train_image_paths, vocab_size); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end 
        % YOU CODE get_bags_of_sifts.m
        if ~exist('image_feats.mat', 'file')
            train_image_feats = get_bag_of_sifts_colour(train_image_paths, step, size); %Allow for different sift parameters
            test_image_feats  = get_bag_of_sifts_colour(test_image_paths, step, size); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end
    
    case 'spatial pyramids grayscale'
        % YOU CODE build_vocabulary.m
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab_size = 50; % you need to test the influence of this parameter
            vocab = build_vocabulary(train_image_paths, vocab_size); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end 
        % YOU CODE get_bags_of_sifts.m
        if ~exist('image_feats.mat', 'file')
            train_image_feats = get_spatial_pyramid_grayscale(train_image_paths, step, size); %Allow for different sift parameters
            test_image_feats  = get_spatial_pyramid_grayscale(test_image_paths, step, size); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end

    case 'spatial pyramids colour'
       % YOU CODE build_vocabulary.m
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab_size = 50; % you need to test the influence of this parameter
            vocab = build_vocabulary(train_image_paths, vocab_size); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end 
        % YOU CODE get_bags_of_sifts.m
        if ~exist('image_feats.mat', 'file')
            train_image_feats = get_spatial_pyramid_colour(train_image_paths, step, size); %Allow for different sift parameters
            test_image_feats  = get_spatial_pyramid_colour(test_image_paths, step, size); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end
end
%% Step 2: Classify each test image by training and using the appropriate classifier
fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER)

switch lower(CLASSIFIER)    
    case 'nearest neighbor'
        % Useful functions: pdist2 (Matlab) and vl_alldist2 (from vlFeat toolbox)
        predicted_categories = nearest_neighbor_classify(1, train_image_feats, train_labels, test_image_feats);
    case 'support vector machine'
        % TODO svm
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats);
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
