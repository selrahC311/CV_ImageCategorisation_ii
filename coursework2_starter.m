%% Step 0: Set up parameters, vlfeat, category list, and image paths.
tic
data_path = '../data/';

FEATURE = 'bag of sift grayscale';
% FEATURE = 'bag of sift colour';
% FEATURE = 'spatial pyramids grayscale';
% FEATURE = 'spatial pyramids colour';

% CLASSIFIER = 'nearest neighbor';
CLASSIFIER = 'support vector machine';

step = 2;
size_ = 4;

sp_level = 1;
LAMBDA = 0.0001;

vocab_size_bow_grayscale = 400;
vocab_size_bow_colour = 700;
vocab_size_sp_grayscale = 300;
vocab_size_sp_colour = 400;

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

    case 'bag of sift grayscale'
        fprintf('step: %d, size: %d, vocab size: %d \n', step, size_, vocab_size_bow_grayscale);
        vocab_path = "vocab_grayscale/vocab_" + vocab_size_bow_grayscale + ".mat";
        img_feats_path = "image_feats/bow_grayscale/step" + step + "/size" + size_+"/";
        img_feats_filename = "img_feat_vocab_" + vocab_size_bow_grayscale + ".mat";
        
        if ~exist(vocab_path, 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocab_grayscale(train_image_paths, vocab_size_bow_grayscale, step, size_); %Also allow for different sift parameters
            save(vocab_path, 'vocab')
        end 
        load(vocab_path)
       
        if exist(img_feats_path+img_feats_filename, 'file')
            load(img_feats_path+img_feats_filename)
        else
            fprintf('No existing image features found. Computing one from images\n')
            train_image_feats = get_bag_of_sifts_grayscale(train_image_paths, step, size_, vocab); %Allow for different sift parameters
            test_image_feats  = get_bag_of_sifts_grayscale(test_image_paths, step, size_, vocab);
        end
    
    case 'bag of sift colour'
        fprintf('step: %d, size: %d, vocab size: %d \n', step, size_, vocab_size_bow_colour);
        vocab_path = "vocab_colour/vocab_" + vocab_size_bow_colour + ".mat";
        img_feats_path = "image_feats/bow_colour/step" + step + "/size" + size_+"/";
        img_feats_filename = "img_feat_vocab_" + vocab_size_bow_colour + ".mat";

        if ~exist(vocab_path, 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocab_colour(train_image_paths, vocab_size_bow_colour, step, size_); %Also allow for different sift parameters
            save(vocab_path, 'vocab')
        end 
        load(vocab_path)

        if exist(img_feats_path+img_feats_filename, 'file')
            load(img_feats_path+img_feats_filename)
        else
            fprintf('No existing image features found. Computing one from images\n')
            train_image_feats = get_bag_of_sifts_colour(train_image_paths, step, size_, vocab); %Allow for different sift parameters
            test_image_feats  = get_bag_of_sifts_colour(test_image_paths, step, size_, vocab); 
        end
    
    case 'spatial pyramids grayscale'
        fprintf('step: %d, size: %d, vocab size: %d \n', step, size_, vocab_size_sp_grayscale)
        vocab_path = "vocab_grayscale/vocab_" + vocab_size_sp_grayscale + ".mat";
        img_feats_path = "image_feats/sp_grayscale/step" + step + "/size" + size_ +"/";
        img_feats_filename = "img_feat_vocab_" + vocab_size_sp_grayscale + "spLevel_"+sp_level+".mat";

        if ~exist(vocab_path, 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocab_grayscale(train_image_paths, vocab_size_sp_grayscale, step, size_); %Also allow for different sift parameters             
            save(vocab_path, 'vocab')
        end 
        load(vocab_path)

        if exist(img_feats_path+img_feats_filename, 'file')
            load(img_feats_path+img_feats_filename)
        else
            fprintf('No existing image features found. Computing one from images\n')
            train_image_feats = get_spatial_pyramid_sifts_grayscale(train_image_paths, sp_level, step, size_, vocab); %Allow for different sift parameters
            test_image_feats  = get_spatial_pyramid_sifts_grayscale(test_image_paths, sp_level, step, size_, vocab); 
        end

    case 'spatial pyramids colour'
        fprintf('step: %d, size: %d, vocab size: %d \n', step, size_, vocab_size_sp_colour)
        vocab_path = "vocab_colour/vocab_" + vocab_size_sp_colour + ".mat";
        img_feats_path = "image_feats/sp_colour/step" + step + "/size" + size_ +"/";
        img_feats_filename = "img_feat_vocab_" + vocab_size_sp_colour + "spLevel_"+sp_level+".mat";

        if ~exist(vocab_path, 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocab_colour(train_image_paths, vocab_size_sp_colour, step, size_); %Also allow for different sift parameters
            save(vocab_path, 'vocab')
        end 
        load(vocab_path)

        if exist(img_feats_path+img_feats_filename, 'file')
            load(img_feats_path+img_feats_filename)
        else
            fprintf('No existing image features found. Computing one from images\n')
            train_image_feats = get_spatial_pyramid_sifts_colour_gpu(train_image_paths, sp_level, step, size_, vocab); %Allow for different sift parameters
            test_image_feats  = get_spatial_pyramid_sifts_colour_gpu(test_image_paths, sp_level, step, size_, vocab); 
        end    

end

mkdir(img_feats_path)
save(img_feats_path+img_feats_filename, 'train_image_feats', 'test_image_feats')

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
        predicted_categories = nearest_neighbor_classify(3, train_image_feats, train_labels, test_image_feats);
    case 'support vector machine'
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, LAMBDA);

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
toc