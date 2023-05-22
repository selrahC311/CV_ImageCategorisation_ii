%% Step 0: Set up parameters, vlfeat, category list, and image paths.

data_path = '../data/';

FEATURE = 'bag of sift grayscale';
% FEATURE = 'bag of sift colour';
% FEATURE = 'spatial pyramids grayscale';
% FEATURE = 'spatial pyramids colour';

% CLASSIFIER = 'nearest neighbor';
CLASSIFIER = 'support vector machine';

step = 3;
size_ = 3;

sp_level = 4;
vocab_size_bow_grayscale = 500;
vocab_size_bow_colour = 500;
vocab_size_sp_grayscale = 500;
vocab_size_sp_colour = 500;

vocab_path_bow_grayscale = "vocab_grayscale/vocab_" + vocab_size_bow_grayscale + ".mat";
vocab_path_bow_colour = "vocab_colour/vocab_" + vocab_size_bow_colour + ".mat";
vocab_path_sp_grayscale = "vocab_grayscale/vocab_" + vocab_size_sp_grayscale + ".mat";
vocab_path_sp_colour = "vocab_colour/vocab_" + vocab_size_sp_colour + ".mat";

img_feats_path_bow_grayscale = "image_feats/bow_grayscale/" + "step" ...
    + step + "/size" + size_ + "img_feat_vocab_" + vocab_size + ".mat";
img_feats_path_bow_colour = "image_feats/bow_colour/" + "step" + step ...
    + "/size" + size_ + "img_feat_vocab_" + vocab_size + ".mat";
img_feats_path_sp_grayscale = "image_feats/sp_grayscale/" + "step" + step ...
    + "/size" + size_ + "img_feat_vocab_" + vocab_size + ".mat";
img_feats_path_sp_colour = "image_feats/sp_colour/" + "step" + step + ...
    "/size" + size_ + "img_feat_vocab_" + vocab_size + ".mat";

LAMBDA = 0.000001;

%% getting categories and path

categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};

abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};
    
num_train_per_cat = 100; 

fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);        

%% Step 1: Represent each image with the appropriate feature

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
        if exist(vocab_path_bow_grayscale, 'file')
            load(vocab_path_bow_grayscale)
        else
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocab_grayscale(train_image_paths, vocab_size, step, size_); %Also allow for different sift parameters
        end
        % img feats
        if ~exist(img_feats_path_bow_grayscale, 'file')
            load(img_feats_path_bow_grayscale)
        else
            fprintf('No existing image features found. Computing one from images\n')
            train_image_feats = get_bag_of_sifts_grayscale(train_image_paths, step, size_, vocab); %Allow for different sift parameters
            test_image_feats  = get_bag_of_sifts_grayscale(test_image_paths, step, size_, vocab);
        end
    
    case 'bag of sift colour'
        % build vocab
        if exist(vocab_path_bow_colour, 'file')
            load(vocab_path_bow_colour)
        else
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocab_colour(train_image_paths, vocab_size, step, size_); %Also allow for different sift parameters
%             save(vocab_path_bow_colour, 'vocab')
        end 
        % bags of sifts
        if exist(img_feats_path_bow_colour, 'file')
            load(img_feats_path_bow_colour)
        else
            fprintf('No existing image features found. Computing one from images\n')
            train_image_feats = get_bag_of_sifts_colour(train_image_paths, step, size_, vocab); %Allow for different sift parameters
            test_image_feats  = get_bag_of_sifts_colour(test_image_paths, step, size_, vocab); 
%             save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end
    
    case 'spatial pyramids grayscale'
        % build vocab
        if exist(vocab_path_sp_grayscale, 'file')
            load(vocab_path_sp_grayscale)
        else
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocab_grayscale(train_image_paths, vocab_size, step, size_); %Also allow for different sift parameters
%             save(vocab_path_sp_grayscale, 'vocab')
        end 
        % sp feats
        if exist(img_feats_path_sp_grayscale, 'file')
            load(img_feats_path_sp_grayscale)
        else
            fprintf('No existing image features found. Computing one from images\n')
            train_image_feats = spatial_pyramid_newest(train_image_paths, sp_level, step, size_, vocab); %Allow for different sift parameters
            test_image_feats  = spatial_pyramid_newest(test_image_paths, sp_level, step, size_, vocab); 
%             save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end

    case 'spatial pyramids colour'
        % build vocab
        if exist(vocab_path_sp_colour, 'file')
            load(vocab_path_sp_colour)
        else
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocab_colour(train_image_paths, vocab_size, step, size_); %Also allow for different sift parameters
            save(vocab_path_sp_colour, 'vocab')
        end 
        % sp feats
        if exist(img_feats_path_sp_colour, 'file')
            load(img_feats_path_sp_colour)
        else
            fprintf('No existing image features found. Computing one from images\n')
            train_image_feats = get_spatial_pyramid_sifts_colour(train_image_paths, sp_level, step, size_, vocab); %Allow for different sift parameters
            test_image_feats  = get_spatial_pyramid_sifts_colour(test_image_paths, sp_level, step, size_, vocab); 
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        end

end
%% Step 2: Classify each test image by training and using the appropriate classifier

fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER)
switch lower(CLASSIFIER)    
    case 'nearest neighbor'
        predicted_categories = knn_classifier(1, train_image_feats, train_labels, test_image_feats);
    case 'support vector machine'
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, LAMBDA);

end

%% Step 3: output to csv
accuracy = accuracy_return(test_labels, categories, predicted_categories);

