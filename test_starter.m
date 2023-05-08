data_path = '../data/';


% FEATURE = "spatial pyramids";
FEATURE = "bag of sift";

% Classifiers
% CLASSIFIER = "knn";
CLASSIFIER = "svm";


categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
    'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
    'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};

%This list of shortened category names is used later for visualization.
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};

%number of training examples per category to use. Max is 100.
num_train_per_cat = 100;

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and test image.
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);


switch lower(FEATURE)
    case "bag of sift"
        % YOU CODE build_vocabulary.m
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab_size = 1000; % you need to test the influence of this parameter
            vocab = build_vocabulary(train_image_paths, vocab_size); %Also allow for different sift parameters
            fprintf('Saving the vocab file')
            save('vocab.mat', 'vocab')
        end

        % YOU CODE get_bags_of_sifts.m
        if ~exist('training_bag.mat', 'file')
            fprintf('Computing training features\n');
            train_image_feats = get_bags_of_sifts(train_image_paths);
            save('training_bag.mat', 'train_image_feats');
        else
            fprintf('Loading training features\n');
            load('training_bag.mat');
        end
        
<<<<<<< HEAD
        if ~exist('test_bag.mat', 'file')
            fprintf('Computing test features\n');
            test_image_feats  = get_bags_of_sifts(test_image_paths);
            save('test_bag.mat', 'test_image_feats')
        else
            fprintf('Loading test features\n');
            load('test_bag.mat');
        end
    case "spatial pyramids"
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab_size = 50; % you need to test the influence of this parameter
            vocab = build_vocabulary(train_image_paths, vocab_size); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end

        % YOU CODE spatial_pyramid.m
        [train_image_feats, res] = spatial_pyramid(train_image_paths, 4); %Allow for different sift parameters
        [test_image_feats, res2]  = spatial_pyramid(test_image_paths, 4);
%         save('image_feats.mat', 'train_image_feats', 'test_image_feats')
=======
        %% Step 3: Build a confusion matrix and score the recognition system
        % You do not need to code anything in this section. 
        
        % This function will recreate results_webpage/index.html and various image
        % thumbnails each time it is called. View the webpage to help interpret
        % your classifier performance. Where is it making mistakes? Are the
        % confusions reasonable?
        create_results_webpage_modified( train_image_paths, ...
                                test_image_paths, ...
                                train_labels, ...
                                test_labels, ...
                                categories, ...
                                abbr_categories, ...
                                predicted_categories)
    end
>>>>>>> main
end



switch lower(CLASSIFIER)
    case "knn"
        predicted_categories = knn_classifier(1,train_image_feats, train_labels, test_image_feats);
    case "svm"
        predicted_categories = svm(train_image_feats, train_labels, test_image_feats);
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
