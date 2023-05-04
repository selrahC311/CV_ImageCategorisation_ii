data_path = '../data/';

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

% YOU CODE build_vocabulary.m
if ~exist('vocab.mat', 'file')
    fprintf('No existing dictionary found. Computing one from training images\n')
    vocab_size = 51; % you need to test the influence of this parameter
    vocab = build_vocabulary(train_image_paths, vocab_size); %Also allow for different sift parameters
    save('vocab.mat', 'vocab')
end

% YOU CODE get_bags_of_sifts.m
if ~exist('image_feats.mat', 'file')
    train_image_feats = get_bags_of_sifts(train_image_paths); %Allow for different sift parameters
    test_image_feats  = get_bags_of_sifts(test_image_paths);
    save('image_feats.mat', 'train_image_feats', 'test_image_feats')
end
