%% uncomment grayscale or colour before run

% FEATURE = 'bag of sift grayscale';
% FEATURE = 'bag of sift colour';
% FEATURE = 'spatial pyramids grayscale';
FEATURE = 'spatial pyramids colour';

total_step = 10;
total_size_ = 16; 
sp_level = 2;

vocab_size = 500;

vocab_path_bow_grayscale = "vocab_grayscale/vocab_" + vocab_size + ".mat";
vocab_path_bow_colour = "vocab_colour/vocab_" + vocab_size + ".mat";
vocab_path_sp_grayscale = "vocab_grayscale/vocab_" + vocab_size + ".mat";
vocab_path_sp_colour = "vocab_colour/vocab_" + vocab_size + ".mat";

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

%% start loop, uncomment grayscale or colour before run

for step = 2:2:total_step
    for size_ = 2:2:total_size_
   
        fprintf("step: " + step + ", size: " + size_ + '\n');
      
        switch FEATURE
            case 'bag of sift grayscale'
                save_path = "image_feats/bow_grayscale/" + "step" + step + "/size" + size_;
                train_image_feats = get_bag_of_sifts_grayscale(train_image_paths, step, size_, vocab_path_bow_grayscale); %Allow for different sift parameters
                test_image_feats  = get_bag_of_sifts_grayscale(test_image_paths, step, size_, vocab_path_bow_grayscale); 
               
            case 'bag of sift colour'
                save_path = "image_feats/bow_colour/" + "step" + step + "/size" + size_;
                train_image_feats = get_bag_of_sifts_colour(train_image_paths, step, size_, vocab_path_bow_colour); %Allow for different sift parameters
                test_image_feats  = get_bag_of_sifts_colour(test_image_paths, step, size_, vocab_path_bow_colour); 
               
            case 'spatial pyramids colour'
                save_path = "image_feats/sp_colour/" + "step" + step + "/size" + size_;
                train_image_feats = get_spatial_pyramid_sifts_colour(train_image_paths, sp_level, step, size_, vocab_path_sp_colour); %Allow for different sift parameters
                test_image_feats  = get_spatial_pyramid_sifts_colour(test_image_paths, sp_level, step, size_, vocab_path_sp_colour); 
               
            case 'spatial pyramids grayscale'
                save_path = "image_feats/sp_grayscale/" + "step" + step + "/size" + size_;
                train_image_feats = spatial_pyramid_newest(train_image_paths, sp_level, step, size_, vocab_path_sp_grayscale); %Allow for different sift parameters
                test_image_feats  = spatial_pyramid_newest(test_image_paths, sp_level, step, size_, vocab_path_sp_grayscale); 
        end
        
        mkdir(save_path);
        filename = "img_feat_vocab_" + vocab_size + ".mat";
        save(save_path + "/" + filename, 'train_image_feats', 'test_image_feats')

    end
end

