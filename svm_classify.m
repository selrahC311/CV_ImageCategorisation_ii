% Based on James Hays, Brown University 

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters. 

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in coursework_starter,
%because unique() sorts them. This shouldn't really matter, though.
% sort/get the diff categories
categories = unique(train_labels); 

% Find the number of categories we are working with
num_categories = length(categories);

% Find the number of images we are working with
num_train = size(train_image_feats, 1);

% Find out how many images we are tryna predict
num_test = size(test_image_feats, 1);

% value for lamda
lambda = 0.000001;

% Scores
scores = zeros(num_categories, num_train);

% for each categorty
for cat_num = 1:num_categories
    % changes the value of the train label to be 1's if them match or 0's
    matched_indices = double(strcmp(categories(cat_num), train_labels));

    % for each lable in the train label
    for i = 1:num_train
        % check to see if it's a match (1 or 0)
        if matched_indices(i) == 0
            % change the 0 to a -1
            matched_indices(i) = -1;
        end
    end

    % Train the svm for the category
    [w, b] = vl_svmtrain(train_image_feats', matched_indices, lambda);

    % get the weights 
    scores(cat_num,:) =  w'*test_image_feats' + b;
end
% get maximum scores
[~, max_indices] = max(scores);

% Flip the dimetion of the matrix
max_indices = max_indices';

%
predicted_categories = categories(max_indices);
