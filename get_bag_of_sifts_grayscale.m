function [image_feats] = get_bag_of_sifts_grayscale(image_paths, step, size)
load('vocab.mat')

% Find out how many images we are processing
total_image = size(image_paths, 1);

% Get the size of the vocab i.e No of clusters
vocab_size = size(vocab, 1);

% Create a matrix to store the counts of each vocab in an image (histogram)
image_feats = zeros(total_image, vocab_size);

% Loop theough every image in the file
for image_count = 1:total_image
    % Read the image and turn it into grayscale
    image = rgb2gray(imread(cell2mat(image_paths(image_count))));
    
    % Make the image of type single to work with the function vl_dsift()
    image = single(image);
    
    % Extract the SIFT Features and Descriptors
    [~, descriptors] = vl_dsift(image, 'step', step, 'Size', size, 'Fast');
    
    % Convert the descriptors to single data type
    descriptors = single(descriptors);
    
    % Find the nearrest vocab to each descriptor found
    [indices, ~] = knnsearch(vocab, descriptors', "K", 1);
        
    image_feats(image_count, :) = histcounts(indices, vocab_size);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.
%}
