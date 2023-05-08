function [image_feats] = get_spatial_pyramid_sifts_grayscale(image_paths)
% TODO spatial pyramids method

% pyramid level - L
% Number of bins per level - K

% Find out how many images we are processing
total_image = size(image_paths, 1);

% Create a matrix to store the counts of each vocab in an image (histogram)
image_feats = zeros(total_image, vocab_size);

% Loop theough every image in the file
for image_count_i = 1:total_image
    for image_count_j = image_count_i:total_image
    
    % Read the image and turn it into grayscale
    image = rgb2gray(imread(cell2mat(image_paths(image_count))));
    
    % Make the image of type single to work with the function vl_dsift()
    image = single(image);
    
    % Extract the SIFT Features and Descriptors
    [~, descriptors] = vl_dsift(image, 'step', 3, 'size', 8, 'Fast');
    
    % Convert the descriptors to single data type
    descriptors = single(descriptors);
    
    
    end
end



