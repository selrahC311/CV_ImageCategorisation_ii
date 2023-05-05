function [image_feats] = get_bag_of_sifts_colour_test(image_paths, colour_space, step, size)

load('vocab.mat')

% Find out how many images we are processing
total_image = length(image_paths);

% Get the size of the vocab i.e No of clusters
vocab_size = length(vocab);

% Create a matrix to store the counts of each vocab in an image (histogram)
image_feats = zeros(total_image, vocab_size);

% Loop theough every image in the file
for image_count = 1:total_image
% Read the image and turn it into grayscale
    image = imread(cell2mat(image_paths(image_count)));
    
    % Make the image of type single to work with the function vl_dsift()
    image = single(image);

    switch colour_space
        case 'rgb'

        case 'hsv'
            image = rbg2hue(image);
        case 'ycbcr'
            image = rgb2ycbcr(image);
        case 'lab'
            image = applycform(image, makecform('srgb2lab'));
            image = im2uint8(image);
        otherwise
            disp("Error, unsupported colour space. ")
    end

    indices_image = [];
    for channel = 1:3
        colour_channel = image(:, :, channel);
        
        % Extract the SIFT Features and Descriptors
        [~, descriptors] = vl_dsift(colour_channel, 'step', step, 'size', size, 10, 'Fast');
        
        % Convert the descriptors to single data type
        descriptors = single(descriptors);
        
        % Compute the Euclidean distances between each SIFT descriptor and
        % all vocabulary words using vl_alldist2
        dists = vl_alldist2(vocab', descriptors');
        
        % Find the nearest vocab to each descriptor found
        [~, indices] = min(dists);
        
        % Concat each channel to single feature
        indices_image = cat(1, indices_image, indices);
    end
    
    image_feats(image_count, :) = histcounts(indices_image, vocab_size);
end
