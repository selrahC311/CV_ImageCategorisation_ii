function [image_feats] = get_bag_of_sifts_colour(image_paths, step, size_, vocab_path)
load(vocab_path);

% Find out how many images we are processing
total_image = size(image_paths, 1);

% Get the size of the vocab i.e No of clusters
vocab_size = size(vocab, 1);

% Create a matrix to store the counts of each vocab in an image (histogram)
image_feats = zeros(total_image, vocab_size);

% Loop theough every image in the file
for image_count = 1:total_image
    % Read the image and turn it into grayscale
    image = imread(cell2mat(image_paths(image_count)));
    
    % Make the image of type single to work with the function vl_dsift()
    image = single(image);

    descriptors = [];
    for channel = 1:3
        colour_channel = image(:, :, channel);
        
        % Extract the SIFT Features and Descriptors
        [~, channel_descriptors] = vl_dsift(colour_channel, 'step', step, 'size', size_, 'Fast');
        
        % Concatenate the descriptors for the current channel
        descriptors = [descriptors, channel_descriptors];
    end

    % Find the nearrest vocab to each descriptor found
    [indices, ~] = knnsearch(vocab, single(descriptors'), "K", 1);
    image_feats(image_count, :) = histcounts(indices, vocab_size);

    % Normalize histogram
    image_feats(image_count, :) = image_feats(image_count, :) / sum(image_feats(image_count, :));
end
