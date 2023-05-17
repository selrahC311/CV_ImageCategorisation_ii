function vocab = build_vocab_colour( image_paths, vocab_size, step, size_)
% Find out how many images we are processing
total_image = size(image_paths, 1);

% Number of visual Words
k = vocab_size;

% initialize empty matrix to store SIFT features
sift_features = [];

for image_count = 1:total_image
    image_grayscale = imread(cell2mat(image_paths(image_count)));
    image = single(image_grayscale);

    descriptors = [];
    for channel = 1:3
        colour_channel = image(:, :, channel);
        
        % Extract the SIFT Features (F) and Descriptors (D)
        [~, colour_descriptors] = vl_dsift(colour_channel, 'step', step, 'size', size_, 'Fast');
        colour_descriptors = colour_descriptors';
        
        % Concatenate descriptors
        descriptors = [descriptors, colour_descriptors];
    end

    % concatenate descriptors to sift_features matrix
    sift_features = [sift_features, descriptors'];
end

fprintf('clustering the centers \n')
% Cluster the the descriptors to k clusters
[centers, ~] = vl_kmeans(single(sift_features), k);
fprintf('done clustering the centres \n')

vocab = centers';
