function vocab = build_vocab_colour( image_paths, vocab_size, step, size_)
% Find out how many images we are processing
total_image = size(image_paths, 1);

% Number of visual Words
k = vocab_size;

% initialize empty matrix to store SIFT features
sift_features = [];

for image_count = 1:total_image
    image = imread(cell2mat(image_paths(image_count)));

    descriptors = [];
    for channel = 1:3
        colour_channel = single(image(:, :, channel));
        
        % Extract the SIFT Features (F) and Descriptors (D)
        [~, colour_descriptors] = vl_dsift(colour_channel, 'step', step, 'size', size_, 'Fast');
        
        % Concatenate descriptors
        descriptors = [descriptors, single(colour_descriptors)];
    end

    % concatenate descriptors to sift_features matrix
    sift_features = [sift_features, single(descriptors)];
end

% Cluster the the descriptors to k clusters
fprintf('clustering the centers \n')
[centers, ~] = vl_kmeans(single(sift_features), k);
fprintf('done clustering the centres \n')

vocab = centers';
