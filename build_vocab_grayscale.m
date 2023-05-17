function vocab = build_vocab_grayscale( image_paths, vocab_size, step, size_)
% Find out how many images we are processing
total_image = size(image_paths, 1);

% Number of visual Words
k = vocab_size;

% initialize empty matrix to store SIFT features
sift_features = [];

for image_count = 1:total_image
    % Read the image and turn it into grayscale
    image_grayscale = rgb2gray(imread(cell2mat(image_paths(image_count))));

    I = single(image_grayscale);
    % Extract the SIFT Features (F) and Descriptors (D)
    [~, descriptors] = vl_dsift(I, 'step', step, 'size', size_, 'Fast');

    % concatenate descriptors to sift_features matrix
    sift_features = [sift_features, descriptors];

    % Print the progress...so far...
%     fprintf('Progress: %d%%\n', round((image_count/total_image)*100));
end

fprintf('clustering the centers \n')
% Cluster the the descriptors to k clusters
[centers, ~] = vl_kmeans(single(sift_features), k);
fprintf('done cliustering the centres \n')
vocab = centers';
