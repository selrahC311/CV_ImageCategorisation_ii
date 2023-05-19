function [image_feats] = spatial_pyramid(imagepath, num_levels)

load('spvocab.mat')

% Find out how many images we are processing
total_image = size(imagepath, 1);

% Get the size of the vocab i.e No of clusters
vocab_size = size(vocab, 1);

% initialise final matrix
image_feats = zeros(total_image, vocab_size*((4^num_levels)-1)/3);


% Loop for every image we are processing
for image_count = 1:total_image
    % Read the image and turn it into grayscale
    image_grayscale = rgb2gray(imread(cell2mat(imagepath(image_count))));


    % Get the width and height
    [height, width, ~] = size(image_grayscale);

    % Initialise the image feature vector
    level_feat_vec = [];

    for level = 1:num_levels
        % Get index for row and hight for each cell region
        row = floor(linspace(1, height, 1+2^(level-1)));    % Geometric series
        col = floor(linspace(1,width, 1+2^(level-1)));      % Geometric series

        

        for i= 2:length(row)
            for j= 2:length(col)
                % Get the region of the image
                region = image_grayscale(row(i-1):row(i), col(j-1):col(j));
                % Make the image of type single to work with the function vl_dsift()
                image = single(region);
                % Extract the SIFT Features and Descriptors
                [~, descriptors] = vl_dsift(image, 'Fast');
                % Convert the descriptors to single data type
                descriptors = single(descriptors);
                % Find the nearrest vocab to each descriptor found
                [indices, ~] = knnsearch(single(vocab), descriptors', "K", 1);
                % find how many times the nth visual cluster apeared in an image
                level_feat_vec =[level_feat_vec, histcounts(indices, vocab_size)];
            end
        end
    end
    % Normalize the histogram
    level_feat_vec = normalize(level_feat_vec);
    % Add the region feature vector to the image feature matrix
    image_feats(image_count, :) = level_feat_vec;
    fprintf('Progress: %d%%\n', round((image_count/total_image)*100));
end
end
