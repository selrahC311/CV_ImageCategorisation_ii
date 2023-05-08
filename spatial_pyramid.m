function [image_feats, indices] = spatial_pyramid(imagepath, num_regions)

load('vocab.mat')

% Find out how many images we are processing
total_image = size(imagepath, 1);

% Get the size of the vocab i.e No of clusters
vocab_size = size(vocab, 1);

image_feats = zeros(total_image, vocab_size*num_regions);


% Loop for every image we are processing
for image_count = 1:total_image
    % Read the image and turn it into grayscale
    image_grayscale = rgb2gray(imread(cell2mat(imagepath(image_count))));


    % Get the width and height
    [height, width, ~] = size(image_grayscale);

    % check if its a perfect square
    sqare_root = sqrt(num_regions);
    if floor(sqare_root) == sqare_root
        row = sqare_root;
        col = sqare_root;
    else
        % Divide the image into 2 rows and find the number of columns
        row = 2;
        col = ceil(num_regions/row);
    end

    % Get the regions width and height
    region_height = round(height/row);
    region_width = round(width/col);

    % Initialise the region feature vector
    region_feats = zeros(1, vocab_size*num_regions);
    
    counter = 1;
    % Number of rows
    for i = 1:row
        % Number of columns
        for j = 1:col
            % Get the region of the image
            region = image_grayscale(floor((i-1)*(region_height/row))+1:floor(i*(region_height/row)), floor((j-1)*(region_width/col))+1:floor(j*(region_width/col)), :);
            % Make the image of type single to work with the function vl_dsift()
            image = single(region);
            % Extract the SIFT Features and Descriptors
            [~, descriptors] = vl_dsift(image, 'Fast');
            % Convert the descriptors to single data type
            descriptors = single(descriptors);
            % Find the nearrest vocab to each descriptor found
            [indices, ~] = knnsearch(vocab, descriptors', "K", 1);
            
            % find how many times the nth visual cluster apeared in an image
            region_feats(1, (counter-1)*(vocab_size)+1:(counter)*(vocab_size)) = histcounts(indices, vocab_size);

            counter = counter + 1;
        end
    end
%     % Normalize the histogram
%     region_feats(1, :) = region_feats(1, :) / sum(region_feats(1, :));

    % Add the region feature vector to the image feature matrix
    image_feats(image_count, :) = region_feats;
    fprintf('Progress: %d%%\n', round((image_count/total_image)*100));
end
end