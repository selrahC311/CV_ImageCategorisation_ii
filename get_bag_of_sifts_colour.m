function [image_feats] = get_bag_of_sifts_colour(image_paths, step, size_, vocab_path)
tic
load(vocab_path);

% num of clusters
vocab_size = size(vocab, 1);

total_image = size(image_paths, 1);
image_feats = zeros(total_image, vocab_size);

for image_count = 1:total_image
    % rgb image
    image = imread(cell2mat(image_paths(image_count)));
    image = single(image);

    % each colour channel
    descriptors = [];
    for channel = 1:3
        colour_channel = image(:, :, channel);
        
        % sift features and descriptors
        [~, channel_descriptors] = vl_dsift(colour_channel, 'step', step, 'size', size_, 'Fast');
        
        % Concatenate descriptors
        descriptors = [descriptors, channel_descriptors];
    end

    % find nearrest vocab to descriptor
    [indices, ~] = knnsearch(vocab, single(descriptors'), "K", 1);
    image_feats(image_count, :) = histcounts(indices, vocab_size);

    % normalize histogram
    image_feats(image_count, :) = image_feats(image_count, :) / sum(image_feats(image_count, :));
end
toc