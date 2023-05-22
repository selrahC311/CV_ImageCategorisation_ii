function image_feats = get_bag_of_sifts_grayscale(image_paths, step, size_, vocab)
tic
% num of clusters
vocab_size = size(vocab, 1);
total_image = size(image_paths, 1);
image_feats = zeros(total_image, vocab_size);

for image_count = 1:total_image
    % grayscale img
    image = rgb2gray(imread(cell2mat(image_paths(image_count))));
    image = single(image);
    
    % sift features and descriptors
    [~, descriptors] = vl_dsift(image, 'step', step, 'size', size_, 'Fast');
    descriptors = single(descriptors);
    
    % find nearrest vocab to descriptor
    [indices, ~] = knnsearch(vocab, descriptors', "K", 1);
    image_feats(image_count, :) = histcounts(indices, vocab_size);

    % normalize histogram
    image_feats(image_count, :) = image_feats(image_count, :) / sum(image_feats(image_count, :));
end
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
