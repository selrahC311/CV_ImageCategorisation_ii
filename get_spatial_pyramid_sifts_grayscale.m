function [image_feats] = get_spatial_pyramid_sifts_grayscale(image_paths, sp_levels, step, size_, vocab_path)
tic
load(vocab_path)

vocab_size = size(vocab, 1);
total_image = size(image_paths, 1);
image_feats = zeros(total_image, (vocab_size * (4^sp_levels - 1)) / 3);

for image_count = 1:total_image
    % grayscale img
    image = rgb2gray(imread(cell2mat(image_paths(image_count))));
    [height, width] = size(image);

    % each sp levels
    level_feat_vec = [];
    for level = 1:sp_levels
        % indices for each cell region
        row = floor(linspace(1, height, 1 + 2^(level-1)));   
        col = floor(linspace(1, width, 1 + 2^(level-1)));

        for i = 2:length(row)
            for j = 2:length(col)
                % each region 
                region = image(row(i-1):row(i), col(j-1):col(j));
                region = single(region);

                % sift features and descriptors
                [~, descriptors] = vl_dsift(region, 'step', step, 'size', size_, 'Fast');
                descriptors = single(descriptors);

                % nearest vocab to descriptors
                [indices, ~] = knnsearch(single(vocab), descriptors', "K", 1);
                level_feat_vec = [level_feat_vec, histcounts(indices, vocab_size)];
            end
        end
    end
    % normalize histogram
    norm_value = norm(level_feat_vec, 2);
    level_feat_vec = level_feat_vec / norm_value;
        
    image_feats(image_count, :) = level_feat_vec;
end
toc