function predicted_categories = random_forest_classify(train_image_feats, train_labels, test_image_feats, num_trees)
    % Category list
    categories = unique(train_labels);
    num_categories = length(categories);

    num_test = size(test_image_feats, 1);
    scores = zeros(num_categories, num_test);

    for cat_num = 1:num_categories
        % Binary labels for current category
        matched_indices = double(strcmp(categories(cat_num), train_labels));
        matched_indices(matched_indices == 0) = -1;

        % Train a random forest for the category
        tree_models = cell(num_trees, 1);
        for tree = 1:num_trees
            % Select random subset of training data with replacement
            subset_indices = randi(size(train_image_feats, 1), size(train_image_feats, 1), 1);
            subset_feats = train_image_feats(subset_indices, :);
            subset_labels = matched_indices(subset_indices);

            % Train a decision tree on the subset
            tree_models{tree} = fitctree(subset_feats, subset_labels);
        end

        % Predict scores for the category using the ensemble of trees
        for tree = 1:num_trees
            tree_scores = predict(tree_models{tree}, test_image_feats);
            scores(cat_num, :) = scores(cat_num, :) + tree_scores';
        end
    end

    % Predict the category with the maximum score for each test image
    [~, max_indices] = max(scores);
    predicted_categories = categories(max_indices');
end
