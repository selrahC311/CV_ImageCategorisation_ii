function predicted_categories = nearest_neighbor_classify(k,train_image_feat, train_label, test_image_feat)
%KNN_CLASSIFIER
distances = pdist2(test_image_feat, train_image_feat, "cityblock");

[~, indices] = sort(distances, 2);
k_nearest_indicies = indices(:,1:k);

% Make the size of the prediction category result
predicted_categories = cell(size(train_image_feat, 1),1);
for i = 1:size(train_image_feat, 1)
    nearest_label = train_label(k_nearest_indicies(i, :));

    % Use histcounts to count the number of occurrences of each string in the array
    [counts, ~] = histcounts(str2double(nearest_label));

    % Find the strings that occur more than once
    if any(counts >= 2)
        predicted_categories{i} = char(mode(str2double(nearest_label)));
    else
        predicted_categories{i} = char(nearest_label(1));
    end
end

