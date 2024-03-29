function accuracy = accuracy_return(test_labels, categories, abbr_categories , predicted_categories)

num_categories = length(categories);

%% Create And Save Confusion Matrix

confusion_matrix = zeros(num_categories, num_categories);
for i=1:length(predicted_categories)
    row = find(strcmp(test_labels{i}, categories));
    column = find(strcmp(predicted_categories{i}, categories));
    confusion_matrix(row, column) = confusion_matrix(row, column) + 1;
end

num_test_per_cat = length(test_labels) / num_categories;
confusion_matrix = confusion_matrix ./ num_test_per_cat;   
accuracy = mean(diag(confusion_matrix));
fprintf('Accuracy (mean of diagonal of confusion matrix) is %.3f\n', accuracy)

% fig_handle = figure; 
% imagesc(confusion_matrix, [0 1]); 
% set(fig_handle, 'Color', [.988, .988, .988])
% axis_handle = get(fig_handle, 'CurrentAxes');
% set(axis_handle, 'XTick', 1:15)
% set(axis_handle, 'XTickLabel', abbr_categories)
% set(axis_handle, 'YTick', 1:15)
% set(axis_handle, 'YTickLabel', categories)
