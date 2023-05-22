clear;
% FEATURE = 'bag of sift grayscale';
FEATURE = 'bag of sift colour';
% FEATURE = 'spatial pyramids grayscale';
% FEATURE = 'spatial pyramids colour';

% CLASSIFIER = 'nearest neighbor';
CLASSIFIER = 'support vector machine';

fclose('all');
fid = fopen('csv/results_bow_colour_svm.csv', 'w+t');

% Open file for writing output message
[success, message, messageid] = mkdir('csv');
fprintf(fid, 'step,size,vocab,accuracy \n');
for step = 2:2:10
    for size_ = 2:2:16
        for vocab_size = 100:200:1000
            fprintf('step: %d, size: %d, vocab size: %d\n', step, size_, vocab_size)
            run("test_starter.m");
            fprintf(fid, '%d,%d,%d,%d \n', step, size_, vocab_size, accuracy);
        end
    end
end

fclose(fid);
