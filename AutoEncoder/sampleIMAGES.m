function patches = sampleIMAGES()
% sampleIMAGES
% return 10000 images for training
    addpath mnistHelper/
    
    IMAGES = loadMNISTImages('train-images.idx3-ubyte');
    
    patchsize = 28;
    numpatches = 10000;

    patches = zeros(patchsize * patchsize, numpatches);

    for i = 1:10000
        patches(:, i) = IMAGES(:, i);
    end
    
    patches = reshape(patches, patchsize, patchsize, size(patches, 2));
    patches = permute(patches, [2 1 3]);
    patches = reshape(patches, size(patches, 1) * size(patches, 2), size(patches, 3));
    
    clear IMAGES;
%     patches = normalizeData(patches);
end

function patches = normalizeData(input_patches)
% normalize data
    input_patches = bsxfun(@minus, input_patches, mean(input_patches));
    pstd = 3 * std(input_patches(:));
    input_patches = max(min(input_patches, pstd), -pstd) / pstd;
    patches = (input_patches + 1) * 0.4 + 0.1;
end

