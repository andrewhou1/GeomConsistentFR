img = imread('FFHQ_relighting_results/00290_rendered_image.png');
filter_img(:, :, 1) = medfilt2(img(:, :, 1));
filter_img(:, :, 2) = medfilt2(img(:, :, 2));
filter_img(:, :, 3) = medfilt2(img(:, :, 3));
face_mask = imread('FFHQ_skin_masks/00290.png')/255.0;
ones_filter = ones(7, 7);

convolved = imfilter(double(face_mask), ones_filter);

border = (convolved < 30 & convolved > 0);

[row, col] = find(border == 1);

for j = 1:length(row)
    img(row(j), col(j), :) = filter_img(row(j), col(j), :);
end

imwrite(img, 'FFHQ_relighting_results/00290_rendered_image.png');
