function image3 = ImageInterpolation(image1, xt, yt)

image1 = double(image1);

xx = yt; yy = xt;

xxt = floor(xx);
yyt = floor(yy);
fx  = xx - xxt;
fy  = yy - yyt;

[mt, nt] = size(xxt);

[m1, n1] = size(image1);

% tic

idxi = 1:mt*nt;
xi = xxt(idxi);
yi = yyt(idxi);
        
idxi1 = (yi-1)*m1 + xi;
idxi2 = (yi-1)*m1 + xi+1;
idxi3 = (yi+1-1)*m1 + xi;
idxi4 = (yi+1-1)*m1 + xi+1;
        
imt1 = image1(idxi1); 
imt2 = image1(idxi2); 
imt3 = image1(idxi3); 
imt4 = image1(idxi4);

for jj = 1:nt
    im1(:, jj) = imt1((jj-1)*mt+1:jj*mt);
    im2(:, jj) = imt2((jj-1)*mt+1:jj*mt);
    im3(:, jj) = imt3((jj-1)*mt+1:jj*mt);
    im4(:, jj) = imt4((jj-1)*mt+1:jj*mt);
end
% t_calc = toc

image3 = im1.*(1 - fx).*(1 - fy) + im3.*(1 - fx).*fy + ...
    im2.*fx.*(1 - fy) + im4.*fx.*fy;

%% image3 = uint8(image3);