close all
clear all
warning off

H = csvread('./MergeHeight.csv');

[m1, n1] = size(H);
[x1, y1] = meshgrid(1:n1, 1:m1);

figure(1)
mesh(x1, -y1, H)
grid on
hold on
colorbar;