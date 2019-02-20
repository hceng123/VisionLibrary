close all
clear all
warning off

File_path = '/Users/chenlining/Desktop/Wistron/EAOI/PatternImage/Test_20180602_3D/FOV20190104';
File_path = '/Users/chenlining/Desktop/Wistron/EAOI/PatternImage/Test_20180602_3D/Chessboard20190113';
File_path = '/Users/chenlining/Desktop/Wistron/EAOI/PatternImage/Test_20180602_3D/FOV20190112';
File_path = '/Users/chenlining/Desktop/Wistron/EAOI/PatternImage/Test_20180602_3D/Chessboard20190126/FOV03';
File_path = '/Users/chenlining/Desktop/Wistron/EAOI/PatternImage/Test_20180602_3D/Chessboard20190214';
% File_path = '/Users/chenlining/Desktop/Wistron/EAOI/PatternImage/Test_20180602_3D/PCB20190122/FOV1950';
% File_path = '/Users/chenlining/Desktop/Wistron/EAOI/PatternImage/Test_20180602_3D/Chessboard20190129';
% File_path = '/Users/chenlining/Desktop/Wistron/EAOI/PatternImage/Test_20180602_3D/PCB20190129';
File_path = './TestCombineImage_2';
D         = struct2cell(dir(File_path));

Save_path = './Distortion';

%%%%%%%%%%%
theta = -5:5; %% (-2.5degC to 2.5degC)

Save_file = 'Distortion.mat';
% Save_file = 'Distortion34.mat';
load([Save_path, '/', Save_file]);

%%%%%%%%%% pixel resolution
px = 0.01588307;
py = 0.01588427;

%%%%%%%%%%% Read Bezier Surface
File_path02 = './BezSurface';
Savefile02  = 'BezSurface.mat';

for i = 1 : 11
    xdistortion0 = xdistortion(:, :, i)
    ydistortion0 = ydistortion(:, :, i)
    file = fopen(['./Distortion/Frame_', int2str(i), '.txt'], 'w');
    for j = 1 : length(xdistortion0(:))
        fprintf(file, "%f, %f\n", xdistortion0(j), ydistortion0(j));
    end
    
    fclose(file);
end

%%%%%%%%%%%%%%% Reading table mapping data
load([File_path02, '/', Savefile02]);

%%%%%%%%%%%%%%% Read Scanning csv
D2              = struct2cell(dir([File_path, '/*.csv']));
Fresh_Filename  = D2(1, :);

fileID  = fopen([File_path, '/', char(Fresh_Filename(1))]);
format1 = repmat({'%f '}, 18, 1); format1 = strjoin(format1);
Data    = textscan(fileID, format1, 'delimiter', ',');
fclose(fileID);

x1 = Data{:, 1};
y1 = Data{:, 2};

%%%%%%%%%%%%%%% Calculate dtheta
PPz3 = reshape(PPz2, nn, mm);
PPz3 = (mm-1)*diff(PPz3, 1, 2);
PPz3 = reshape(PPz3, nn*(mm-1), 1);
dydu = CalculateSurface02(x1, y1, Para, PPz3, mm-1, nn);
dydx = dydu/(Para(2) - Para(1));

%%%%%%%%%%%%%%%% auto-Calculate mm/nn
idx = find(abs(diff(y1)) > 5);
idx = idx(1);

nn = idx;
mm = length(x1)/nn;

ktheta = 1;
dtheta = reshape(dydx, nn, mm)';
dtheta = round(dtheta/(0.5/2048)*ktheta);

ctf = 6;
%%% Calculate Nx/Ny
%%%%%%%%%%%
m1 = 2048;
n1 = 2040;

nt = 6; %% remove 4 start/end pixel per col/row

m = m1 - 2*nt;
n = n1 - 2*nt;

xdistortion0 = xdistortion;
ydistortion0 = ydistortion;
%%%%%%%%%%%% Overlapping point
Nx = round(n1-abs(x1(2) - x1(1))/px) - 2*nt;
Ny = round(m1-abs(y1(nn+1) - y1(nn))/py) - 2*nt;

if Nx < 0; Nx = 0; end
if Ny < 0; Ny = 0; end

%%%%%%%%%%%% Phase Selection
ss = 2;
%%%%%%%%%%%% Start Image 48/0
ns = 0;
%%%%%%%%%%%% pixel shift, set 0, internal testing
dpixel = 0;

% image2 = ImageInterpolation(image1, ut1, vt1);

for jj = 1:nn*mm    
    File_path01 = [File_path, '/', char(D{1, jj+3})];
    D1          = struct2cell(dir([File_path01, '/*.bmp']));
    Fresh_Filename  = D1(1, :);
    
    len = length(Fresh_Filename);
    
    %%%%%%%%%%%%%%%%%%%%
    if ss == 3
        image(:, :, 1) = imread([File_path01, '/', char(Fresh_Filename(ss+ns))]);
        image(:, :, 2) = imread([File_path01, '/', char(Fresh_Filename(ss+1+ns))]);
        image(:, :, 3) = imread([File_path01, '/', char(Fresh_Filename(ss+2+ns))]);
    elseif ss == 4
        image(:, :, 1) = imread([File_path01, '/', char(Fresh_Filename(ss+2+ns))]);
        image(:, :, 2) = image(:, :, 1);
        image(:, :, 3) = image(:, :, 1);
    else
        image(:, :, 1) = imread([File_path01, '/', char(Fresh_Filename(ss+ns))]);
        image(:, :, 2) = image(:, :, 1);
        image(:, :, 3) = image(:, :, 1);
    end
    
    %%%%%%%%% fix distortion here
    %         imaget          = zeros(m1-2*nt, n1-2*nt, 3, 'uint8');
    %         imaget(:, :, 1) = image(idxdis);
    %         imaget(:, :, 2) = image(idxdis + m1*n1);
    %         imaget(:, :, 3) = image(idxdis + m1*n1*2);
    
    idx1 = ceil(jj/nn);
    idx2 = jj - (idx1 - 1)*nn;
    idx0 = (idx2-1)*mm + idx1;
    
    idx  = -dtheta(idx0) + ctf;
    
    if idx < 1; idx = 1; end
    if idx > length(theta); idx = length(theta); end
    %idx  = 0 + 6;
    
    %         xdistortion = xdistortion0;
    %         ydistortion = ydistortion0;
    
    xdistortion = xdistortion0(:, :, idx);
    ydistortion = ydistortion0(:, :, idx);
    
    %         xdistortion = round(xdistortion);
    %         ydistortion = round(ydistortion);
    
    xdistortion = xdistortion(1+nt:m1-nt, 1+nt:n1-nt);
    ydistortion = ydistortion(1+nt:m1-nt, 1+nt:n1-nt);
    
    imaget          = zeros(m1-2*nt, n1-2*nt, 3, 'uint8');
    imaget(:, :, 1) = ImageInterpolation(image(:, :, 1), xdistortion, ydistortion);
    
    if ss == 3
        imaget(:, :, 2) = ImageInterpolation(image(:, :, 2), xdistortion, ydistortion);
        imaget(:, :, 3) = ImageInterpolation(image(:, :, 3), xdistortion, ydistortion);
    else
        imaget(:, :, 2) = imaget(:, :, 1);
        imaget(:, :, 3) = imaget(:, :, 1);
    end
    %       imaget = image(1+nt:m1-nt, 1+nt:n1-nt, :);
    
    eval(['image', num2str(jj), ' = imaget;']);
end

for jj = 1:mm

    kk = nn;

    R = zeros(m, kk*n - (kk-1)*Nx, 'uint8');
    G = zeros(m, kk*n - (kk-1)*Nx, 'uint8');
    B = zeros(m, kk*n - (kk-1)*Nx, 'uint8');

    for ii = 1:kk%kk:-1:1

        eval(['imagetemp = image', num2str((jj-1)*nn + ii), ';']);
        
        %pp = kk - ii + 1;
        pp = ii;
        
        if ii ~= kk

            R(:, (((pp-1)*(n - Nx)) + 1):(pp*(n - Nx))) = imagetemp(:, 1:n-Nx, 1);
            G(:, (((pp-1)*(n - Nx)) + 1):(pp*(n - Nx))) = imagetemp(:, 1:n-Nx, 2);
            B(:, (((pp-1)*(n - Nx)) + 1):(pp*(n - Nx))) = imagetemp(:, 1:n-Nx, 3);
            
        else
            R(:, (((pp-1)*(n - Nx)) + 1):(pp*n - (pp-1)*Nx)) = imagetemp(:, :, 1);
            G(:, (((pp-1)*(n - Nx)) + 1):(pp*n - (pp-1)*Nx)) = imagetemp(:, :, 2);
            B(:, (((pp-1)*(n - Nx)) + 1):(pp*n - (pp-1)*Nx)) = imagetemp(:, :, 3);    
            
        end

    end
    
    imaget = zeros(m, kk*n - (kk-1)*Nx, 3, 'uint8');

    imaget(:, :, 1) = R;
    imaget(:, :, 2) = G;
    imaget(:, :, 3) = B;
    
    eval(['imagel', num2str(jj), ' = imaget;']);
end   

R = zeros(jj*m - (jj-1)*Ny, kk*n - (kk-1)*Nx, 'uint8');
G = zeros(jj*m - (jj-1)*Ny, kk*n - (kk-1)*Nx, 'uint8');
B = zeros(jj*m - (jj-1)*Ny, kk*n - (kk-1)*Nx, 'uint8');

for ii = 1:jj
    eval(['imagetemp = imagel', num2str(ii), ';'])
    
    dn   = dpixel*(ii-1);
    lenn = length(imagetemp(1, :, :));
    
    if dn >= 0
        imagetemp(:, 1+dn:lenn, :) = imagetemp(:, 1:lenn-dn, :);
        imagetemp(:, 1:dn, :)      = 0;
    else
        imagetemp(:, 1:lenn-dn, :)    = imagetemp(:, 1+dn:lenn, :);
        imagetemp(:, lenn-dn+1:lenn, :) = 0;
    end
        
    
    if ii ~= jj
    
        R((((ii-1)*(m - Ny)) + 1):(ii*(m - Ny)), :) = imagetemp(1:m-Ny, :, 1);
        G((((ii-1)*(m - Ny)) + 1):(ii*(m - Ny)), :) = imagetemp(1:m-Ny, :, 2);
        B((((ii-1)*(m - Ny)) + 1):(ii*(m - Ny)), :) = imagetemp(1:m-Ny, :, 3);
        
    else
        
        R((((ii-1)*(m - Ny)) + 1):(ii*m - (ii-1)*Ny), :) = imagetemp(:, :, 1);
        G((((ii-1)*(m - Ny)) + 1):(ii*m - (ii-1)*Ny), :) = imagetemp(:, :, 2);
        B((((ii-1)*(m - Ny)) + 1):(ii*m - (ii-1)*Ny), :) = imagetemp(:, :, 3);
        
    end
        
end
   
imagebig(:, :, 1) = R;
imagebig(:, :, 2) = G;
imagebig(:, :, 3) = B;

figure(1)
imshow(imagebig)
grid on
hold on

%%%%%%%%%%%%%%%
for ii = 1:nn-1
    tt = (n-Nx)*ii;
    plot([tt, tt], [1, mm*m-(mm-1)*Ny], 'c')
end
for ii = 1:mm-1
    tt = (m-Ny)*ii;
    plot([1, nn*n-(nn-1)*Nx], [tt, tt], 'c')
end

%%%%%%%%%%%%% Save Imaage
% imwrite(imagebig,[File_path, '/WholeBoard.jpg'],'jpg');
% imwrite(imagebig(:, :, 1),[File_path, '/WholeBoard.bmp'],'bmp');