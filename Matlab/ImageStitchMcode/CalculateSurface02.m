function zp1 = CalculateSurface02(xt, yt, Para, PPz, mm, nn)

% mm = 5;
% nn = 5;

[m1, n1] = size(xt);
% [u, v]   = meshgrid((0:n1-1)/(n1-1), (0:m1-1)/(m1-1));

xmin = Para(1); xmax = Para(2);
ymin = Para(3); ymax = Para(4);

u = (xt(:) - xmin)/(xmax - xmin);
v = (yt(:) - ymin)/(ymax - ymin);

len = length(u);

%%%%%%%%%%%% Poly nomial coefficient
PolyParamm = ParaFromPolyNomial(mm);
PolyParann = ParaFromPolyNomial(nn);

%%%%%%%%%%%%%% u^ii*(1-u)^(n-ii-1), ii = 0 ~ n-1 
Pm1 = repmat(0:mm-1, len, 1);
Pm2 = repmat(mm-1:-1:0, len, 1);
Pm3 = repmat(u, 1, mm);
Pm4 = repmat(1-u, 1, mm);

%%%%%%%%%%%%%% v^ii*(1-v)^(v-ii-1), ii = 0 ~ n-1
Pn1 = repmat(0:nn-1, len, 1);
Pn2 = repmat(nn-1:-1:0, len, 1);
Pn3 = repmat(v, 1, nn);
Pn4 = repmat(1-v, 1, nn);

%%%%%%%%%%%%%%%%%%%%%%%%%%% 
P1 = (Pm3.^Pm1).*(Pm4.^Pm2).*repmat(PolyParamm, len, 1);
P2 = (Pn3.^Pn1).*(Pn4.^Pn2).*repmat(PolyParann, len, 1);

% P1 = [(1-u).^4, 4*(1-u).^3.*u,6*(1-u).^2.*u.^2,4*(1-u).*u.^3, u.^4];
% P2 = [(1-v).^4, 4*(1-v).^3.*v,6*(1-v).^2.*v.^2,4*(1-v).*v.^3, v.^4];

% P1 = [u14, 4*u13.*u, 6*u12.*u2, 4*(1-u).*u3, u4];
% P2 = [v14, 4*v13.*v, 6*v12.*v2, 4*(1-v).*v3, v4];

xx = zeros(len, mm*nn);

for ii = 1:mm
    for jj = 1:nn
        xx(:, (ii-1)*nn+jj) = P1(:, ii).*P2(:, jj);
    end
end

% P11 = P1.*repmat(v14, 1, 5);
% P12 = P1.*repmat(4*v13.*v, 1, 5);
% P13 = P1.*repmat(6*v12.*v2, 1, 5);
% P14 = P1.*repmat(4*(1-v).*v3, 1, 5);
% P15 = P1.*repmat(v4, 1, 5);
%    
% xx = [P11, P12, P13, P14, P15];

zp1 = xx*PPz;
zp1 = reshape(zp1, m1, n1);

%%%%%%%%%%%%%%%%%%
function PolyPara = ParaFromPolyNomial(n)

%% set input limit, n > 3

if n < 3; n = 3; end
    
PolyPara = zeros(1, n);
PolyPara(1:3) = [1, 2, 1];

for ii = 1:n-3
    PolyPara = PolyPara + [0, PolyPara(1:n-1)];
end