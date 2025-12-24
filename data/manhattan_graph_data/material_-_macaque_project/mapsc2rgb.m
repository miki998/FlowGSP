%
% mapsc2rgb
%
% Converts a MxN matrix mat to a MxNx3 matrix outrgbmat, according to the
% input colormap map (e.g.: map = jet; close;).
% Note that indexed values in mat are rescaled to the range [1,n], with n
% number of rgb entries in the colormap map.
% The user can provide predefined maximum and minimum values for the colormap. 
%
% Alessandra Griffa
% École polytechnique fédérale de Lausanne EPFL
% Signal Processing Laboratory 5 LTS5 
% May 2013
%

function outrgbmat = mapsc2rgb(mat, map, minval, maxval)

    % Handle inputs
    if nargin < 3
        minval = min(mat(:));
        maxval = max(mat(:));
    end

    % Initialize output rgb image
    outrgbmat = zeros(size(mat,1),size(mat,2),3);
    
    % Rescale input matrix values to the range [1,size(map,1)]
    if maxval < max(mat)
        mat(mat>maxval) = maxval;
    end
    if minval > min(mat)
        mat(mat<minval) = minval;
    end 
    ix = ((mat-minval) * (size(map,1)-1) / (maxval-minval)) + 1;

    % Generate rgb output image
    for i = 1:1:size(mat,1)
        for j = 1:1:size(mat,2)
            outrgbmat(i,j,:) = map(round(ix(i,j)),:);
        end
    end
    
end

