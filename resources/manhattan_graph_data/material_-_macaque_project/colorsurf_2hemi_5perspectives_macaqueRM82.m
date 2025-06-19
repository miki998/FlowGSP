%
%
% Alessandra Griffa
% University of Geneva
% Ecole polytechnique federale de Lausanne EPFL | MIPLab
% Jun 2021
%

function [k1,k2,k3,k4,k5] = colorsurf_2hemi_5perspectives_macaqueRM82(qinfo, CM, llist, my_lighting, nview)


% Initialize figures handles
k1 = NaN;
k2 = NaN;
k3 = NaN;
k4 = NaN;
k5 = NaN;


% FACEs and VERTICEs for both hemisphere
label = qinfo.Surf.labels;  % CAREFULL: SOME LABEL ARE EQUAL TO 0 - WHAT TO DO WITH THESE LABELS ??
v = qinfo.Surf.vertices; % v: vertices; f: faces of triangles
f = qinfo.Surf.faces;
nv = size(v,1); % number of vertices


%% Split left and right hemisphere
if nview == 5
    ir = find(label > 0 & label < 100); % indexes right hemisphere vertives
    v_rh = v(ir,:);
    label_rh = label(ir);
    nv_rh = length(ir);
    f_rh = zeros(size(f));
    mask = zeros(size(f));
    for i = 1:nv_rh
        mask(f==ir(i)) = 1;
        f_rh(f==ir(i)) = i;
    end
    f_rh = f_rh(find(sum(mask,2)==3),:);

    il = find(label > 100); % indexes left hemisphere vertices
    v_lh = v(il,:);
    label_lh = label(il);
    nv_lh = length(il);
    f_lh = zeros(size(f));
    mask = zeros(size(f));
    for i = 1:nv_lh
        mask(f==il(i)) = 1;
        f_lh(f==il(i)) = i;
    end
    f_lh = f_lh(find(sum(mask,2)==3),:);
end




%% Assign a color to each vertex
% Generate a matrix vColor(#verices x 3), where each line contains the
vColor = ones(nv,size(CM,2)) .* 211/255;    % default color: grey
for i = 1:length(llist)
    ix = find(label == llist(i));
    if ~isempty(ix)
        thisColor = CM(i,1:end);
        vColor(ix,1:end) = repmat(thisColor,length(ix),1);
    end
end
if nview == 5
    vColor_rh = vColor(ir,:);
    vColor_lh = vColor(il,:);
end

    

%% PLOT
% PLOT SINGLE HEMISPHERES
% Surface plot according to vColor colortable
%close all;
k1 = figure;%('Visible','on');
p = patch('faces',f,'vertices',v,'facecolor','flat','edgecolor','none','facealpha',1);
set(p,'FaceVertexCData',vColor);
daspect([1 1 1]);
view([-90 0]);
camlight('headlight');
lighting(my_lighting);
axis tight off;
set(gcf,'Color','None');

k2 = figure;%('Visible','on');
p = patch('faces',f,'vertices',v,'facecolor','flat','edgecolor','none','facealpha',1);
set(p,'FaceVertexCData',vColor);
daspect([1 1 1]);
view([90 0]);
camlight('headlight');
lighting(my_lighting);
axis tight off;
set(gcf,'Color','None');

k3 = figure;
p = patch('faces',f,'vertices',v,'facecolor','flat','edgecolor','none','facealpha',1);
set(p,'FaceVertexCData',vColor);
daspect([1 1 1]);
view(3); % view(3) sets the default three-dimensional view, az = ???37.5, el = 30
view([50 -40 70]); % sets the viewpoint to the Cartesian coordinates x, y, and z
camlight; % creates a light right and up from camera
lighting(my_lighting);
set(gcf,'color','w');
view([0 90]);
axis off;
set(gcf, 'Color', 'None');



% PLOT 5 views
% 5 view
if nview == 5
    
    k4 = figure;
    ha = tight_subplot(2,2,[0.01 0.2],[.01 .01],[.01 .01]);

    % LEFT HEMI SIDE
    axes(ha(1)); 
    p = patch('faces',f_lh,'vertices',v_lh,'facecolor','flat','edgecolor','none','facealpha',1);
    set(p,'FaceVertexCData',vColor_lh);
    daspect([1 1 1]);
    view([-90 0]);
    camlight('headlight');
    %lighting gouraud; % specify lighting algorithm        
    lighting(my_lighting);
    %alpha(0.3)
    axis tight off;    

    % RIGHT HEMI SIDE
    axes(ha(2)); 
    p = patch('faces',f_rh,'vertices',v_rh,'facecolor','flat','edgecolor','none','facealpha',1);
    set(p,'FaceVertexCData',vColor_rh);
    daspect([1 1 1]);
    view([90 0]);
    camlight('headlight');
    lighting(my_lighting);
    %alpha(0.3)
    axis tight off;

    % RIGHT HEMI MEDIAL
    axes(ha(4)); 
    p = patch('faces',f_rh,'vertices',v_rh,'facecolor','flat','edgecolor','none','facealpha',1);
    set(p,'FaceVertexCData',vColor_rh);
    daspect([1 1 1]);
    view([-90 0]);
    camlight('headlight');
    lighting(my_lighting); % specify lighting algorithm
    %alpha(0.3)
    axis tight off;   

    % LEFT HEMI MEDIAL
    axes(ha(3)); 
    p = patch('faces',f_lh,'vertices',v_lh,'facecolor','flat','edgecolor','none','facealpha',1);
    set(p,'FaceVertexCData',vColor_lh);
    daspect([1 1 1]);
    view([90 0]);
    camlight('headlight');
    lighting(my_lighting);
    %alpha(0.3)
    axis tight off; 

    set(gcf, 'Color', 'None')

    % PLOT BOTH HEMISPHERE, TRANSVERSAL VIEW
    k5 = figure;
    ha = tight_subplot(1,1,[0 0],[.2 .2],[.2 .2]);
    axes(ha(1));
    p = patch('faces',f_rh,'vertices',v_rh,'facecolor','flat','edgecolor','none','facealpha',1);
    set(p,'FaceVertexCData',vColor_rh);
    p = patch('faces',f_lh,'vertices',v_lh,'facecolor','flat','edgecolor','none','facealpha',1);
    set(p,'FaceVertexCData',vColor_lh);
    daspect([1 1 1]);
    view(3); % view(3) sets the default three-dimensional view, az = ???37.5, el = 30
    view([50 -40 70]); % sets the viewpoint to the Cartesian coordinates x, y, and z
    camlight; % creates a light right and up from camera
    lighting(my_lighting);
    set(gcf,'color','w');
    view([0 90]);
    axis off;
    set(gcf, 'Color', 'None');

end















