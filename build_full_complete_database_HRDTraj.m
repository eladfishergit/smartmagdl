% This program generates a "theoretical" heatmap and the field sampled on a trajectory path.
% It then simulate the measurement of a smartphone magnetometer that walks
% on the path.

close all
clear
clc

load traj_data

% Earth magnetic field
Bxearth = 3e-6;
Byearth = 29e-6;
Bzearth = 33e-6;

% pitch and roll variations - in degrees converted to radians
pitch_sigma = 2*pi/180;
roll_sigma = 2*pi/180;

% Grid
Xmax = 30;% meters
Ymax = 30;% meters
mgrid =300;
xg = linspace(-Xmax/2,Xmax/2,mgrid)*1000; % in millimeters
yg = linspace(-Ymax/2,Ymax/2,mgrid)*1000; % in millimeters

maxd10 = 100; % max distance in mm for estimating the field on a grid from data on the path
maxd25 = 250; % max distance in mm for estimating the field on a grid from data on the path
maxd50 = 500; % max distance in mm for estimating the field on a grid from data on the path

% Choose the output directories 
data_dir_root = 'data_dir_root';
data_dir_root10 = 'data_dir_root\Test10\';
data_dir_root25 = 'data_dir_root\Test25\';
data_dir_root50 = 'data_dir_root\Test50\';

windowSize = 1;   %  moving window size
Nd = 10; % Number of points to add between each pair

Nsim = 100;
f = waitbar(0, 'Starting');

for ksim = 1:Nsim
    waitbar(ksim/Nsim, f, sprintf('Progress: %d %%', floor(ksim/Nsim*100)));
    pause(0.1);
    fname = [data_dir_root  'info_' num2str(ksim) '.mat'];
    fnameA = [data_dir_root  'TheorA_' num2str(ksim) '.mat'];
    fnameB = [data_dir_root  'TheorB_' num2str(ksim) '.mat'];
    fnameC = [data_dir_root  'TheorC_' num2str(ksim) '.mat'];
    fnameD = [data_dir_root  'TheorD_' num2str(ksim) '.mat'];

    fnameF = [data_dir_root  'SimA_' num2str(ksim) '.mat'];

    % compute the Ground truth
    plot_flag = 0;
    [Bxt,Byt,Bzt,Q,mPos,Mom,Diam,Len] = build_ground_truth_all(plot_flag,P);
    Bxt_e = Bxt+Bxearth;
    Byt_e = Byt+Byearth;
    Bzt_e = Bzt+Bzearth;

    save(fname,'mPos','Mom','Diam','Len')
    save (fnameA,'Bxt','Byt','Bzt')

    % Sensitivity matrix
    ds = rand(1,3)*1e-3;
    S = diag(1-ds);

    B_ground_truth = sqrt(Bxt_e.^2+Byt_e.^2+Bzt_e.^2)*1e6;
    save (fnameB,'xg','yg','Bxt_e','Byt_e','Bzt_e','B_ground_truth')

    xrot_o = Q(:,1);
    yrot_o = Q(:,2);

    Np = size(Q,1);
    % now sample the full (exact) field on the trajectory points
    Bxtraj = zeros(1,Np);
    Bytraj = zeros(1,Np);
    Bztraj = zeros(1,Np);
    Bxtraj_e = zeros(1,Np);
    Bytraj_e = zeros(1,Np);
    Bztraj_e = zeros(1,Np);
    for i = 1:Np
        xi = xrot_o(i);
        yi = yrot_o(i);
        [~,ix] = min(abs(yi-xg));
        [~,ij] = min(abs(xi-yg));
        Bxtraj(i) = Bxt(ix,ij);
        Bytraj(i) = Byt(ix,ij);
        Bztraj(i) = Bzt(ix,ij);

        Bxtraj_e(i) = Bxt_e(ix,ij);
        Bytraj_e(i) = Byt_e(ix,ij);
        Bztraj_e(i) = Bzt_e(ix,ij);

    end

    Bttraj = sqrt(Bxtraj.^2+Bytraj.^2+Bztraj.^2);
    Bttraj_e = sqrt(Bxtraj_e.^2+Bytraj_e.^2+Bztraj_e.^2);
    save (fnameC,'Bxtraj','Bytraj','Bztraj','Bttraj')

    % Densify the path trajectory
    [~, ~, newZx] = densifyTrajectory(xrot_o, yrot_o, Bxtraj, Nd);
    [~, ~, newZy] = densifyTrajectory(xrot_o, yrot_o, Bytraj, Nd);
    [~, ~, newZz] = densifyTrajectory(xrot_o, yrot_o, Bztraj, Nd);
    clear xrot yrot Bxtraj Bytraj Bztraj Bttraj Np
    Bxtraj = newZx;
    Bytraj = newZy;
    Bztraj = newZz;
    Bttraj = sqrt(Bxtraj.^2+Bytraj.^2+Bztraj.^2);
    clear newX newY newZx newZy newZz

    % Densify the path trajectory
    [newX, newY, newZx] = densifyTrajectory(xrot_o, yrot_o, Bxtraj_e, Nd);
    [~, ~, newZy] = densifyTrajectory(xrot_o, yrot_o, Bytraj_e, Nd);
    [~, ~, newZz] = densifyTrajectory(xrot_o, yrot_o, Bztraj_e, Nd);
    xrot = newX;
    yrot = newY;
    Bxtraje = newZx;
    Bytraje = newZy;
    Bztraje = newZz;
    Bttraje = sqrt(Bxtraje.^2+Bytraje.^2+Bztraje.^2);
    Np=length(xrot);

    save(fnameD,'xrot', 'yrot','Bxtraje','Bytraje','Bztraje','Bttraje')

    % Pixelize the picture (in fact 1 pixel)
    % Get dimensions of the input matrix
    [rows, cols] = size(B_ground_truth);

    % Calculate the size of the output matrix
    outputRows = floor(rows / windowSize);
    outputCols = floor(cols / windowSize);
    rb =(0:outputRows-1)* windowSize + 1;
    cb =(0:outputCols-1)* windowSize + 1;

    output10 = NaN(outputRows,outputCols);
    output25 = NaN(outputRows,outputCols);
    output50 = NaN(outputRows,outputCols);

    for i = 1:outputRows
        for j = 1:outputCols
            d = sqrt((xrot-xg(rb(j))).^2+(yrot-yg(cb(i))).^2);
            ind10 = find(d<=maxd10);
            if ~isempty(ind10)
                output10(i,j) = mean(Bttraje(ind10));
            end
            ind25 = find(d<=maxd25);
            if ~isempty(ind25)
                output25(i,j) = mean(Bttraje(ind25));
            end
            ind50 = find(d<=maxd50);
            if ~isempty(ind50)
                output50(i,j) = mean(Bttraje(ind50));
            end
        end
    end

    % Display the result
    Btheor_tot10 = output10;
    Btheor_tot10(isnan(output10)) = median(output10(~isnan(output10)));
    Btheor_tot25 = output25;
    Btheor_tot25(isnan(output25)) = median(output25(~isnan(output25)));
    Btheor_tot50 = output50;
    Btheor_tot50(isnan(output50)) = median(output50(~isnan(output50)));

    fnameE10 = [data_dir_root10 'TheorE10_',num2str(ksim), '.mat'];
    save(fnameE10,'Btheor_tot10')
    fnameE25 = [data_dir_root25 'TheorE25_',num2str(ksim), '.mat'];
    save(fnameE25,'Btheor_tot25')
    fnameE50 = [data_dir_root50 'TheorE50_',num2str(ksim), '.mat'];
    save(fnameE50,'Btheor_tot50')

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % sort the pitch and roll angle within the given intervals
    pitch = randn(1,Np)*pitch_sigma;
    roll = randn(1,Np)*roll_sigma;

    save(fname,'pitch','roll','-append')

    % Calibration matrix. It includes the misalignement and the soft-iron
    % effect. The overall effect transform the "ideal" sphere defined by Bt
    % when it is unperturbated by magnetic anomaly into an ellispsoid
    dev_from_diag = 0.25;
    dev_from_offdiag = 0.05;
    rdiag = 1-dev_from_diag+2*rand(1,3)*dev_from_diag;
    rodiag = -dev_from_offdiag+2*rand(1,6)*dev_from_offdiag;
    T = diag(rdiag)+[0 rodiag(1:2);rodiag(3) 0 rodiag(4);rodiag(5:6) 0 ];
    save(fname,'T','-append')

    % read angles and prepare rotation matrix
    dxr = diff(xrot);
    dyr = diff(yrot);
    % Exclude points with dxr = 0 since no angle can be defined for such values

    ig = find(dxr~=0);
    xroti  = xrot(ig);
    yroti  = yrot(ig);
    headg = atan2(diff(yroti),diff(xroti));
    Npp = length(ig);
    azi = zeros(1,Npp);
    azi(1:end-1) = headg;
    azi(end) =headg(end);

    Bxtrajj = Bxtraj(ig) ;Bytrajj = Bytraj(ig) ;Bztrajj = Bztraj(ig) ;
    Btraj = [Bxtrajj' Bytrajj' Bztrajj'];

    % Gaussian noise epsilon
    mu_noise = 0;
    sigma_noise = 1; % microTesla
    sigma_noise = sigma_noise*1e-6;
    eps = mu_noise+randn(Npp,3)*sigma_noise;
    save(fname,'eps','-append')

    Bxx = zeros(1,Npp);
    Byy = zeros(1,Npp);
    Bzz= zeros(1,Npp);

    for k = 1:Npp
        ca  = cos(azi(k));
        sa = sin(azi(k));
        cb  = cos(pitch(k));
        sb = sin(pitch(k));
        cr  = cos(roll(k));
        sr = sin(roll(k));
        R = [ ca*cb  ca*sb*sr-sa*cr  ca*sb*cr+sa*sr; sa*cb sa*sb*sr+ca*cr sa*sb*cr-ca*sb; -sb cb*sr cb*cr];
        Bearth_rot = R*[Bxearth Byearth Bzearth]';
        bunrot(1) = Btraj(k,1);
        bunrot(2) = Btraj(k,2);
        bunrot(3) = Btraj(k,3);
        brot = R*bunrot';
        Ber(1) = brot(1)+Bearth_rot(1);
        Ber(2) = brot(2)+Bearth_rot(2);
        Ber(3) = brot(3)+Bearth_rot(3);

        bcal = S*(T*Ber');

        Bxx(k) = bcal(1) ;
        Byy(k) = bcal(2) ;
        Bzz(k) = bcal(3) ;

    end

    clear Bx By Bz

    %add an offset (from hard iron and other source)
    sigma_offset = 0.1; % in microTesla
    sigma_offset = sigma_offset*1e-6;
    Boffset = randn(1,3)*sigma_offset;
    save(fname,'Boffset','-append')

    Bx = Bxx'+eps(:,1)+Boffset(1);
    By = Byy'+eps(:,2)+Boffset(2);
    Bz = Bzz'+eps(:,3)+Boffset(3);

    Btt = sqrt(Bx.^2+By.^2+Bz.^2);

    % Compute regular interpolation
    [Xq,Yq] = meshgrid(xg,yg);
    clear Bintr
    Bintr = griddata(xroti,yroti,Btt,Xq,Yq,"natural");
    bad_ind = find(isnan(Bintr));

    Bb = Bintr;
    Bb(isnan(Bintr)) = median(Bintr(~isnan(Bintr)));
    clear Bintr
    Bintr = Bb;

    save(fnameF,'xroti', 'yroti','Bx','By','Bz','Btt','Bintr')

    % Pixelize the picture (here 1 pixel)
    % Get dimensions of the input matrix
    [rows, cols] = size(B_ground_truth);

    % Calculate the size of the output matrix
    outputRows = floor(rows / windowSize);
    outputCols = floor(cols / windowSize);
    rb =(0:outputRows-1)* windowSize + 1;
    cb =(0:outputCols-1)* windowSize + 1;

    output = NaN(outputRows,outputCols);
    for i = 1:outputRows
        for j = 1:outputCols
            d = sqrt((xroti-xg(rb(j))).^2+(yroti-yg(cb(i))).^2);
            ind10 = find(d<=maxd10);
            if ~isempty(ind10)
                output10(i,j) = mean(Btt(ind10));
            end
            ind25 = find(d<=maxd25);
            if ~isempty(ind25)
                output25(i,j) = mean(Btt(ind25));
            end
            ind50 = find(d<=maxd50);
            if ~isempty(ind50)
                output50(i,j) = mean(Btt(ind50));
            end
        end
    end

    Bsim_tot10 = output10;
    Bsim_tot10(isnan(output10)) = median(output10(~isnan(output10)));

    Bsim_tot25 = output25;
    Bsim_tot25(isnan(output25)) = median(output25(~isnan(output25)));

    Bsim_tot50 = output50;
    Bsim_tot50(isnan(output50)) = median(output50(~isnan(output50)));

    fnameG10 = [data_dir_root10 'SimB10_',num2str(ksim), '.mat'];
    save(fnameG10,'Bsim_tot10')
    fnameG25 = [data_dir_root25 'SimB25_',num2str(ksim), '.mat'];
    save(fnameG25,'Bsim_tot25')
    fnameG50 = [data_dir_root50 'SimB50_',num2str(ksim), '.mat'];
    save(fnameG50,'Bsim_tot50')

end
