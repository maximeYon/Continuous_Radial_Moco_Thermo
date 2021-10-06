%% Main function for Continuous cardiac thermometry via simultaneous
% catheter tracking and undersampled radial golden angle acquisition
% for radiofrequency ablation monitoring
%% Maxime Yon 03/06/2021
% maxime.yon@gmail.com

%% Clean
clearvars; close all; clc;

%% Addpath
current_path = pwd;
addpath(genpath([current_path filesep 'function_needed' filesep 'functions_supp']));
addpath([current_path filesep 'function_needed' filesep 'wgrissom-k-space-thermometry']);
addpath([current_path filesep 'function_needed' filesep 'ismrm_sunrise_matlab-master']);
addpath([current_path filesep 'function_needed' filesep 'ismrmrd' filesep 'matlab']);
run([current_path filesep 'function_needed' filesep 'fessler_functions'  filesep 'irt' filesep 'setup.m'])

addpath([current_path filesep 'function_needed' filesep 'gpuNUFFT/CUDA/bin']);
addpath([current_path filesep 'function_needed' filesep 'gpuNUFFT/gpuNUFFT']);

%% Define code Options
%% OPTIONS
% Generals and pre-treatments
Resolution = 0.8594; % Spatial resolution of the images and temperature maps in mm
Number_of_spokes_per_image = 40; % if empty all the spokes will be used to create one image
micro_coil_selection = 4; % 1 tip; 2 proximal; 3 distal; 4 averaging
smoo = 2; % Smoothing of the motion curves
N_resp_phase = 10; % Number of respiratory phase for Thermometry library 12
N_card_phase = 1; % Number of cardiac phase for Thermometry library
Calibration = 3000; % Number of projections for calibration
display = 1 ; % 1 to display intermediate figures

% Thermometry
lambda = 0.1;
beta = 0.05;
algp = struct; acqp = struct;
algp.order = 6; % polynomial order
algp.lam = [lambda -1]; % 0.1 -1 %  first entry is sparsity for temp phase, second entry is sparsity for attenuation 0.1
algp.beta = beta; % roughness regularization parameter
clearvars lambda beta;


%% Load the data and noise
% data
filename='/home/maximeyon/mount/maxime.yon/Data/2020-07-23_ablationGel_Motion_new/CV_Tracking_Maxime_GA_abl7_50W_60s.h5'; % abl 4, 1-2 are good

%Noise : if no noise comment the filename line
filename_noise ='/home/maximeyon/mount/maxime.yon/Data/2020-07-23_ablationGel_Motion_new/NOISE_42563_426664_426669_373_20200723-161740.h5';


%% Open the .h5 data
if exist('filename_noise','var')==1
    [Kspace_proj,kspace_cath,parameter] = read_Siemens_Radial_Catheter_Kspace(filename,'true',filename_noise);
else
    [Kspace_proj,kspace_cath,parameter] = read_Siemens_Radial_Catheter_Kspace(filename,'true');
end

%% Save Parameters
parameter.Resolution = Resolution;
parameter.Ny = parameter.Nx;
parameter.Nspokes = Number_of_spokes_per_image; % if empty all the spokes will be used to create one image
parameter.micro_coil_selection = micro_coil_selection; % 1 tip; 2 proximal; 3 distal; 4 averaging
parameter.smoo = smoo;
parameter.N_resp_phase = N_resp_phase;
parameter.N_card_phase = N_card_phase;
parameter.im_cal = parameter.N_resp_phase.*parameter.N_card_phase;
parameter.display = display;
parameter.Nimages = floor(size(Kspace_proj,2)/parameter.Nspokes);
parameter.Calibration = Calibration;

clearvars Number_of_spokes_per_image micro_coil_selection smoo N_resp_phase N_card_phase im_cal;

%% Remove oversampling
Kspace_proj = Kspace_proj(size(Kspace_proj,1)*0.25+1:size(Kspace_proj,1)*0.75,:,:);
kspace_cath = kspace_cath(size(kspace_cath,1)*0.25+1:size(kspace_cath,1)*0.75,:,:);

%% Compute the trajectory and DCF for the whole dataset
osf=1; GA=2; readout =size(Kspace_proj,1);
[ trajtotale, ~ , angles] = ComputeRADIALtraj(readout,size(Kspace_proj,2),osf,GA);
trajtotale = reshape(trajtotale,readout,size(Kspace_proj,2),3);
trajtotale = single(trajtotale);

% DCF
trajDCF = double(reshape(trajtotale, size(trajtotale,1)*size(trajtotale,2),3));
osf=1; verbose = 0; numIter = 10; effMtx  = readout;
DCF = sdc3_MAT(trajDCF',numIter,effMtx,verbose,osf);
DCF = reshape(DCF,readout,size(DCF,1)/readout,1);
clearvars osf verbose numIter effMtx GA trajDCF;

%% compute the angle of the projections
angle_proj= (angles/pi*180)+90;
parameter.angle_proj = ((angle_proj/360-floor(angle_proj/360))*360);
clearvars angles;

% trajtotale = trajtotale./2;

%% Catheter tracking
[parameter,Centroid,Z_intensity] = my_catheter_tracking(parameter,kspace_cath);

%% Catheter motion filtering
parameter.Fs = 1/(parameter.TR.*10.^-3*2);
[Centroid] = Quantitative_frequency_filter(Centroid,parameter.Fs,parameter.smoo);
[Z_intensity] = Quantitative_frequency_filter(Z_intensity,parameter.Fs,parameter.smoo./2);

%% 2D rigid motion correction
[parameter,Kspace_proj] = my_2D_rigid_MOCO(parameter,Centroid,Kspace_proj,trajtotale);

%% Create ECG and respiratory phase list
[parameter,list_proj_ECG] = my_ECG_sorting(parameter,Centroid); % Here the spot intensity is chosen for ECG
% [parameter,list_proj_Resp] = my_Respiratory_sorting(parameter,Z_intensity);% Here the spot intensity is chosen for Resp otherwise Centroid (XY motion) can be used
[parameter,list_proj_Resp] = my_Respiratory_sorting(parameter,Centroid);% Here the spot intensity is chosen for Resp otherwise Centroid (XY motion) can be used

%% Check phases resp
ind_centroid = 1:size(Centroid,1);
figure(101)
hold on
for ind_Resp = 1:size(list_proj_Resp,2)
    ind_without_zeros = list_proj_Resp(:,ind_Resp);
    ind_without_zeros = ind_without_zeros(ind_without_zeros~=0);
    plot(ind_centroid(ind_without_zeros),Centroid(ind_without_zeros),'*')    
end

%% Resolution reduction to increase SNR and time resolution
parameter.Npixel = (round((parameter.FOVx/parameter.Resolution)/2)).*2;
Ind_zeros = (size(Kspace_proj,1)-parameter.Npixel)/2;
Kspace_proj(1:Ind_zeros,:,:) = 0; Kspace_proj(end-Ind_zeros:end,:,:) = 0;
clearvars Ind_zeros;

%% Undersampled Thermometry
% Initialization and library computing
[parameter,acqp,L] = my_undersampled_thermo_init(parameter,acqp,trajtotale,list_proj_Resp,list_proj_ECG,Kspace_proj);

%parallele Thermometry computing
[parameter,Thermometry] = my_undersampled_thermo_computing(parameter,acqp,algp,L,Kspace_proj,trajtotale);

%% gpuNUFFT magnitude reconstruction
[parameter,img_comb,img_comb_full] = my_radial_gpuNUFFT(parameter,Kspace_proj,trajtotale,DCF);

%% Save variables
Thermometry = single(Thermometry);
img_comb = single(img_comb); img_comb_full = single(img_comb_full);
save(['/home/maximeyon/mount/maxime.yon/Grissom_for_Olivier/gel_motion/Ablation_Gel_motion' num2str(parameter.TE) '_ms_' num2str(parameter.Nspokes) '_spokes_noMOCO.mat'],'img_comb','img_comb_full','Thermometry','parameter','-v7.3');

% save(['/home/maximeyon/mount/maxime.yon/Grissom_for_Olivier/gel_motion/Raw_data_Gel_motion_' num2str(parameter.TE) '_ms_.mat'],'Centroid','DCF','Kspace_proj','Kspace_proj_MOCO','list_proj_Resp','parameter','trajtotale','-v7.3');

% %% Display
% DynamicDisplay_single_im(img_comb_full,'Temperature',Thermometry);

