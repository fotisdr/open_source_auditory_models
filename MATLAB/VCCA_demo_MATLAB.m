function [handleFigure, structOutputs] = VCCA_demo_MATLAB(bListen)
% function [handleFigure, structOutputs] = VCCA_demo_MATLAB(bListen)
%
% 1. Description:
%     VCCA demo using MATLAB or GNU Octave. In this demo, the sound
%     'scribe_male_talker.wav' is processed using the auditory model
%     named relanoiborra2019_featureextraction.m (available within the
%     AMT toolbox) using a fixed characteristic frequency of about 
%     2000 Hz and comparing the normal-hearing cochlear configuration
%     with simulations adopting a hearing-impaired cochlea.
%
% 2. Stand-alone example:
% % 2.1 Processing, listening to the resulting sounds and storing the figure:
% bListen = 1;
% [handleFigure, structOutputs] = VCCA_demo_MATLAB(bListen);
% fnameHere = 'figForDemo-relanoiborra2019.png'; % arbitrary name
% bDoFig = ~exist(fnameHere,'file'); % only saved if it does not exist
% if bDoFig
%     exp2eval = sprintf('print -dpng ''%s''',fnameHere);
%     eval(exp2eval);
% end
% fnameSound = 'scribe_male_talker-1-NH-output.wav';
% bStoreSound = ~exist(fnameSound,'file'); % only saved if it does not exist
% if bStoreSound
%    audiowrite(fnameSound, structOutputs.outsig_afb, structOutputs.fs);
% end
% fnameSound = 'scribe_male_talker-2-HI-output.wav';
% bStoreSound = ~exist(fnameSound,'file'); % only saved if it does not exist
% if bStoreSound
%    audiowrite(fnameSound, structOutputs.outsigHI_afb, structOutputs.fs);
% end
%
% % 2.2 Only processing:
% VCCA_demo_MATLAB;
%
% Author: Alejandro Osses
% Date: 12/06/2024

structOutputs = [];

if nargin == 0
    bListen = 0; % skip the listening to the sounds
end
clc

dir_this_script = [fileparts(mfilename('fullpath')) filesep]; % directory of this script
cd(dir_this_script);

%%% 1. Initialisation:
if ~exist('amt_start_for_open_source_auditory_models.m','file')
    dirAMT = [dir_this_script 'amtoolbox-1.5.0' filesep]; % one level up

    addpath(dirAMT);
    amt_start_for_open_source_auditory_models;
end

%%% 2. Loading audio file:
dirSound = [fileparts(dir_this_script(1:end-1)) filesep 'Python' filesep 'ICNet' filesep];
fileSound = [dirSound 'scribe_male_talker.wav'];
[insig,fs] = audioread(fileSound);

durInitial = 7.5; % s, initial duration across the sound
durFinal = 12.5; % s, final duration

idxI = round(durInitial*fs)+1;
idxF = round(durFinal*fs);
insig = insig(idxI:idxF,1);

if bListen
    %%% If you want to listen to the input signal:
    fprintf('%s.m: Listening to %s\n',mfilename,fileSound);
    sound(insig,fs);
end

dBFS = 94;
lvl = 20*log10(rms(insig))+dBFS;
fprintf('Level of the input signal=%.1f dB (assuming a dB FS equal to %.0f dB SPL)\n',lvl,dBFS);

%%% 3. Process:
% flags_for_the_model = {'no_internalnoise','no_ihc','no_an'}; % 'flow',f0,'fhigh',f0,'basef',f0
flags_for_the_model = {'no_internalnoise','no_ihc','no_an'}; % 'flow',f0,'fhigh',f0,'basef',f0
[~, ~, outsig_afb, fc] = relanoiborra2019_featureextraction(insig, fs, flags_for_the_model{:});

idx_fc = find(fc>2000,1);

outsig_afb = outsig_afb(:,idx_fc);

if bListen
    %%% If you want to listen to the normal-hearing output:
    fprintf('%s.m: Listening to the NH output for %s\n',mfilename,fileSound);
    sound(outsig_afb,fs);
end

[~, ~, outsigHI_afb] = relanoiborra2019_featureextraction(insig, fs,...
                            'subject', 'HIx', ... % keyval to choose the hearing impaired profile
                            flags_for_the_model{:});
outsigHI_afb = outsigHI_afb(:,idx_fc);

if bListen
    %%% If you want to listen to the hearing-impaired output:
    fprintf('%s.m: Listening to the HI output for %s\n',mfilename,fileSound);
    sound(outsigHI_afb,fs);
end

%%% 4. Plotting the resulting waveforms:
t = (1:size(outsig_afb,1))/fs + durInitial;

figure;
plot(t, outsig_afb,'b-'); hold on; grid on;
plot(t, outsigHI_afb,'r-');

xlabel('Time (s)');
ylabel('Cochlear filter bank output (a. u.)');
title(sprintf('Analysis for signal within the band at %.1f Hz',fc(idx_fc)));

legend('NH profile','HIx profile');

xlim([durInitial durFinal]);

Pos = get(gcf,'Position');
Pos(3:4) = [1200 420];
set(gcf,'Position',Pos);

handleFigure = gcf;

structOutputs.outsig_afb = outsig_afb;
structOutputs.outsigHI_afb = outsigHI_afb;
structOutputs.fs = fs;
