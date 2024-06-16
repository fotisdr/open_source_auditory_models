function amt_start_for_open_source_auditory_models(varargin)
% function amt_start_for_open_source_auditory_models(varargin)
%
% This function starts the auditory modelling toolbox (AMT) in its
%   version 1.5, but only adding a minimal number of directories
%   as required by the MATLAB codes used in open_source_auditory_models
%   repository.
%
% Note that AMT is a toolbox that is licenced under the GPL version 3.0.
%   Specific details about the licence should be checked at the official
%   AMT website (amtoolbox.org). You should always refer to the latest
%   AMT version.
%
%   #Author: Alejandro Osses (2024)

dir_this_script = [fileparts(mfilename('fullpath')) filesep]; % directory of this script
addpath(dir_this_script);

dir_common = [dir_this_script 'common' filesep];
addpath(dir_common);

dir_core = [dir_this_script 'core' filesep];
addpath(dir_core);

dir_data = [dir_this_script 'data' filesep];
addpath(dir_data);

dir_defaults = [dir_this_script 'defaults' filesep];
addpath(dir_defaults);

dir_mex = [dir_this_script 'mex' filesep];
addpath(dir_mex);

dir_modelstages = [dir_this_script 'modelstages' filesep];
addpath(dir_modelstages);

dir_thirdparty = [dir_this_script 'thirdparty' filesep];
p = genpath(dir_thirdparty);
addpath(p)

%where are we?
resetpath = pwd;
basepath = [fileparts(mfilename('fullpath')) filesep];

%how should I display?
if any(strcmp(varargin, 'silent'))
    dispFlag = 'silent';
    silent = 1;
elseif any(strcmp(varargin, 'documentation'))
    dispFlag = 'documentation';
    silent = 0;
else
    dispFlag = 'verbose';
    silent = 0;
end

if any(strcmp(varargin, 'install'))
    install = 1;
    dispFlag = 'verbose';
else
    install = 0;
end

%% In Octave, disable annoying warnings
if exist('OCTAVE_VERSION','builtin')
  warning('off','Octave:shadowed-function'); % Functions shadowing other functions
  warning('off','Octave:savepath-local'); % Saving the search paths locally
end

% get the default configuration without having the paths added yet
definput.keyvals.path=basepath;
cd(fullfile(basepath,'defaults'));
definput = arg_amt_configuration(definput);
cd(resetpath);
definput.keyvals.amtrunning = 1;

% display splash screen
if ~silent
  disp(' ');
  disp('****************************************************************************');
  disp(' ');
  disp(['The Auditory Modeling Toolbox (AMT) version ', definput.keyvals.version{1}(5:end),'.']);
  disp('Brought to you by the AMT Team. See http://amtoolbox.org for more information.');
  disp(' ');
end

% thirdpartypath=fullfile(basepath,'thirdparty');
%% Checking and installing the obligatory io package if in Octave
if exist('OCTAVE_VERSION','builtin')
  if ~local_loadoctpkg('io')
    error(['Cannot load the package "io".'  ...
           'Check the package list with "pkg list" and/or install the package with "pkg install io"']);
  end
end

%% Compile binaries on install
if install
    try
        amt_mex_for_open_source_auditory_models;
    catch
        error('AMT_MEX failed.');
    end
end

%%% Display the internal configuration
if ~silent
  disp(' ');
  disp('****************************************************************************');
  disp(' ');
  disp('The AMT is released under multiple licenses, with the GPL v3 being the primary one.');
  disp('  For models NOT licensed under the GPL3, the corresponding license ');
  disp('  is displayed on the first run.');
  disp(' ');
end

function filepath = local_loadtoolbox(toolboxname, targetfile, installationpath)
% Load a thirdparty toolbox to the AMT if locally available

if exist(targetfile,'file')
  filepath = fileparts(which(targetfile));  % file available, nothing to do here.
else
  if exist('OCTAVE_VERSION','builtin')
  filepath = rfsearch (installationpath, targetfile, 2);
  if ~isempty(filepath), filepath = fullfile(installationpath,fileparts(filepath)); end
  else
    filepath = dir(fullfile(installationpath, '**', targetfile)); %check if targetfile exists somewhere
    if ~isempty(filepath), filepath = filepath.folder; end
  end
  if ~isempty(filepath) %if found, great, let's add it to the path
    disp(['* Searching for ' upper(toolboxname) ' toolbox:']);
    disp(['    Package found in ' filepath '.']);
    disp('    Path added to the search path for later.');
    addpath(filepath);
    savepath;
  else
    filepath = []; % report not loaded
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function loaded = local_loadoctpkg(pkgname)
% Loads an Octave package, if not loaded
% Returns 0 if not installed or not being able to load
loaded = 0;
[~,info]=pkg('list');
for ii = 1:numel(info)
  if strcmp(pkgname, info{ii}.name)
    if ~info{ii}.loaded
      try
        eval(sprintf('pkg load %s', pkgname));
        loaded = 1;
      catch
        loaded = 0;
      end
    else
      loaded = 1;
    end
  end
end
