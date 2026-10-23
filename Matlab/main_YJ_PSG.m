%% 1. Select one recording on every run (no stale workspace result reuse)
base='D:\Meeting\2026 1030 meeting'; % Explicit project path, also for Run Section
inputPath=''; result=[]; % Clear prior selection before opening dialogs
addpath(base);
addpath(fullfile(base,'Function'),'-begin');
[systemIndex,confirmed]=listdlg( ...
    'PromptString','PSG 형식을 선택하세요', ...
    'SelectionMode','single', ...
    'ListString',{'EDF: 파일선택','EMB / Embla (.ebm): 폴더선택','NOX (.ndf): 폴더선택'}, ...
    'ListSize',[370 100]);
if ~confirmed
    fprintf('Selection cancelled.\n');
    return;
end
if systemIndex==1
    [selectedFile,selectedFolder]=uigetfile( ...
        {'*.edf;*.EDF','EDF recording'},'EDF 파일 하나 선택', ...
        fullfile(base,'Data'),'MultiSelect','off');
    if isequal(selectedFile,0), return; end
    inputPath=fullfile(selectedFolder,selectedFile);
else
    formats={'EBM','NDF'};
    selectedFormat=formats{systemIndex-1};
    inputPath=uigetdir(fullfile(base,'Data'), ...
        sprintf('%s 채널 파일들이 들어 있는 검사 폴더 선택',selectedFormat));
    if isequal(inputPath,0), return; end
    entries=dir(inputPath); entries=entries(~[entries.isdir]);
    assert(any(endsWith(string({entries.name}),['.' lower(selectedFormat)],'IgnoreCase',true)), ...
        'PSG:Selection','선택한 폴더에 %s 파일이 없습니다. 개별 검사 폴더를 선택하세요.',selectedFormat);
end

%% 2. Import the selected recording; never prompt to save or write results
% This section is safe to run independently after a cancelled selection.
% base='D:\Meeting\2026 1030 meeting';
assert(isfolder(fullfile(base,'Function')),'PSG:Base','Project Function folder not found: %s',base);
addpath(base,'-begin');
addpath(fullfile(base,'Function'),'-begin');
result=[];
validSelection=exist('inputPath','var') && ...
    ((ischar(inputPath)&&isrow(inputPath)&&~isempty(inputPath)) || ...
     (isstring(inputPath)&&isscalar(inputPath)&&~ismissing(inputPath)&&strlength(inputPath)>0));
if ~validSelection
    fprintf('선택된 경로가 없습니다. 1번 파일/폴더 선택 섹션부터 실행하세요.\n');
    return;
end
if ~(isfolder(inputPath)||isfile(inputPath))
    fprintf('선택한 경로가 존재하지 않습니다. 1번 섹션에서 다시 선택하세요.\n');
    return;
end
result=PSG_Read(inputPath,'Save',false,'Reload',true);
if isempty(result), return; end
disp(result.channelTable);

%% 3. Plot all available channels and scoring, in the attached stacked format
if ~exist('result','var') || isempty(result)
    fprintf('불러온 데이터가 없습니다. 파일/폴더 선택 후 import를 먼저 실행하세요.\n');
    return;
end
plotWindow=[];        % [] = recording after initial exclusion; or [startDatetime endDatetime]
channelsPerTab=7;     % one figure, seven channels per tab
skipInitialSeconds=180; % Display only; original signals and full event counts are retained
yAxisRangeFraction=0.30; % Y-axis width relative to automatic limits
[figures,Events,SleepStages,PlotData,EventCounts]=PSG_Plot(result,'Window',plotWindow, ...
    'ChannelsPerTab',channelsPerTab,'SkipInitialSeconds',skipInitialSeconds,'YAxisRangeFraction',yAxisRangeFraction);
% No PSG_Save, save, savefig, exportgraphics, or writetable calls here.
