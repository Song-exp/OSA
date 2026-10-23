function summary = main_YJ_PSG_All(dataRoot, varargin)

% PSG: main_YJ_PSG_All();
% CBCT: main_YJ_PSG_All([], 'Process','CBCT');
% PSG 재처리: main_YJ_PSG_All([], 'PatientIDs',[4 7 12 23 25 33 34 43 45 50 53 54 72 74 76 78 89 91 93 101 102], 'Overwrite',true);
% 저장 파일 읽어오기
% S = load('G:\데이터정리\Result\mat\P_52.mat');
% plot(S.Signals.SpO2.time, S.Signals.SpO2.data)
%% 1. Setting
base = 'E:\ClinicalData\[성인] ENT_ORTHO\2026 1012 빅데이터학과 파일 전달\Matlab';
defaultDataPath = 'E:\ClinicalData\[성인] ENT_ORTHO\2026 1012 빅데이터학과 파일 전달\Data\PSG';
defaultFigurePath = 'E:\ClinicalData\[성인] ENT_ORTHO\2026 1012 빅데이터학과 파일 전달\Result\PSG\Figure';
defaultMatPath = 'E:\ClinicalData\[성인] ENT_ORTHO\2026 1012 빅데이터학과 파일 전달\Result\PSG\mat';
addpath(fullfile(base, 'Function'), '-begin');

p = inputParser;
addParameter(p, 'FigurePath', defaultFigurePath, @(x) ischar(x) || isstring(x));
addParameter(p, 'MatPath', defaultMatPath, @(x) ischar(x) || isstring(x));
addParameter(p, 'PatientIDs', [], @(x) isnumeric(x) && all(isfinite(x(:)) & x(:)>=1 & fix(x(:))==x(:)));
addParameter(p, 'ChannelsPerTab', 7, @(x) isscalar(x) && x>=1 && fix(x)==x);
addParameter(p, 'MaxPlotPoints', 100000, @(x) isscalar(x) && x>=100);
addParameter(p, 'SkipInitialSeconds', 180, @(x) isscalar(x) && isfinite(x) && x>=0);
addParameter(p, 'YAxisRangeFraction', 0.50, @(x) isscalar(x) && isfinite(x) && x>0 && x<=1);
addParameter(p, 'Overwrite', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'Process', 'PSG', @(x) any(strcmpi(x,{'PSG','CBCT'})));
parse(p, varargin{:});
options = p.Results;
if nargin<1, dataRoot=[]; end
if strcmpi(options.Process,'CBCT')
    % CBCT 전용 저장 경로. 명시적으로 전달한 경로는 그대로 사용합니다.
    if ismember('MatPath',p.UsingDefaults)
        options.MatPath='E:\ClinicalData\[성인] ENT_ORTHO\2026 1012 빅데이터학과 파일 전달\Result\CBCT\mat';
    end
    if ismember('FigurePath',p.UsingDefaults)
        options.FigurePath='E:\ClinicalData\[성인] ENT_ORTHO\2026 1012 빅데이터학과 파일 전달\Result\CBCT\Figure';
    end
    summary=processCBCT(dataRoot,options);
    if nargout==0, assignin('base','CBCT_All_Summary',summary); end
    return;
end

%% 2. 폴더 선택: 환자 폴더들을 포함하는 상위 폴더
if nargin < 1 || isempty(dataRoot)
    dataRoot = uigetdir(defaultDataPath, '전체 PSG 데이터 폴더 선택');
end
summary = table;
if isequal(dataRoot, 0)
    fprintf('폴더 선택을 취소했습니다.\n');
    return;
end
assert(isfolder(dataRoot), 'PSG:BatchPath', '데이터 폴더가 없습니다: %s', dataRoot);

%% 3. _E / _N 환자 폴더 검색 및 환자 번호 순 정렬
folders = dir(dataRoot);
folders = folders([folders.isdir]);
patientIDs = [];
selectedFolders = [];
for k = 1:numel(folders)
    token = regexp(folders(k).name, '^P_(\d+)(?:_.*)?_([EN])$', 'tokens', 'once');
    if isempty(token)
        continue;
    end
    id = str2double(token{1});
    if ~isempty(options.PatientIDs) && ~ismember(id, options.PatientIDs)
        continue;
    end
    patientIDs(end+1,1) = id; 
    selectedFolders(end+1,1) = k; 
end
assert(~isempty(patientIDs), 'PSG:BatchEmpty', 'P_번호..._E 또는 _N 폴더가 없습니다.');
assert(numel(unique(patientIDs))==numel(patientIDs), 'PSG:DuplicateID', ...
    '같은 P_번호를 가진 폴더가 둘 이상입니다. 저장 이름이 겹치므로 먼저 확인하세요.');
if ~isempty(options.PatientIDs)
    missingIDs = setdiff(options.PatientIDs, patientIDs);
    assert(isempty(missingIDs), 'PSG:MissingID', '찾지 못한 환자 번호: %s', mat2str(missingIDs));
end
[patientIDs, order] = sort(patientIDs);
folders = folders(selectedFolders(order));

if ~isfolder(options.FigurePath), mkdir(options.FigurePath); end
if ~isfolder(options.MatPath), mkdir(options.MatPath); end
count = numel(folders);
summary = table("P_"+string(patientIDs), string({folders.name})', ...
    repmat("Pending",count,1), zeros(count,1), zeros(count,1), strings(count,1), ...
    'VariableNames', {'Patient','Folder','Status','Channels','ElapsedSeconds','Message'});

%% 4. 환자별 읽기 -> 그래프 -> MAT/FIG 저장
for k = 1:count
    timer = tic;
    patientName = char(summary.Patient(k));
    sourceFolder = fullfile(folders(k).folder, folders(k).name);
    figureFile = fullfile(options.FigurePath, [patientName '.fig']);
    matFile = fullfile(options.MatPath, [patientName '.mat']);
    fprintf('\n[%d/%d] %s\n', k, count, sourceFolder);

    if isfile(figureFile) && isfile(matFile) && ~options.Overwrite
        summary.Status(k) = "Skipped";
        summary.Message(k) = "FIG/MAT already exist";
        fprintf('기존 FIG/MAT가 모두 있어 건너뜁니다. 변경한 표시 설정으로 다시 저장하려면 Overwrite=true를 사용하세요.\n');
        continue;
    end
    try
        assert(options.Overwrite || (~isfile(figureFile) && ~isfile(matFile)), 'PSG:PartialOutput', ...
            'FIG 또는 MAT 중 하나만 있습니다. 기존 파일을 확인한 후 재실행하세요.');
        processPatient(sourceFolder, figureFile, matFile, options);
        info = whos('-file', matFile, 'Signals');
        assert(~isempty(info), 'PSG:SavedResult', 'MAT에 Signals가 없습니다.');
        summary.Status(k) = "Saved";
        saved = load(matFile, 'SaveInfo');
        summary.Channels(k) = saved.SaveInfo.ChannelCount;
        summary.Message(k) = string(matFile);
    catch ME
        summary.Status(k) = "Failed";
        summary.Message(k) = string(ME.message);
        warning('PSG:BatchFailed', '%s: %s', patientName, ME.message);
        if strcmp(ME.identifier,'PSG:SaveFailed')
            summary.ElapsedSeconds(k) = toc(timer);
            summary.Status(k+1:end) = "NotRun";
            break; 
        end
    end
    summary.ElapsedSeconds(k) = toc(timer);
end

disp(summary);
fprintf('완료 %d / 기존 결과 %d / 실패 %d\n', ...
    nnz(summary.Status=="Saved"), nnz(summary.Status=="Skipped"), nnz(summary.Status=="Failed"));
if nargout == 0
    assignin('base', 'PSG_All_Summary', summary);
end
end

function processPatient(sourceFolder, figureFile, matFile, options)
if endsWith(sourceFolder, '_E', 'IgnoreCase', true)
    expectedFormat = 'EBM';
else
    expectedFormat = 'NDF';
end
result = PSG_Read(sourceFolder, 'Save', false, 'Reload', true, 'ShowProgress', false);
assert(strcmp(result.format, expectedFormat), 'PSG:FolderFormat', ...
    '폴더 접미사와 실제 파일 형식이 다릅니다: %s', sourceFolder);

[fig, Events, SleepStages, PlotData, EventCounts] = PSG_Plot(result, ...
    'ChannelsPerTab', options.ChannelsPerTab, ...
    'MaxPlotPoints', options.MaxPlotPoints, 'SkipInitialSeconds',options.SkipInitialSeconds, 'YAxisRangeFraction',options.YAxisRangeFraction, 'Visible', 'off'); 
% 실행할 때 결과창 띄울려면 'Visible', 'off'  -> 'on'으로 변경하고, 아래
% figure(fig);  %주석 해지
drawnow; 
result.outputFile = char(matFile);
[Signals, SignalMap] = standardizeSignals(result);
ChannelTable = result.channelTable;
RecordingInfo = rmfield(result, {'channels','channelTable'});
for k = 1:height(SignalMap)
    PlotData.Channels(k).standardName = char(SignalMap.StandardName(k));
end
SaveInfo = struct('SchemaVersion', 'PSG-batch-1.2', 'SourceFolder', sourceFolder, ...
    'SavedAt', datetime('now'), 'FigureFile', figureFile, 'MatFile', matFile, ...
    'ChannelCount', numel(result.channels), 'SkipInitialSeconds',options.SkipInitialSeconds, ...
    'EventCountScope','Full exported scoring; no first-minute exclusion', ...
    'FullData', 'Signals.(standardName).time / data / unit / fs; original source names retained', ...
    'DisplayedData', 'PlotData.Channels(k).time / data; NaT/NaN preserve gaps', ...
    'CommonSignals', 'Signals.ECG / SpO2 / NasalPressure etc.; .time/.data/.unit/.available', ...
    'UnitPolicy', 'Original reader values and units retained; variable names only are harmonized');

figureTemp = [tempname(options.FigurePath) '.fig'];
matTemp = [tempname(options.MatPath) '.mat'];
fileCleanup = onCleanup(@() removeTemporary(figureTemp, matTemp)); 

try
    save(matTemp, 'Signals', 'SignalMap', 'ChannelTable', 'RecordingInfo', ...
        'Events', 'SleepStages', 'EventCounts', 'PlotData', 'SaveInfo', '-v7.3');
    set(fig, 'Visible', 'off', 'WindowStyle', 'normal'); % 결과창 띄우려면 'Visible', 'on'으로 변경
    % movegui(fig, 'onscreen');
    drawnow;
    savefig(fig, figureTemp);
    close(fig);  % 저장 후 메모리 해제 --> 추가
catch ME
    error('PSG:SaveFailed', '저장 실패. 디스크 공간/파일 크기 제한을 확인하세요: %s', ME.message);
end
assert(options.Overwrite || (~isfile(figureFile) && ~isfile(matFile)), 'PSG:OutputExists', '저장 중 같은 이름의 결과가 생성되었습니다.');
movefile(matTemp, matFile, 'f');
movefile(figureTemp, figureFile, 'f');
fprintf('저장: %s\n      %s\n', figureFile, matFile);
end

function removeTemporary(figureTemp, matTemp)
if isfile(figureTemp), delete(figureTemp); end
if isfile(matTemp), delete(matTemp); end
end
function [Signals, SignalMap] = standardizeSignals(result)
aliases = {
    'ECG',            {'EKG','ECG'},            {'EKG','ECG'};
    'SpO2',           {'SpO2'},                {'SpO2'};
    'SpO2BeatToBeat',  {'SpO2 BB'},             {'SpO2 B-B'};
    'NasalPressure',  {'Nasal Pressure'},      {'Nasal Pressure'};
    'Thermistor',     {'Thermistor'},          {'Thermistor'};
    
    'Thorax',         {'Thorax'},              {'Thorax RIP'};
    'Abdomen',        {'Abdomen'},             {'Abdomen RIP'};
    'Pleth',          {'Plethysmogram'},       {'Pleth'};
    'Pulse',          {'Pulse'},               {'Pulse'};
    'HeartRate',      {'Heart Rate_CU','Heart Rate_DR','Heart Rate'}, {'Heart Rate'};
    'RespRate',       {'RR_CU','RR_DR'},        {'Resp Rate'};
    'LeftLegEMG',     {'Leg-L'},               {'Left Leg'};
    'RightLegEMG',    {'Leg-R'},               {'Right Leg'};
    'EEG_C3',         {'C3'},                  {'C3'};
    'EEG_C4',         {'C4'},                  {'C4'};
    'EEG_F3',         {'F3'},                  {'F3'};
    'EEG_F4',         {'F4'},                  {'F4'};
    'EEG_O1',         {'O1'},                  {'O1'};
    'EEG_O2',         {'O2'},                  {'O2'};
    };
if strcmp(result.format,'EBM'), aliasColumn=2; else, aliasColumn=3; end
names = string({result.channels.name});
Signals = struct;
channelCount = numel(names);
standardNames = strings(channelCount,1);
used = false(channelCount,1);
emptySignal = struct('available',false,'time',NaT(0,1),'data',[], ...
    'unit','','fs',NaN,'sourceName','','sourceIndex',NaN,'system',result.format, ...
    'sourceFile','','segments',table,'metadata',struct);

for row = 1:size(aliases,1)
    key = aliases{row,1};
    match = find(ismember(lower(strtrim(names)), lower(string(aliases{row,aliasColumn}))));
    assert(numel(match)<=1,'PSG:AmbiguousSignal','공통 이름 %s에 대응하는 채널이 둘 이상입니다.',key);
    Signals.(key) = emptySignal; 
    if isempty(match), continue; end
    Signals.(key) = makeSignal(result,match,emptySignal);
    standardNames(match) = string(key);
    used(match) = true;
end

for index = find(~used)'
    key = matlab.lang.makeValidName([result.format '_' char(names(index))]);
    if isfield(Signals,key), key=sprintf('%s_Channel%d',key,index); end
    Signals.(key) = makeSignal(result,index,emptySignal);
    standardNames(index) = string(key);
end
SignalMap = table((1:channelCount)',names(:),standardNames,used, ...
    'VariableNames',{'SourceIndex','SourceName','StandardName','CommonSignal'});
end

function signal = makeSignal(result,index,signal)
ch = result.channels(index);
signal.available = true;
signal.time = ch.time;
signal.data = ch.data;
signal.unit = ch.unit;
signal.fs = ch.fs;
signal.sourceName = ch.name;
signal.sourceIndex = index;
signal.sourceFile = ch.sourceFile;
signal.segments = ch.segments;
signal.metadata = ch.metadata;

end
%% 
function summary=processCBCT(dataRoot,options)
defaultCBCTPath='E:\ClinicalData\[성인] ENT_ORTHO\데이터정리\CBCT';
if isempty(dataRoot)
    dataRoot=uigetdir(defaultCBCTPath,'전체 CBCT 데이터 폴더 선택');
end
summary=table;
if isequal(dataRoot,0),fprintf('CBCT 폴더 선택을 취소했습니다.\n');return;end
assert(isfolder(dataRoot),'CBCT:BatchPath','CBCT 폴더가 없습니다: %s',dataRoot);
folders=dir(dataRoot);folders=folders([folders.isdir]);ids=[];selected=[];
for k=1:numel(folders)
    token=regexp(folders(k).name,'^P_(\d+)_CBCT$','tokens','once');
    if isempty(token),continue;end
    id=str2double(token{1});
    if ~isempty(options.PatientIDs)&&~ismember(id,options.PatientIDs),continue;end
    ids(end+1,1)=id;selected(end+1,1)=k; %#ok<AGROW>
end
assert(~isempty(ids),'CBCT:BatchEmpty','선택된 P_번호_CBCT 폴더가 없습니다.');
assert(numel(unique(ids))==numel(ids),'CBCT:DuplicateID','환자 번호가 중복되었습니다.');
if ~isempty(options.PatientIDs)
    missing=setdiff(options.PatientIDs,ids);
    assert(isempty(missing),'CBCT:MissingID','CBCT 폴더가 없는 번호: %s',mat2str(missing));
end
[ids,order]=sort(ids);folders=folders(selected(order));
if ~isfolder(options.MatPath),mkdir(options.MatPath);end
if ~isfolder(options.FigurePath),mkdir(options.FigurePath);end
summary=table("P_"+string(ids),string({folders.name})',repmat("Pending",numel(ids),1), ...
    zeros(numel(ids),1),zeros(numel(ids),1),strings(numel(ids),1), ...
    'VariableNames',{'Patient','Folder','Status','Slices','ElapsedSeconds','Message'});
for k=1:numel(ids)
    timer=tic;
    output=fullfile(options.MatPath,sprintf('P_%d.mat',ids(k)));
    figureOutput=fullfile(options.FigurePath,sprintf('P_%d.fig',ids(k)));
    if isfile(output)&&isfile(figureOutput)&&~options.Overwrite
        summary.Status(k)="Skipped";summary.Message(k)="CBCT FIG/MAT already exist";continue;
    end
    try
        assert(options.Overwrite || (~isfile(output)&&~isfile(figureOutput)), ...
            'CBCT:PartialOutput','FIG 또는 MAT 중 하나만 있습니다. Overwrite=true로 다시 생성하세요.');
        source=fullfile(folders(k).folder,folders(k).name);
        fprintf('\n[CBCT %d/%d] %s\n',k,numel(ids),source);
        CBCTData=CBCT(source,'ShowFigure',false);
        fig=drawCBCTFigure(CBCTData,ids(k));
        SaveInfo=struct('SchemaVersion','CBCT-batch-1.1','SavedAt',datetime('now'), ...
            'SourceFolder',source,'MatFile',output,'FigureFile',figureOutput, ...
            'FigureDescription','Three central sections and rotatable display isosurface; complete volume stored in MAT','Reader',which('CBCT'));
        temporary=[tempname(options.MatPath) '.mat'];
        temporaryFigure=[tempname(options.FigurePath) '.fig'];
        cleanup=onCleanup(@()removeTemporary(temporaryFigure,temporary));
        save(temporary,'CBCTData','SaveInfo','-v7.3');
        savefig(fig,temporaryFigure);
        assert(options.Overwrite||(~isfile(output)&&~isfile(figureOutput)),'CBCT:OutputExists','같은 이름의 결과가 생성되었습니다.');
        movefile(temporary,output,'f');
        movefile(temporaryFigure,figureOutput,'f');clear cleanup;
        summary.Status(k)="Saved";summary.Slices(k)=size(CBCTData.Volume,3);
        summary.Message(k)=string(output);
    catch ME
        summary.Status(k)="Failed";summary.Message(k)=string(ME.message);
        warning('CBCT:BatchFailed','P_%d: %s',ids(k),ME.message);
    end
    clear CBCTData;
    summary.ElapsedSeconds(k)=toc(timer);
end
disp(summary);
fprintf('CBCT 완료 %d / 기존 결과 %d / 실패 %d\n', ...
    nnz(summary.Status=="Saved"),nnz(summary.Status=="Skipped"),nnz(summary.Status=="Failed"));
end

function fig=drawCBCTFigure(data,id)
delete(findall(groot,'Type','figure','Tag','CBCTPreview'));
v=data.RescaledVolume;dims=size(v);mid=ceil(dims/2);
fig=figure('Color','w','Name',sprintf('P_%d | CBCT',id),'NumberTitle','off', ...
    'Tag','CBCTPreview','Position',[60 40 1400 950],'Visible','on','WindowStyle','normal');
layout=tiledlayout(fig,2,2,'TileSpacing','loose','Padding','compact');
planes={v(:,:,mid(3)),squeeze(v(mid(1),:,:))',squeeze(v(:,mid(2),:))'};
labels={'Column (voxel)','Row (voxel)';'Column (voxel)','Slice (voxel)';'Row (voxel)','Slice (voxel)'};
titles={sprintf('Row-column | slice %d',mid(3)),sprintf('Column-slice | row %d',mid(1)),sprintf('Row-slice | column %d',mid(2))};
lo=Inf;hi=-Inf;
for k=1:3
    finite=planes{k}(isfinite(planes{k}));
    if ~isempty(finite),lo=min(lo,min(finite));hi=max(hi,max(finite));end
end
if ~isfinite(lo)||~isfinite(hi),lo=0;hi=1;end
if hi<=lo,lo=lo-0.5;hi=hi+0.5;end
for k=1:3
    ax=nexttile(layout);imagesc(ax,planes{k},[double(lo) double(hi)]);
    axis(ax,'image');colormap(ax,gray(256));
    % DICOM patient Z는 머리 방향으로 증가합니다. 
    if k>1
        dz=diff(data.SliceTable.PatientPositionMm(:,3));
        if all(dz>1e-6)
            set(ax,'YDir','normal'); % 높은 슬라이스 번호가 머리 쪽
        elseif all(dz < -1e-6)
            set(ax,'YDir','reverse'); % 낮은 슬라이스 번호가 머리 쪽
        else
            warning('CBCT:DisplayOrientation', ...
                '슬라이스 Z 좌표가 단조롭지 않습니다. 원래 배열 방향을 표시합니다.');
        end
    end
    xlabel(ax,labels{k,1});ylabel(ax,labels{k,2});title(ax,titles{k});
    bar=colorbar(ax);bar.Label.String=data.Unit;
end
ax3D=nexttile(layout);
drawCBCTSurface(ax3D,data,double(lo),double(hi));
% 축 자체에 회전 interaction을 저장하여 openfig 후에도 드래그 회전 가능.
ax3D.Interactions=rotateInteraction;
title(layout,sprintf('P_%d | CBCT | %d x %d x %d',id,dims),'Interpreter','none');
drawnow;
end

function drawCBCTSurface(ax,data,lo,hi)
maxGridSize=160;          % 회전 속도를 위한 3D 표시 해상도
surfaceFraction=0.35;     % 표시 임계값: 단면 명암 범위 내 비율 (HU 기준 아님)
v=data.RescaledVolume;sz=size(v);step=max(1,ceil(max(sz)/maxGridSize));
rows=unique([1:step:sz(1),sz(1)]);
cols=unique([1:step:sz(2),sz(2)]);
slices=unique([1:step:sz(3),sz(3)]);
small=v(rows,cols,slices);
threshold=lo+surfaceFraction*(hi-lo);
surface=isosurface(small,threshold);
if isempty(surface.vertices)
    text(ax,0.5,0.5,'No surface at selected display threshold','Units','normalized','HorizontalAlignment','center');
    title(ax,'3D preview');return;
end
% isosurface의 x/y/z 인덱스를 원래 열/행/슬라이스 인덱스로 변환.
vertices=surface.vertices;
c=interp1(1:numel(cols),double(cols),vertices(:,1));
r=interp1(1:numel(rows),double(rows),vertices(:,2));
k=interp1(1:numel(slices),double(slices),vertices(:,3));
positions=double(data.SliceTable.PatientPositionMm);
origin=interp1((1:sz(3))',positions,k);
spacings=double(data.Spatial.PixelSpacings);
if size(spacings,1)==1,spacings=repmat(spacings,sz(3),1);end
spacing=interp1((1:sz(3))',spacings,k);
orientation=double(data.SliceTable.PatientOrientation(1,:));
surface.vertices=origin+((c-1).*spacing(:,2))*orientation(1:3) ...
    +((r-1).*spacing(:,1))*orientation(4:6);
if size(surface.faces,1)>100000,surface=reducepatch(surface,100000);end
patch(ax,surface,'FaceColor',[0.88 0.83 0.72],'EdgeColor','none', ...
    'FaceLighting','gouraud','Tag','CBCT3DSurface');
axis(ax,'equal');axis(ax,'tight');grid(ax,'on');view(ax,135,20);
set(ax,'ZDir','normal');camup(ax,[0 0 1]);
light(ax,'Position',[1 -1 1],'Style','infinite');
light(ax,'Position',[-1 1 0.5],'Style','infinite');
xlabel(ax,'Patient X (mm)');ylabel(ax,'Patient Y (mm)');zlabel(ax,'Patient Z (mm)');
title(ax,{'3D surface | drag to rotate',sprintf('Display threshold: %.1f (%s)',threshold,data.Unit)},'Interpreter','none');
ax.Tag='CBCT3DAxes';
ax.UserData=struct('Threshold',threshold,'GridStep',step,'DisplayOnly',true);
end
