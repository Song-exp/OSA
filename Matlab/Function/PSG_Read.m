function result = PSG_Read(inputPath, varargin)

if nargin>=1 && isequal(inputPath,0)
    result=[];
    return;
end
if nargin>=1 && ~isempty(inputPath)
    assert((ischar(inputPath)&&isrow(inputPath)) || ...
        (isstring(inputPath)&&isscalar(inputPath)&&~ismissing(inputPath)), ...
        'PSG_Read:InputType','inputPath must be a folder/file path or cancellation value 0.');
end
base=fileparts(fileparts(mfilename('fullpath')));
if nargin<1 || isempty(inputPath)
    inputPath=uigetdir(fullfile(base,'Data'),'PSG 기록 폴더 선택');
    if isequal(inputPath,0), result=[]; return; end
end
p=inputParser;
addParameter(p,'Save',false,@(x)isempty(x)||(islogical(x)&&isscalar(x)));
addParameter(p,'Reload',false,@(x)islogical(x)&&isscalar(x));
addParameter(p,'OutputFile','',@(x)ischar(x)||isstring(x));
addParameter(p,'ShowProgress',true,@islogical);
parse(p,varargin{:}); options=p.Results;
inputPath=char(inputPath);
oldPath=path; pathCleanup=onCleanup(@()path(oldPath)); 
addpath(fullfile(base,'Function'),'-begin');
addpath(fullfile(base,'Function','PSG_readers'),'-begin');

if isfolder(inputPath)
    files=dir(inputPath); files=files(~[files.isdir]);
elseif isfile(inputPath)
    files=dir(inputPath);
else
    error('PSG_Read:Path','폴더/파일을 찾을 수 없습니다: %s',inputPath);
end
ext=strings(numel(files),1);
for k=1:numel(files), ext(k)=lower(string(getExtension(files(k).name))); end
supported=ismember(ext,[".ebm",".ndf",".edf"]);
files=files(supported); ext=ext(supported);
formats=unique(ext);
if isempty(formats), error('PSG_Read:Empty','이 폴더에 EBM/NDF/EDF가 없습니다. 기록 폴더를 선택하세요.'); end
if numel(formats)>1, error('PSG_Read:Mixed','서로 다른 형식이 섞여 있습니다. 형식별 폴더 또는 EDF 파일을 선택하세요.'); end
format=erase(upper(formats),'.');
if format=="EDF" && numel(files)>1
    error('PSG_Read:MultipleEDF','EDF 파일이 여러 개입니다. 읽을 EDF 파일 경로를 지정하세요.');
end
[~,order]=sort(lower(string({files.name}))); files=files(order);
sourceFolder=files(1).folder;
[~,recordName,recordExt]=fileparts(sourceFolder); recordName=[recordName recordExt];
output=char(options.OutputFile);
if isempty(output), output=fullfile(base,'PSG_results',[recordName '_PSG.mat']); end
sourceFiles=string(fullfile({files.folder},{files.name}))';
if ~options.Reload && isfile(output)
    savedFile=dir(output);
    if savedFile.datenum >= max([files.datenum])
        loadTimer=tic;
        S=load(output,'result');
        if isfield(S,'result') && isstruct(S.result) && isscalar(S.result) && ...
                all(isfield(S.result,{'schemaVersion','sourceFiles','channels','channelTable'})) && ...
                strcmp(S.result.schemaVersion,'PSG-1.0') && ...
                isequal(S.result.sourceFiles,sourceFiles) && ...
                (format~="EBM" || (isfield(S.result,'ebmScalingVersion') && ...
                strcmp(S.result.ebmScalingVersion,'SpO2-percent-1')))
            result=S.result;
            fprintf('기존 MAT 사용: %.2f초 (원본 읽기 / 재저장 없음)\n%s\n',toc(loadTimer),output);
            return;
        end
        warning('PSG_Read:CacheMismatch','MAT의 기록 정보가 달라 원본을 읽습니다.');
    else
        fprintf('원본 파일이 MAT보다 최근에 변경되어 다시 읽습니다.\n');
    end
end

result=struct('schemaVersion','PSG-1.0','format',char(format), ...
    'sourceFolder',sourceFolder,'sourceFiles',string(fullfile({files.folder},{files.name}))', ...
    'createdAt',datetime('now'),'channels',[],'channelTable',table, ...
    'annotations',[],'outputFile','');

% Reject pre-fix EBM caches so tiny SpO2 values cannot silently reappear.
if format=="EBM", result.ebmScalingVersion='SpO2-percent-1'; end

result.excludedChannels=table(strings(0,1),strings(0,1),'VariableNames',{'SourceFile','Reason'});
bar=[];
if options.ShowProgress
    bar=waitbar(0,'PSG 데이터 읽기','Name','PSG_Read', ...
        'CreateCancelBtn','setappdata(gcbf,''cancelled'',true)');
    setappdata(bar,'cancelled',false);
end
barCleanup=onCleanup(@()closeProgress(bar));
if format=="EDF"
    progress(bar,0,'EDF 전체 채널 읽기');
    [result.channels,result.annotations]=PSG_read_edf(result.sourceFiles(1));
else
    for k=1:numel(files)
        progress(bar,(k-1)/numel(files),sprintf('%d/%d  %s',k,numel(files),files(k).name));
        file=fullfile(files(k).folder,files(k).name);
        if format=="EBM", ch=PSG_read_ebm(file); else, ch=PSG_read_ndf(file); end
        % 확인된 P43 예외: 다른 검사 날짜(2019-03-21)의 채널을 제외합니다.
        % 원본이 올바른 날짜로 교체되면 이 조건에 해당하지 않습니다.
        [~,patientFolder]=fileparts(sourceFolder);
        if format=="EBM" && strcmpi(patientFolder,'P_43_PSG_E') && ...
                dateshift(ch.time(1),'start','day')==datetime(2019,3,21)
            reason="Different recording date (2019-03-21); P43 recording is 2018-10-30. Excluded by user request.";
            result.excludedChannels(end+1,:)={string(file),reason};
            warning('PSG:ExcludedChannel','%s: %s',file,reason);
            continue;
        end
        if isempty(result.channels), result.channels=ch; else, result.channels(end+1)=ch; end
    end
end
progress(bar,1,'채널 목록 정리');

n=numel(result.channels);
names=strings(n,1); units=names; fs=zeros(n,1); counts=fs; starts=NaT(n,1); ends=starts;
for k=1:n
    ch=result.channels(k);
    assert(numel(ch.data)==numel(ch.time),'신호/시간 길이가 다릅니다: %s',ch.name);
    assert(~isempty(ch.data) && ~any(isnat(ch.time)),'빈 채널 또는 잘못된 시각: %s',ch.name);
    names(k)=ch.name; units(k)=ch.unit; fs(k)=ch.fs; counts(k)=numel(ch.data);
    starts(k)=ch.time(1); ends(k)=ch.time(end);
end
result.channelTable=table((1:n)',names,fs,units,counts,starts,ends, ...
    'VariableNames',{'Index','Name','Fs','Unit','Samples','StartTime','EndTime'});

closeProgress(bar);
result=PSG_Save(result,output,options.Save);
fprintf('%s: %d개 채널, 총 %.0f개 샘플\n',format,n,sum(counts));
end

function ext=getExtension(name)
[~,~,ext]=fileparts(name);
end
function progress(bar,value,message)
if isempty(bar), return; end
if ~isgraphics(bar) || getappdata(bar,'cancelled'), error('PSG_Read:Cancelled','읽기를 취소했습니다.'); end
waitbar(value,bar,message); drawnow;
end
function closeProgress(bar)
if ~isempty(bar) && isgraphics(bar), delete(bar); end
end
