function [Events,Stages,EventCounts] = PSG_Events(result)
% Read recording-local scoring without modifying or saving any file.
% NOX: active scoring revision, .NET ticks (100 ns since 0001-01-01).
% EMB: RemLogic exports. EDF: matching-stem TXT events / CSV stages.
Events=table(NaT(0,1),NaT(0,1),zeros(0,1),strings(0,1),strings(0,1), ...
    'VariableNames',{'StartTime','EndTime','DurationSeconds','EventType','Group'});
Stages=table(NaT(0,1),NaT(0,1),zeros(0,1),'VariableNames',{'StartTime','EndTime','Level'});
start=min(result.channelTable.StartTime); stop=max(result.channelTable.EndTime);
folder=result.sourceFolder;
countNames=["APNEA-CENTRAL";"APNEA-MIXED";"APNEA-OBSTRUCTIVE"; ...
    "AROUSAL-RERA";"AROUSAL-RESP";"HYPOPNEA"];
EventCounts=table(countNames,nan(6,1),'VariableNames',{'EventType','Count'});
countAvailable=false;
scoringSource='';
switch upper(result.format)
    case 'NDF'
        file=fullfile(folder,'Data.ndb');
        if ~isfile(file), warning('PSG:Events','No Data.ndb: events unavailable.'); return; end
        db=sqlite(file,'readonly'); cleanup=onCleanup(@()close(db)); %#ok<NASGU>
        [q,scoringSource]=readNoxMarkers(db);
        countAvailable=true;
        for k=1:height(q)
            name=upper(string(q.kind(k)));
            a=ticksTime(q.a(k)); b=ticksTime(q.b(k));
            assert(b>=a,'PSG:EventTime','Negative NOX event duration.');
            [Events,Stages]=append(Events,Stages,a,b,name);
        end
    case 'EBM'
        files={fullfile(folder,'EN-Apnea Events.txt'),fullfile(folder,'EN-Stage Events.txt')};
        for f=1:numel(files)
            if ~isfile(files{f}), warning('PSG:Events','Missing scoring: %s',files{f}); continue; end
            if f==1, countAvailable=true; scoringSource=files{f}; end
            txt=readText(files{f},'windows-949');
            date=regexp(txt,'Recording Date:\s*(\d{4}-\d{2}-\d{2})','tokens','once');
            assert(~isempty(date),'PSG:EventDate','Missing recording date: %s',files{f});
            base=datetime(date{1},'InputFormat','yyyy-MM-dd');
            lines=splitlines(string(txt));
            % RemLogic은 Position 열이 있는 5열 / 없는 4열을 모두 출력합니다.
            % 열 번호를 고정하지 않고 실제 헤더에서 각 열을 찾습니다.
            headerRow=[];
            for k=1:numel(lines)
                columns=strtrim(split(lines(k),sprintf('\t')));
                if any(strcmpi(columns,'Event')) && any(startsWith(columns,'Time','IgnoreCase',true))
                    headerRow=k;
                    timeColumn=find(startsWith(columns,'Time','IgnoreCase',true));
                    eventColumn=find(strcmpi(columns,'Event'));
                    durationColumn=find(startsWith(columns,'Duration','IgnoreCase',true));
                    break;
                end
            end
            assert(~isempty(headerRow),'PSG:EventHeader','Scoring column header not found: %s',files{f});
            assert(isscalar(timeColumn)&&isscalar(eventColumn)&&isscalar(durationColumn), ...
                'PSG:EventHeader','Missing or ambiguous Time/Event/Duration columns: %s',files{f});
            requiredColumns=max([timeColumn,eventColumn,durationColumn]);
            previousScoringTime=NaT;
            for k=headerRow+1:numel(lines)
                if strlength(strtrim(lines(k)))==0, continue; end
                cols=split(lines(k),sprintf('\t'));
                assert(numel(cols)>=requiredColumns,'PSG:EventRow','Incomplete scoring row %d: %s',k,files{f});
                name=upper(strtrim(cols(eventColumn)));
                a=clockTime(cols(timeColumn),base,start,stop,previousScoringTime);
                previousScoringTime=a;
                d=str2double(cols(durationColumn)); assert(isfinite(d)&&d>=0,'PSG:Duration','Invalid event duration at row %d: %s',k,files{f});
                [Events,Stages]=append(Events,Stages,a,a+seconds(d),name);
            end
        end
    case 'EDF'
        [~,stem]=fileparts(result.sourceFiles(1));
        file=fullfile(folder,stem+'.txt');
        base=dateshift(start,'start','day');
        if isfile(file)
            countAvailable=true; scoringSource=file;
            lines=splitlines(string(readText(file,'UTF-8')));
            for k=1:numel(lines)
                cols=split(lines(k),sprintf('\t'));
                if numel(cols)<6, continue; end
                if any(strcmpi(strtrim(cols(1)),{'Time','Time [hh:mm:ss]','Start','Start Time','Onset'})), continue; end
                name=upper(strtrim(cols(6)));
                if strlength(name)==0, continue; end
                a=clockTime(cols(1),base,start,stop);
                parts=str2double(split(cols(5),':'));
                assert(all(isfinite(parts)) && numel(parts)<=3,'PSG:Duration','Invalid EDF event duration.');
                d=sum(parts(:)'.*60.^((numel(parts)-1):-1:0));
                [Events,Stages]=append(Events,Stages,a,a+seconds(d),name);
            end
        else
            warning('PSG:Events','Matching EDF event TXT not found; no event shading.');
        end
        file=fullfile(folder,stem+'.csv');
        if isfile(file)
            lines=splitlines(string(readText(file,'UTF-8'))); times=NaT(0,1); levels=[];
            for k=2:numel(lines)
                cols=split(lines(k),','); if numel(cols)<3, continue; end
                level=stageLevel(erase(strtrim(cols(3)),'"'));
                times(end+1,1)=clockTime(cols(2),base,start,stop); %#ok<AGROW>
                levels(end+1,1)=level; %#ok<AGROW>
            end
            if numel(times)>=2
                interval=median(seconds(diff(times)));
                assert(interval>0,'PSG:StageTime','Invalid stage order.');
                ends=[times(2:end);min(times(end)+seconds(interval),stop)];
                Stages=table(times,ends,levels,'VariableNames',Stages.Properties.VariableNames);
            end
        end
end
% Full exported scoring counts, before display/recording clipping.
if countAvailable
    for k=1:height(EventCounts)
        EventCounts.Count(k)=nnz(Events.EventType==EventCounts.EventType(k));
    end
end
EventCounts.Properties.UserData=struct('Scope','Full exported scoring; no first-minute exclusion', ...
    'Source',scoringSource);
Events=sortrows(Events,'StartTime'); Stages=sortrows(Stages,'StartTime');
% Scoring may extend slightly beyond raw data. Keep overlapping intervals only.
outside=Events.EndTime<start | Events.StartTime>stop;
if any(outside), warning('PSG:OutsideEvents','Excluded %d events outside signal coverage.',nnz(outside)); end
Events(outside,:)=[];
Stages(Stages.EndTime<start | Stages.StartTime>stop,:)=[];
fprintf('Scoring: %d events, %d sleep-stage intervals.\n',height(Events),height(Stages));
end

function [E,S]=append(E,S,a,b,name)
if startsWith(name,'SLEEP-')
    S(end+1,:)={a,b,stageLevel(name)};
    return;
end
% Harmonize only equivalent source labels; keep other types distinct.
if name=="RERA", name="AROUSAL-RERA"; end
if name=="AROUSAL-RESPIRATORY", name="AROUSAL-RESP"; end
if contains(name,'HYPOPNEA'), group="HYPOPNEA";
elseif contains(name,'APNEA'), group="APNEA";
elseif startsWith(name,'AROUSAL'), group="AROUSAL";
else, group="OTHER";
end
E(end+1,:)={a,b,seconds(b-a),name,group};
end

function [q,source]=readNoxMarkers(db)
% NOX exports use either revisions or a single temporary scoring group.
tables=fetch(db,'SELECT name FROM sqlite_master WHERE type=''table''');
names=string(tables.name);
if any(names=="scoring_revision") && any(names=="scoring_marker")
    revisions=fetch(db,'SELECT id FROM scoring_revision WHERE is_deleted=0');
    assert(height(revisions)>0,'PSG:Revision','No active NOX scoring revision.');
    revision=[];
    if height(revisions)==1
        revision=revisions.id(1);
    elseif all(ismember(["temporary_scoring_group","temporary_scoring_group_to_key"],names))
        groups=fetch(db,'SELECT id FROM temporary_scoring_group WHERE is_deleted=0');
        if height(groups)==1
            current=fetch(db,sprintf('SELECT key_id FROM temporary_scoring_group_to_key WHERE group_id=%d',groups.id(1)));
            for k=1:height(revisions)
                keys=fetch(db,sprintf('SELECT key_id FROM scoring_revision_to_key WHERE revision_id=%d',revisions.id(k)));
                if ~isempty(current) && isequal(sort(unique(current.key_id)),sort(unique(keys.key_id)))
                    revision(end+1)=revisions.id(k); %#ok<AGROW>
                end
            end
        end
    end
    assert(isscalar(revision),'PSG:Revision', ...
        'Cannot uniquely match NOX revisions to the current scoring group. Explicit review required.');
    markerTable='scoring_marker';
    keysSQL=sprintf('SELECT key_id FROM scoring_revision_to_key WHERE revision_id=%d',revision);
    source=sprintf('Data.ndb: scoring_revision %d',revision);
elseif all(ismember(["temporary_scoring_marker","temporary_scoring_group","temporary_scoring_group_to_key"],names))
    groups=fetch(db,'SELECT id FROM temporary_scoring_group WHERE is_deleted=0');
    assert(height(groups)==1,'PSG:Revision','Expected one active NOX temporary scoring group.');
    markerTable='temporary_scoring_marker';
    keysSQL=sprintf('SELECT key_id FROM temporary_scoring_group_to_key WHERE group_id=%d',groups.id(1));
    source=sprintf('Data.ndb: temporary_scoring_group %d',groups.id(1));
else
    error('PSG:NoxSchema','No supported NOX scoring tables found.');
end
sql=sprintf(['SELECT CAST(m.starts_at AS TEXT) AS a, CAST(m.ends_at AS TEXT) AS b, m.type AS kind ' ...
    'FROM %s m WHERE m.is_deleted=0 AND m.key_id IN (%s) ORDER BY m.starts_at'],markerTable,keysSQL);
q=fetch(db,sql);
fprintf('Scoring source: %s\n',source);
end
function level=stageLevel(name)
name=upper(string(name)); level=NaN;
groups={ ["W","SLEEP-S0","SLEEP-WAKE"], ["REM","R","SLEEP-REM"], ...
    ["N1","SLEEP-S1","SLEEP-N1"],["N2","SLEEP-S2","SLEEP-N2"], ...
    ["N3","SLEEP-S3","SLEEP-N3"],["N4","SLEEP-S4","SLEEP-N4"]};
for k=1:numel(groups), if any(name==groups{k}), level=k-1; return; end; end
end
function t=ticksTime(value)
text=char(string(value)); ticks=int64(0);
assert(all(text>='0' & text<='9'),'PSG:Ticks','Invalid ticks.');
for c=text, ticks=ticks*int64(10)+int64(double(c)-48); end
perDay=int64(864000000000);
d=idivide(ticks,perDay,'floor'); remainder=rem(ticks,perDay);
t=datetime(1,1,1)+caldays(double(d))+seconds(double(remainder)/1e7);
end
function t=clockTime(text,base,start,stop,previousTime)
if nargin<5, previousTime=NaT; end
token=regexp(char(text),'(?:(오전|오후|AM|PM)\s*)?(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)','tokens','once');
assert(~isempty(token),'PSG:Clock','Cannot parse time: %s',text);
% Read numeric time independently because optional unmatched groups vary.
nums=regexp(char(text),'(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)','tokens','once');
h=str2double(nums{1}); m=str2double(nums{2}); s=str2double(nums{3});
if contains(string(text),["오후","PM"]), h=mod(h,12)+12;
elseif contains(string(text),["오전","AM"]), h=mod(h,12); end
candidates=base+days(-1:3)+hours(h)+minutes(m)+seconds(s);
inside=candidates>=start-minutes(5) & candidates<=stop+minutes(5);
% 기록 종료 후에도 stage 행이 이어질 수 있습니다. 앞 행에서 확인한 날짜와
% 연속되는 시각이 하나일 때만 날짜를 복원하며, 범위 밖 행은 호출부에서 제외합니다.
if nnz(inside)==0 && ~isnat(previousTime)
    continued=candidates>=previousTime & candidates<previousTime+hours(12);
    if nnz(continued)==1
        t=candidates(continued);
        return;
    end
end
assert(nnz(inside)==1,'PSG:Clock','Scoring time outside or ambiguous: %s',text);
t=candidates(inside);
end
function text=readText(file,encoding)
fid=fopen(file,'rb'); assert(fid>=0,'Cannot open scoring file.'); c=onCleanup(@()fclose(fid)); %#ok<NASGU>
b=fread(fid,Inf,'*uint8')';
if numel(b)>=3 && isequal(b(1:3),uint8([239 187 191])), b=b(4:end); encoding='UTF-8'; end
if numel(b)>=2 && isequal(b(1:2),uint8([255 254])), b=b(3:end); encoding='UTF-16LE';
elseif numel(b)>=2 && isequal(b(1:2),uint8([254 255])), b=b(3:end); encoding='UTF-16BE'; end
text=native2unicode(b,encoding);
end
