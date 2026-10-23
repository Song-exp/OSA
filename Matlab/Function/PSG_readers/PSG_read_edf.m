function [channels,annotations]=PSG_read_edf(file)
% MATLAB 기본 EDF/EDF+ 읽기. record start를 사용하여 EDF+D 단절을 보존.
% https://www.mathworks.com/help/signal/ref/edfread.html
file=char(file); info=edfinfo(file);
[records,annotations]=edfread(file,'TimeOutputType','datetime');
labels=info.SignalLabels;
assert(width(records)==numel(labels),'EDF 헤더/데이터 채널 수 불일치');
recordTime=records.Properties.RowTimes;
for k=1:numel(labels)
    blocks=records{:,k}; counts=cellfun(@numel,blocks);
    fs=info.NumSamples(k)/seconds(info.DataRecordDuration);
    times=cell(size(blocks));
    for j=1:numel(blocks)
        blocks{j}=blocks{j}(:);
        times{j}=recordTime(j)+seconds((0:counts(j)-1)'/fs);
    end
    segments=table(cumsum([1;counts(1:end-1)]),counts,recordTime,repmat(fs,height(records),1), ...
        'VariableNames',{'FirstSample','Samples','StartTime','Fs'});
    metadata=struct('unitStatus','from EDF header','transducer',info.TransducerTypes(k), ...
        'physicalMin',info.PhysicalMin(k),'physicalMax',info.PhysicalMax(k), ...
        'digitalMin',info.DigitalMin(k),'digitalMax',info.DigitalMax(k),'prefilter',info.Prefilter(k), ...
        'timePolicy','EDF record timestamps; original sample rates');
    ch=PSG_channel(labels(k),vertcat(blocks{:}),vertcat(times{:}),fs, ...
        info.PhysicalDimensions(k),file,segments,metadata);
    if k==1, channels=ch; else, channels(k)=ch; end %#ok<AGROW>
end
end
