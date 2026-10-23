function ch=PSG_read_ndf(file)
% Nox NOX03 NDF 읽기. 다른 제조사의 .ndf에는 적용하지 않습니다.
% Layout reference: https://github.com/jussivirkkala/Noxturnal-NDF
% 원본 XML, 정밀 주파수(514), 구간별 시각(512)을 보존합니다.
file=char(file); fid=fopen(file,'rb','ieee-le');
assert(fid>=0,'NDF 파일 열기 실패'); cleanup=onCleanup(@()fclose(fid));
assert(isequal(fread(fid,4,'*uint8')',uint8([78 79 88 3])),'지원하지 않는 NDF 서명');
info=dir(file); chunks={}; times={}; segStart=NaT(0,1); segCount=[]; segFs=[];
stamp=NaT; fs=NaN; xml=''; unknown=[];
while ftell(fid)<info.bytes
    type=fread(fid,1,'uint16'); len=fread(fid,1,'uint32'); pos=ftell(fid);
    assert(~isempty(len) && pos+len<=info.bytes,'잘린 NDF 블록');
    switch type
        case 256 % UTF-16 XML channel header
            xml=native2unicode(fread(fid,len,'*uint8')','UTF-16LE'); xml(xml==0)=[];
            label=tag(xml,'Label'); unit=tag(xml,'Unit'); encoding=tag(xml,'Format');
            gain=str2double(tag(xml,'Scale')); offset=str2double(tag(xml,'Offset'));
            nominalFs=str2double(tag(xml,'SamplingRate')); fs=nominalFs;
            assert(isfinite(gain)&&isfinite(offset)&&fs>0,'잘못된 NDF 환산/주파수');
            assert(isempty(regexp(xml,'<Key>Data_Encryption</Key>.*?<Value>[1-9]\d*</Value>','once')), ...
                '암호화 NDF는 지원하지 않습니다. 제조사에서 EDF로 내보내세요.');
        case 512 % YYYYMMDDTHHMMSS.ffffff
            text=native2unicode(fread(fid,len,'*uint8')','UTF-16LE'); text(text==0)=[];
            stamp=datetime(text,'InputFormat','yyyyMMdd''T''HHmmss.SSSSSS');
        case 514 % precise sampling frequency for this segment
            assert(len==8,'잘못된 NDF 주파수 블록'); fs=fread(fid,1,'double');
        case 513 % digital signal block
            assert(~isempty(xml) && ~isnat(stamp) && isfinite(fs)&&fs>0,'NDF 메타데이터 누락');
            switch encoding
                case 'Int16', precision='int16'; width=2;
                case 'Int32', precision='int32'; width=4;
                case 'UInt32', precision='uint32'; width=4;
                case 'Double', precision='double'; width=8;
                case 'Byte', precision='uint8'; width=1;
                otherwise, error('지원하지 않는 NDF 샘플 형식: %s',encoding);
            end
            assert(mod(len,width)==0,'불완전한 NDF 샘플');
            raw=fread(fid,len/width,precision);
            if ~isempty(raw)
                chunks{end+1,1}=raw*gain+offset; %#ok<AGROW>
                times{end+1,1}=stamp+seconds((0:numel(raw)-1)'/fs); %#ok<AGROW>
                segStart(end+1,1)=stamp; segCount(end+1,1)=numel(raw); segFs(end+1,1)=fs; %#ok<AGROW>
                stamp=stamp+seconds(numel(raw)/fs);
            end
        otherwise
            unknown(end+1)=type; %#ok<AGROW>
    end
    assert(fseek(fid,pos+len,'bof')==0,'NDF 블록 이동 실패');
end
assert(~isempty(chunks),'NDF 데이터 블록 없음');
first=cumsum([1;segCount(1:end-1)]);
segments=table(first,segCount,segStart,segFs, ...
    'VariableNames',{'FirstSample','Samples','StartTime','Fs'});
metadata=struct('headerXML',xml,'nominalFs',nominalFs,'scale',gain,'offset',offset, ...
    'encoding',encoding,'unitStatus','from NDF XML','otherBlockTypes',unique(unknown), ...
    'timePolicy','Segment timestamps + precise rate block 514; no invented gap samples', ...
    'readerReference','https://github.com/jussivirkkala/Noxturnal-NDF');
fs=segFs(1); if any(abs(segFs-fs)>1e-9), fs=NaN; end
ch=PSG_channel(label,vertcat(chunks{:}),vertcat(times{:}),fs,unit,file,segments,metadata);
end

function value=tag(xml,name)
token=regexp(xml,['<' name '>(.*?)</' name '>'],'tokens','once');
if isempty(token), value=''; else, value=token{1}; end
value=strrep(strrep(strrep(value,'&amp;','&'),'&lt;','<'),'&gt;','>');
end
