function ch=PSG_read_ebm(file)
% EbmRead를 호출하고 반환된 단위 및 배율을 유지합니다.
% EbmRead의 zero padding은 실제 센서 샘플과 분리하고, 구간별 시간을 보존.
file=char(file); [allData,header]=EbmRead(file);
fid=fopen(file,'rb','ieee-le'); assert(fid>=0,'EBM 열기 실패');
cleanup=onCleanup(@()fclose(fid));
signature=[];
for k=1:80
    b=fread(fid,1,'uint8'); assert(~isempty(b),'EBM 서명 누락');
    if b==26, break; end
    signature(end+1)=b; %#ok<AGROW>
end
assert(contains(char(signature),'Embla'),'EBM 서명 오류');
assert(fread(fid,1,'uint8')==0,'현재 검증된 EBM은 little-endian wide-ID 형식입니다.');
offset=ftell(fid); marker=fread(fid,5,'uint8');
assert(all(marker==255),'검증되지 않은 EBM short-ID 형식');
fseek(fid,offset+31,'bof');
starts=NaT(0,1); readerStarts=[]; counts=[]; stamp=NaT; fs=header.samplingrate;
sizeInfo=dir(file); freshTime=false; trailingPaddingBytes=0;
while ftell(fid)<sizeInfo.bytes
    id=fread(fid,1,'uint32'); len=fread(fid,1,'uint32'); pos=ftell(fid);
    % 일부 파일은 마지막 센서 블록 뒤를 0xFF로 채웁니다.
    % 남은 영역이 전부 FF인 경우만 패딩으로 인정합니다.
    if isequal(id,double(intmax('uint32'))) && isequal(len,double(intmax('uint32')))
        tail=fread(fid,Inf,'*uint8');
        assert(all(tail==255),'PSG:EBMTail','손상되었거나 지원하지 않는 EBM 끝 영역: %s',file);
        trailingPaddingBytes=8+numel(tail);
        break;
    end
    assert(~isempty(len)&&pos+len<=sizeInfo.bytes,'PSG:EBMBlock','잘린 EBM 블록: %s',file);
    if id==132
        y=fread(fid,1,'int16'); a=fread(fid,6,'uint8');
        stamp=datetime(y,a(1),a(2),a(3),a(4),a(5)+a(6)/100); freshTime=true;
        readerStamp=datenum(y,a(1),a(2),a(3),a(4),a(5)+0.01*a(6));
    elseif id==32
        assert(freshTime && mod(len,2)==0,'EBM 시각/데이터 블록 관계를 확인하세요.');
        starts(end+1,1)=stamp; readerStarts(end+1,1)=readerStamp; counts(end+1,1)=len/2; freshTime=false; %#ok<AGROW>
    elseif id==134
        assert(fread(fid,1,'int32')/1000==fs,'변동 주파수 EBM은 지원하지 않습니다.');
    end
    fseek(fid,pos+len,'bof');
end
assert(~isempty(counts),'EBM 센서 데이터 없음');
% EbmRead의 연결 인덱스를 재현하여 패딩을 제외한 실제 블록을 가져옵니다.
blocks=cell(numel(counts),1); times=blocks; cursor=0; previousLength=0;
for k=1:numel(counts)
    fill=0;
    if k>1
        % EbmRead와 같은 datenum 생성 및 연산 순서를 유지해야 합니다.
        % 수학적으로 같은 식도 부동소수점 반올림에서 1샘플 차이가 납니다.
        previousEnd=readerStarts(k-1)+previousLength/(fs*60*60*24);
        gap=readerStarts(k)-previousEnd;
        fill=round(max(0,gap*60*60*24*fs));
    end
    idx=cursor+fill+(1:counts(k));
    assert(idx(end)<=numel(allData),'PSG:EBMMapping','EBM 반환 길이 불일치: %s (block %d)',file,k);
    blocks{k}=allData(idx); times{k}=starts(k)+seconds((0:counts(k)-1)'/fs);
    cursor=idx(end); previousLength=fill+counts(k);
end
assert(cursor==numel(allData),'PSG:EBMMapping', ...
    'EBM 반환 샘플의 일부가 매핑되지 않았습니다: %s (mapped %d / returned %d)',file,cursor,numel(allData));
segments=table(cumsum([1;counts(1:end-1)]),counts,starts,repmat(fs,numel(counts),1), ...
    'VariableNames',{'FirstSample','Samples','StartTime','Fs'});
metadata=struct('header',header,'unitStatus',header.outputunit,'scalingVersion',header.scalingVersion, ...
    'removedPaddingSamples',numel(allData)-sum(counts),'trailingPaddingBytes',trailingPaddingBytes,'reader',which('EbmRead'), ...
    'timePolicy','Original record timestamps; no zero-filled gap samples');
% 리더에서 검증한 단위만 표시합니다. 기타 채널은 기존 미확인 상태를 유지합니다.
ch=PSG_channel(header.channelname,vertcat(blocks{:}),vertcat(times{:}),fs, ...
    header.outputunit,file,segments,metadata);
end
