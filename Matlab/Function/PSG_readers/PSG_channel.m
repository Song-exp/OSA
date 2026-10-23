function ch=PSG_channel(name,data,time,fs,unit,file,segments,metadata)
% 모든 형식에 공통인 채널 구조. 원래 이름을 필드명으로 바꾸지 않음.
ch=struct('name',char(name),'data',data(:),'time',time(:),'fs',fs, ...
    'unit',char(unit),'sourceFile',char(file),'segments',segments,'metadata',metadata);
ch.time.Format='yyyy-MM-dd HH:mm:ss.SSSSSS';
end
