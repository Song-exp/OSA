function result=PSG_Save(result,output,doSave)
% PSG_Save  저장 여부를 선택하고 result를 비압축 MAT로 저장.
% result = PSG_Save(result);  % 기존 result도 원본 재읽기 없이 저장 가능
% result = PSG_Save(result,output,true); % 질문 없이 저장 (배치 실행용)
if nargin<2 || isempty(output)
    output=result.outputFile;
    if isempty(output)
        [~,name,ext]=fileparts(result.sourceFolder);
        output=fullfile(fileparts(fileparts(mfilename('fullpath'))),'PSG_results',[name ext '_PSG.mat']);
    end
end
output=char(output);
if nargin<3, doSave=[]; end
if isempty(doSave)
    message=sprintf('데이터를 비압축 MAT 파일로 저장할까요?\n%s',output);
    if isfile(output), message=sprintf('%s\n기존 파일을 덮어씁니다.',message); end
    answer=questdlg(message,'PSG 저장','저장','저장 안 함','저장 안 함');
    doSave=strcmp(answer,'저장');
end
if ~doSave, return; end
parent=fileparts(output);
if ~isempty(parent) && ~isfolder(parent), mkdir(parent); end
result.outputFile=output;
save(output,'result','-v7.3','-nocompression');
writetable(result.channelTable,[output(1:end-4) '_channels.csv']);
fprintf('비압축 MAT 저장: %s\n',output);
end