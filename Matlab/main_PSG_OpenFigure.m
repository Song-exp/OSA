function fig = main_PSG_OpenFigure(figureFile)
% 저장 시 숨겨둔 PSG FIG도 화면에 표시합니다. 원본 파일은 변경하지 않습니다.
% 사용: main_PSG_OpenFigure
% 또는: fig = main_PSG_OpenFigure('전체 경로\P_1.fig');
fig = [];
if nargin < 1 || isempty(figureFile)
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    figureFolder = fullfile(projectRoot,'Result','PSG','Figure');
    if ~isfolder(figureFolder), figureFolder = projectRoot; end
    [name, folder] = uigetfile('*.fig','PSG 결과 FIG 선택',fullfile(figureFolder,'*.fig'));
    if isequal(name,0), return; end
    figureFile = fullfile(folder,name);
end
assert(isfile(figureFile),'PSG:FigureNotFound','FIG 파일이 없습니다: %s',figureFile);
% 저장된 Visible='off' 속성을 무시하고 새 창으로 엽니다.
fig = openfig(figureFile,'new','visible');
for k = 1:numel(fig)
    set(fig(k),'WindowStyle','normal','Visible','on');
    movegui(fig(k),'onscreen');
end
drawnow;
end