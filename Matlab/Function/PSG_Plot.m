function [figures, Events, Stages, PlotData, EventCounts] = PSG_Plot(result, varargin)
%% 1. Setting
settings.channelsPerTab = 8;
settings.maxPlotPoints = 100000;
settings.figurePosition = [40 40 1600 950];
settings.signalColor = [0.08 0.25 0.48];
settings.signalLineWidth = 0.6;
settings.stageColor = [0.20 0.15 0.48];
settings.stageLineWidth = 1.3;
settings.eventNames = ["APNEA-CENTRAL", "APNEA-MIXED", "APNEA-OBSTRUCTIVE", ...
    "AROUSAL-RERA", "AROUSAL-RESP", "HYPOPNEA"];
settings.eventColors = [0.55 0.04 0.07; 0.78 0.16 0.19; 0.96 0.38 0.40; ...
    0.12 0.30 0.82; 0.24 0.58 0.95; 0.55 0.80 0.20];
settings.eventAlpha = 0.28; % 0 = transparent, 1 = opaque
settings.timeFormat = 'MM-dd HH:mm:ss';

p = inputParser;
addParameter(p, 'Window', [], @(x) isempty(x) || (isdatetime(x) && numel(x) == 2));
addParameter(p, 'ChannelsPerTab', [], @(x) isempty(x) || (isscalar(x) && x >= 1 && fix(x) == x));
addParameter(p, 'ChannelsPerFigure', settings.channelsPerTab, @(x) isscalar(x) && x >= 1 && fix(x) == x);
addParameter(p, 'MaxPlotPoints', settings.maxPlotPoints, @(x) isscalar(x) && x >= 100);
addParameter(p, 'SkipInitialSeconds', 180, @(x) isscalar(x) && isfinite(x) && x>=0);
addParameter(p, 'YAxisRangeFraction', 0.30, @(x) isscalar(x) && isfinite(x) && x>0 && x<=1);
addParameter(p, 'Visible', 'on', @(x) any(strcmp(x, {'on', 'off'})));
parse(p, varargin{:});
options = p.Results;

channelsPerTab = options.ChannelsPerTab;
if isempty(channelsPerTab)
    channelsPerTab = options.ChannelsPerFigure;
end

%% 2. 이벤트, 수면단계, 표시할 시간 범위
[Events, Stages, EventCounts] = PSG_Events(result);
window = options.Window;
if isempty(window)
    window = [min(result.channelTable.StartTime), max(result.channelTable.EndTime)];
end
window(1)=max(window(1),min(result.channelTable.StartTime)+seconds(options.SkipInitialSeconds));
settings.skipInitialSeconds=options.SkipInitialSeconds;
settings.yAxisRangeFraction=options.YAxisRangeFraction;
assert(window(2) > window(1), 'PSG:Window', '종료 시각은 시작 시각보다 뒤여야 합니다.');

%% 3. Figure 하나와 탭 그룹 생성
delete(findall(groot, 'Type', 'figure', 'Tag', 'PSGPreview'));
[~, subject] = fileparts(result.sourceFolder);
figures = figure('Color', 'w', 'Name', sprintf('%s | %s', subject, result.format), ...
    'NumberTitle', 'off', 'Tag', 'PSGPreview', ...
    'MenuBar', 'figure', 'ToolBar', 'figure', ...
    'Position', settings.figurePosition, 'Visible', options.Visible);
tabGroup = uitabgroup(figures, 'Units', 'normalized', 'Position', [0 0 1 1]);

channelCount = numel(result.channels);
tabCount = ceil(channelCount / channelsPerTab);
allAxes = gobjects(0, 1);
% 네 번째 출력: 그래프에 실제 전달한 시간/값. result의 전체 샘플과 구분.
PlotData = struct('Window', window, 'ChannelsPerTab', channelsPerTab, ...
    'MaxPlotPoints', options.MaxPlotPoints, 'Settings', settings, ...
    'Channels', struct([]), 'SleepStage', struct);
PlotData.Channels = repmat(struct('index', 0, 'name', '', 'unit', '', ...
    'time', NaT(0,1), 'data', []), channelCount, 1);

%% 4. 각 탭: 수면단계 한 줄 + 해당 채널들의 그래프
for tabIndex = 1:tabCount
    firstChannel = (tabIndex - 1) * channelsPerTab + 1;
    lastChannel = min(tabIndex * channelsPerTab, channelCount);
    channelIndices = firstChannel:lastChannel;

    tabTitle = sprintf('Channels %d-%d', firstChannel, lastChannel);
    currentTab = uitab(tabGroup, 'Title', tabTitle);
    layout = tiledlayout(currentTab, numel(channelIndices) + 1, 1, ...
        'TileSpacing', 'compact', 'Padding', 'compact');
    axesInTab = gobjects(numel(channelIndices) + 1, 1);

    axesInTab(1) = nexttile(layout);
    [stageTime, stageLevel] = drawSleepStages(axesInTab(1), Stages, window, settings);
    PlotData.SleepStage = struct('time', stageTime, 'level', stageLevel);

    for row = 1:numel(channelIndices)
        channel = result.channels(channelIndices(row));
        ax = nexttile(layout);
        axesInTab(row + 1) = ax;
        showLegend = (row == 1); 
        [plotTime, plotSignal, displayUnit] = drawChannel(ax, channel, Events, window, options.MaxPlotPoints, settings, showLegend);
        PlotData.Channels(channelIndices(row)) = struct( ...
            'index', channelIndices(row), 'name', channel.name, 'unit', displayUnit, ...
            'time', plotTime, 'data', plotSignal);
    end

    for row = 1:numel(axesInTab)
        ax = axesInTab(row);
        grid(ax, 'on');
        xlim(ax, window);
        xtickformat(ax, settings.timeFormat);
        % Explicit interactions persist when the saved FIG is reopened.
        ax.Interactions = [zoomInteraction panInteraction dataTipInteraction];
        axtoolbar(ax, {'zoomin','zoomout','pan','datacursor','restoreview'});
        ax.Toolbar.Visible = 'on';
    end
    xlabel(axesInTab(end), 'Recorded date/time');
    title(layout, sprintf('%s | channels %d-%d / %d', ...
        subject, firstChannel, lastChannel, channelCount), 'Interpreter', 'none');
    allAxes = [allAxes; axesInTab]; 
end

linkaxes(allAxes, 'x');
tabGroup.SelectedTab = tabGroup.Children(end);
fprintf('Plot ready: %d channels in one figure with %d channel tabs. PSG_Plot does not save files.\n', channelCount, tabCount);
end

function [time, level] = drawSleepStages(ax, Stages, window, settings)
Stages=Stages(Stages.EndTime>window(1) & Stages.StartTime<window(2),:);
Stages.StartTime=max(Stages.StartTime,window(1));
Stages.EndTime=min(Stages.EndTime,window(2));
if isempty(Stages)
    time = window(:);
    level = [NaN; NaN];
    plot(ax, time, level);
    text(ax, 0.5, 0.5, 'Sleep staging unavailable', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center');
else
    % stageCount = height(Stages);
    % time = NaT(stageCount * 3, 1);
    % level = nan(stageCount * 3, 1);
    % time(1:3:end) = Stages.StartTime;
    % time(2:3:end) = Stages.EndTime;
    % level(1:3:end) = Stages.Level;
    % level(2:3:end) = Stages.Level;
    % plot(ax, time, level, 'Color', settings.stageColor, 'LineWidth', settings.stageLineWidth);
    time = [Stages.StartTime; Stages.EndTime(end)];
    level = [Stages.Level; Stages.Level(end)];

    stairs(ax, time, level, ...
        'Color', settings.stageColor, ...
        'LineWidth', settings.stageLineWidth, ...
        'LineStyle', '-', ...
        'Marker', 'none');
end
yticks(ax, 0:5);
yticklabels(ax, {'W', 'REM', 'N1', 'N2', 'N3', 'N4'});
set(ax, 'YDir', 'reverse');
ylim(ax, [-0.4 5.4]);
ylabel(ax, 'Sleep stage');
end

function [time, signal, displayUnit] = drawChannel(ax, channel, Events, window, maxPoints, settings, showLegend)
[time, signal] = preparePlotData(channel, window, maxPoints);
if isempty(time)
    time = window(:);
    signal = [NaN; NaN];
end
curve = plot(ax, time, signal, 'Color', settings.signalColor, ...
    'LineWidth', settings.signalLineWidth);

% 3분 제외 후 자동 Y축 폭의 일부만 표시합니다. 데이터 값은 변경하지 않습니다.
finiteSignal=signal(isfinite(signal));
if ~isempty(finiteSignal)
    automaticLimits=ylim(ax);
    center=median(finiteSignal);
    halfWidth=diff(automaticLimits)*settings.yAxisRangeFraction/2;
    if isfinite(halfWidth) && halfWidth>0
        ylim(ax,center+[-halfWidth halfWidth]);
    end
end
% 축에는 이름과 확인된 단위만 표시합니다. 주의 문구는 단위로 쓰지 않습니다.
displayUnit = getDisplayUnit(channel);
label = string(channel.name);
if strlength(displayUnit) > 0
    label = label + ' (' + displayUnit + ')';
end
ylabel(ax, label, 'Interpreter', 'none');
hold(ax, 'on');
drawEvents(ax, Events, window, settings, showLegend);
uistack(curve, 'top');
end

function drawEvents(ax, Events, window, settings, showLegend)
yLimits = ylim(ax);
groupCount = numel(settings.eventNames);
legendHandles = gobjects(groupCount, 1);

for groupIndex = 1:groupCount
    eventName = settings.eventNames(groupIndex);
    color = settings.eventColors(groupIndex, :);
    inWindow = Events.EventType == eventName & ...
        Events.EndTime > window(1) & Events.StartTime < window(2);

    if any(inWindow)
        startTime = max(Events.StartTime(inWindow), window(1));
        endTime = min(Events.EndTime(inWindow), window(2));
        left = ruler2num(startTime, ax.XAxis)';
        right = ruler2num(endTime, ax.XAxis)';

        xCorners = [left; right; right; left];
        yCorners = repmat(yLimits([1 1 2 2])', 1, numel(left));
        patch(ax, xCorners, yCorners, color, 'FaceAlpha', settings.eventAlpha, ...
            'EdgeColor', 'none', 'HandleVisibility', 'off', ...
            'HitTest', 'off', 'PickableParts', 'none');
    end
    legendHandles(groupIndex) = patch(ax, NaN, NaN, color, ...
        'FaceAlpha', settings.eventAlpha, 'EdgeColor', 'none', ...
        'HitTest', 'off', 'PickableParts', 'none');
end

ylim(ax, yLimits);
if showLegend
    legend(ax, legendHandles, cellstr(settings.eventNames), ...
        'Location', 'northeast', 'Orientation', 'horizontal', 'NumColumns', 3, 'Interpreter','none');
end
end

function [time, signal] = preparePlotData(channel, window, maxPoints)
inWindow = channel.time >= window(1) & channel.time <= window(2);
time = channel.time(inWindow);
signal = channel.data(inWindow);
if isempty(time)
    return;
end

gapIndices = find(seconds(diff(time)) > 1.5 / channel.fs) + 1;
missingIndices = find(~isfinite(signal));

if numel(signal) > maxPoints
    binCount = floor(maxPoints / 2);
    edges = round(linspace(1, numel(signal) + 1, binCount + 1));
    extremaIndices = zeros(binCount * 2, 1);
    for bin = 1:binCount
        first = edges(bin);
        last = edges(bin + 1) - 1;
        [~, minIndex] = min(signal(first:last));
        [~, maxIndex] = max(signal(first:last));
        extremaIndices(2 * bin - 1) = first + minIndex - 1;
        extremaIndices(2 * bin) = first + maxIndex - 1;
    end

    missingStarts = [];
    if ~isempty(missingIndices)
        missingStarts = missingIndices([true; diff(missingIndices) > 1]);
    end
    keepIndices = unique([1; extremaIndices; numel(signal); ...
        gapIndices; max(1, gapIndices - 1); missingStarts]);
    [~, gapIndices] = ismember(gapIndices, keepIndices);
    time = time(keepIndices);
    signal = signal(keepIndices);
end

for gap = numel(gapIndices):-1:1
    index = gapIndices(gap);
    time = [time(1:index - 1); NaT; time(index:end)];
    signal = [signal(1:index - 1); NaN; signal(index:end)];
end
end
function unit = getDisplayUnit(channel)
% EBM 원본 리더는 수정하지 않습니다. 반환 값에 맞는 단위만 표시합니다.
unit = strtrim(string(channel.unit));
if ismissing(unit)
    unit = "";
end
if ~contains(lower(unit), ["ebmread-scaled", "verify", "unverified", "unknown"])
    return; % NOX/EDF 헤더에서 읽은 정상 단위 또는 빈 단위
end
unit = "";
if ~isfield(channel,'sourceFile') || ~isfile(channel.sourceFile)
    return;
end
[~,~,extension] = fileparts(channel.sourceFile);
if ~strcmpi(extension,'.ebm')
    return;
end

% 현재 EBM의 0x99 블록은 단위 문자열입니다.
% V만 EbmRead의 nV -> V 배율과 맞으므로, 다른 단위는 여기서 붙이지 않습니다.
fid = fopen(channel.sourceFile,'rb','ieee-le');
if fid < 0, return; end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
info = dir(channel.sourceFile);
for k = 1:80
    byte = fread(fid,1,'uint8');
    if isempty(byte), return; end
    if byte == 26, break; end
end
if byte ~= 26 || fread(fid,1,'uint8') ~= 0, return; end
firstBlock = ftell(fid);
marker = fread(fid,5,'uint8');
if numel(marker) ~= 5 || ~all(marker == 255), return; end
fseek(fid,firstBlock+31,'bof');
while ftell(fid) + 8 <= info.bytes
    id = fread(fid,1,'uint32');
    bytes = fread(fid,1,'uint32');
    nextBlock = ftell(fid) + bytes;
    if nextBlock > info.bytes, return; end
    if id == hex2dec('99') && bytes <= 256
        text = fread(fid,bytes,'*uint8')';
        text = char(text(text ~= 0));
        if strcmp(strtrim(text),'V'), unit = "V"; end
        return;
    end
    fseek(fid,nextBlock,'bof');
end
end
