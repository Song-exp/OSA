function result=CBCT(folderPath,varargin)
% CBCT Read one DICOM series, retaining raw voxels and patient coordinates.
% result=CBCT(folderPath,'ShowFigure',false)
p=inputParser;
addParameter(p,'ShowFigure',false,@(x)islogical(x)&&isscalar(x));
parse(p,varargin{:});
if nargin<1 || isempty(folderPath)
    folderPath=uigetdir('E:\ClinicalData\[성인] ENT_ORTHO\데이터정리\CBCT','CBCT 환자 폴더 선택');
end
result=[];
if isequal(folderPath,0),return;end
assert(isfolder(folderPath),'CBCT:Path','CBCT 폴더가 없습니다: %s',folderPath);
collection=dicomCollection(folderPath,'IncludeSubfolders',true,'DisplayWaitbar',false);
assert(height(collection)==1,'CBCT:Series', ...
    'DICOM 시리즈가 %d개입니다. 한 시리즈만 포함된 환자 폴더가 필요합니다: %s',height(collection),folderPath);
files=string(collection.Filenames{1});files=files(:);n=numel(files);
assert(n>=2,'CBCT:Slices','단일 프레임 DICOM 슬라이스가 2개 이상 필요합니다.');
positions=zeros(n,3);orientations=zeros(n,6);slopes=ones(n,1);intercepts=zeros(n,1);
hasRescale=false(n,1);units=strings(n,1);
for k=1:n
    info=dicominfo(files(k));
    if k==1,firstHeader=info;end
    assert(~isfield(info,'NumberOfFrames') || double(info.NumberOfFrames)==1, ...
        'CBCT:Frames','다중 프레임 DICOM은 별도 처리가 필요합니다: %s',files(k));
    assert(isfield(info,'ImagePositionPatient') && isfield(info,'ImageOrientationPatient'), ...
        'CBCT:Geometry','환자 좌표/방향 정보가 없습니다: %s',files(k));
    positions(k,:)=double(info.ImagePositionPatient(:))';
    orientations(k,:)=double(info.ImageOrientationPatient(:))';
    hasSlope=isfield(info,'RescaleSlope');hasIntercept=isfield(info,'RescaleIntercept');
    assert(hasSlope==hasIntercept,'CBCT:Rescale','Slope/Intercept 중 하나만 존재합니다: %s',files(k));
    if hasSlope
        slopes(k)=double(info.RescaleSlope);intercepts(k)=double(info.RescaleIntercept);hasRescale(k)=true;
    end
    if isfield(info,'RescaleType'),units(k)=string(info.RescaleType);end
end
assert(all(isfinite(slopes))&&all(isfinite(intercepts)),'CBCT:Rescale','잘못된 픽셀 변환 계수입니다.');
assert(max(abs(orientations-orientations(1,:)),[],'all')<1e-6,'CBCT:Geometry','슬라이스 방향이 서로 다릅니다.');

% 기존 CBCT 코드의 dicomreadVolume을 재사용합니다. 보간하지 않습니다.
[raw,spatial,dimension]=dicomreadVolume(collection,'MakeIsotropic',false);
volume=squeeze(raw);clear raw;
assert(ndims(volume)==3 && size(volume,3)==n,'CBCT:Shape','예상하지 못한 DICOM 볼륨 차원입니다.');
actualPositions=double(spatial.PatientPositions);
assert(isequal(size(actualPositions),[n 3]),'CBCT:Geometry','슬라이스 좌표 수가 맞지 않습니다.');
order=zeros(n,1);
for k=1:n
    distance=sqrt(sum((positions-actualPositions(k,:)).^2,2));
    matches=find(distance<1e-4);
    assert(isscalar(matches),'CBCT:Geometry','중복되거나 일치하지 않는 슬라이스 좌표입니다.');
    order(k)=matches;
end
assert(numel(unique(order))==n,'CBCT:Geometry','슬라이스 매핑이 중복되었습니다.');
files=files(order);slopes=slopes(order);intercepts=intercepts(order);
hasRescale=hasRescale(order);units=units(order);orientations=orientations(order,:);
rescaled=zeros(size(volume),'single');
for k=1:n
    rescaled(:,:,k)=single(double(volume(:,:,k))*slopes(k)+intercepts(k));
end
normal=cross(orientations(1,1:3),orientations(1,4:6));normal=normal/norm(normal);
sliceSpacing=abs(diff(actualPositions*normal'));
assert(all(sliceSpacing>0),'CBCT:Geometry','중복된 슬라이스 평면입니다.');
zSpacing=median(sliceSpacing);
regular=all(abs(sliceSpacing-zSpacing)<=max(1e-4,zSpacing*1e-3));
if ~regular,zSpacing=NaN;end
pixelSpacings=double(spatial.PixelSpacings);
assert(size(pixelSpacings,2)==2,'CBCT:Geometry','픽셀 간격 형식을 확인하세요.');
regularPixels=max(abs(pixelSpacings-pixelSpacings(1,:)),[],'all')<1e-6;
voxelSize=[pixelSpacings(1,:) zSpacing];
if ~regularPixels,voxelSize(1:2)=NaN;end
unit='Rescaled DICOM values';
if ~any(hasRescale),unit='Raw DICOM values';
elseif all(strcmpi(strtrim(units),'HU')),unit='HU';end
sliceTable=table((1:n)',files,actualPositions,orientations,slopes,intercepts,hasRescale,units, ...
    'VariableNames',{'Slice','SourceFile','PatientPositionMm','PatientOrientation','RescaleSlope','RescaleIntercept','HasRescale','RescaleType'});
result=struct('SourceFolder',char(folderPath),'Volume',volume,'RescaledVolume',rescaled, ...
    'Unit',unit,'Spatial',spatial,'SliceTable',sliceTable,'VoxelSizeMm',voxelSize, ...
    'VoxelSizeOrder','row, column, slice','SliceSpacingMm',sliceSpacing, ...
    'RegularSpacing',regular&&regularPixels,'VolumeDimension',dimension, ...
    'FirstSourceHeader',firstHeader,'SeriesInstanceUID',string(collection.SeriesInstanceUID(1)), ...
    'SchemaVersion','CBCT-1.0','CreatedAt',datetime('now'));
if p.Results.ShowFigure
    sliceViewer(rescaled);
end
fprintf('CBCT: %d x %d x %d voxels; %s\n',size(volume,1),size(volume,2),size(volume,3),unit);
end
