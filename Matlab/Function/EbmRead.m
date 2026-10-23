function [ data, varargout ]=EbmRead( fileName )
%   =======================================
%   Embla file reader for Matlab
%   =======================================
%
%   Version 2.0
%   Based on ebmread.m version 1.10 by Rognvaldur J. Saemundsson
%   Modified by Joris Coppens, 071014, TSD Netherlands institute for Neuroscience
%
%    USAGE:
%
%      [ data header ] = EbmRead( fileName );
%
%      fileName  : Filename, including path. If omitted a standard get file dialog will appear.
%
%      data : Calibrated samples; verified SpO2 percent channels retain percent values.
%
%      header: Struct containing
%         header.fileversion      ( should be 4 )
%         header.channelname
%         header.subjectinfo
%         header.channel         ( channel number )
%         header.samplingrate      ( samples per second )
%         header.unitgain         ( multiplier for calibration of raw data in volts)
%         header.starttime      ( number that gives the startdate in internal Matlab units)
%         header.starttimedescr   ( string with readable start time of the recording)
%
%   Multiple recordings in one embla file are supported. Missing data is filled with zeros.
%   Embla files with large gaps between recordings will result in errors due to lack of memory 
%   for the zeros.

%   The original comments:
%    ======================================
%    Embla File Format - EBM Developers Kit
%    ======================================
%
%    Flaga hf. Medical Devices
%    Copyright (c) 1996-1997.
%    All Rights Reserved.
%
%    FILE:         EbmRead.m
%    AUTHOR:       Rognvaldur J. Saemundsson
%
%    VERSION:      1.12
%    MODIFIED:     Joris Coppens
%    Modified line 198 to give an error, where the fseek operation could fail in V1.10.
%    It was failing on non-integer offsets, now it is explicitly turned into an integer.
%    Furthermore, error checking was added.
%
%    Return header structure with more information than just the samplerate
%
%    OVERVIEW
%    ========
%    This software reads EBM files from Embla into a Matlab matrix.
%
%    Technical questions should be directed to:
%
%       support@flaga.is
%
%    USAGE:
%
%      [ data samplingRate ] = EbmRead( fileName, start, noSamples, scaled );
%
%      fileName  : Filename, including path. If [] a standard get file dialog will appear.
%      start     : Starting index of samples to be read (first sample in file is 0)
%      noSamples : Number of samples to be read. If 'ALL' it reads from start to the end of the file.
%      scaled    : If 1 the data is scaled according to unit gain in the file (default is 1)
%                  Scaling should only be used if the unit of the data file is V.
%
%      data : Matrix holding the data (noSamples x 1)
%      samplingRate : Sampling rate of the data (Hz)
%

%      Definitions

EBM_RAWDATA_HEADER = 'Embla data file';
EBM_RESULTS_HEADER = 'Embla results file';

EBM_R_VERSION=hex2dec('80');
EBM_R_SUBJECT_INFO=hex2dec('d0');
EBM_R_HEADER=hex2dec('81');
EBM_R_TIME = hex2dec('84');
EBM_R_CHANNEL=hex2dec('85');
EBM_R_SAMPLING_RATE = hex2dec( '86' );
EBM_R_UNIT_GAIN = hex2dec( '87' );
EBM_R_CHANNEL_NAME=hex2dec('90');
EBM_R_UNIT=hex2dec('99');
header.physicalunit = '';
header.outputunit = 'EbmRead-scaled (verify unit)';
EBM_R_DATA =   hex2dec( '20' );
EBM_UNKNOWN_DATASIZE = hex2dec( 'FFFFFFFF' );
EBM_END_OF_SIGNATURE = hex2dec( '1A' );
EBM_MAX_SIGNATURE_SIZE = 80;
EBM_MAC_ENDIAN = hex2dec( 'FF' );
EBM_INTEL_ENDIAN = hex2dec( '00' );

ERROR_UNKNOWN_SIGNATURE = 'This is not a Embla data file';
ERROR_FILE_NOT_FOUND = 'Could not open data file';
ERROR_UNKNOWN = 'Failure in reading the file';
ERROR_CANCEL = 'Operation was canceled';

DW_SIZE = 4;   % Size of double word
W_SIZE = 2;    % Size of word
BYTE_SIZE = 1; % Size of byte

endian = 'ieee-be';  % Big endian by default (Mac endian)
reopen = 0;          %   Determines if we have to reopen the file after detecting the endian format


%   Open the file in the default endian format

if nargin<1
   [ file path ] = uigetfile('.ebm', 'Select an EBM Data File:' );
   if ( file == 0 )
      error( ERROR_CANCEL );
   end;
   fileName = [ path, file ];
end;

fp= fopen( fileName, 'rb', endian );

if ( fp == - 1 )
   error( ERROR_FILE_NOT_FOUND );
end;

% Read the signature of the file

signature = zeros( 1, EBM_MAX_SIGNATURE_SIZE );
i = 1;
signature( i ) = fread( fp, 1, 'char' );

while  ( signature( i ) ~= EBM_END_OF_SIGNATURE & i < EBM_MAX_SIGNATURE_SIZE )
   i=i+1;
   signature( i ) = fread( fp, 1, 'char' );
end;

if ( i == EBM_MAX_SIGNATURE_SIZE )
   error( ERROR_UNKNOWN_SIGNATURE );
end;

if ( findstr( signature, EBM_RAWDATA_HEADER ) ~= 1 )
   if ( findstr( signature, EBM_RESULTS_HEADER ) ~= 1 )
      error( ERROR_UNKNOWN_SIGNATURE );
   end;
end;

% Read the endian format

ch = fread( fp, 1, 'uchar' );

if ( ch == EBM_MAC_ENDIAN )
   endian = 'ieee-be';
elseif ( ch == EBM_INTEL_ENDIAN )
   endian = 'ieee-le';
   reopen = 1;
   pos = ftell( fp );
   fclose( fp );
   fp = fopen( fileName, 'rb', endian );
   if ( fp == -1 )
      error( ERROR_UNKNOWN );
   end;
   fseek( fp, pos, 'bof' );
else
   error( ERROR_UNKNOWN );
end;

wideId = 1;

% Store the position of the start of the block structure
% If this is not a file with 8 bit block IDs then we will change
% this again.

firstBlockOffset = ftell( fp );

% If the file has wide Id's we will get 5 bytes of 0xFF
% 8 bit interprets this as an invalid block for the rest of the file.
%


ch = fread( fp, 1, 'uchar' );

if( ch == hex2dec( 'FF' ) )
   ch = fread( fp, 1, 'ulong' );
   if ( ch == hex2dec( 'FFFFFFFF' ) )
      % We have 32 bit block IDs so we skip the rest of the
      % 32 byte header and store the position of the block
      % structure which should start right after.

      firstBlockOffset = firstBlockOffset + 31;
      wideId = 1;
   end;
end;

% Seek back to the start of the block structure
fseek( fp, firstBlockOffset, 'bof' );


% Find the data block

rec = 0;
recnum=0;
while 1

   if( wideId ~= 0 )
      rec = fread( fp, 1, 'ulong' );
   else
      rec = fread( fp, 1, 'uchar' );
   end;

   recSize = fread( fp, 1, 'long' );
   recPos = ftell( fp );


   if ( rec == EBM_R_VERSION)
      minor=fread(fp, 1, 'int8');
      major=fread(fp, 1, 'int8');
      header.fileversion=major+0.01*minor;
   end

   if (rec == EBM_R_SUBJECT_INFO)
      tmp=fread(fp, recSize, 'int8');
      header.subjectinfo=deblank(char(tmp'));
   end

   if (rec == EBM_R_HEADER)
      tmp=fread(fp, recSize, 'int8');
      header.extra=deblank(char(tmp'));
   end

   if (rec == EBM_R_TIME)
      year=fread(fp, 1, 'int16');
      month=fread(fp, 1, 'int8');
      day=fread(fp, 1, 'int8');
      hour=fread(fp, 1, 'int8');
      minute=fread(fp, 1, 'int8');
      second=fread(fp, 1, 'int8');
      hsec=fread(fp, 1, 'int8');
      recnum=recnum+1;
      if recnum==1
         header.starttime=datenum(year, month, day, hour, minute, second+ 0.01*hsec);
         header.starttimedescr=datestr(header.starttime, 0);
      end
      starttime(recnum)=datenum(year, month, day, hour, minute, second+ 0.01*hsec);
   end

   if (rec == EBM_R_CHANNEL)
      header.channel=fread(fp, 1, 'int16');
   end

   % Check if we have the sampling rate

   if ( rec == EBM_R_SAMPLING_RATE )
      % The sampling rate is in mHz
      header.samplingrate = fread( fp, 1, 'long' ) / 1000;
   end;

   if ( rec == EBM_R_UNIT_GAIN )
      header.rawunitgain = fread( fp, 1, 'long' );
      header.unitgain = header.rawunitgain * 1e-9; % Preserve legacy scaling except verified SpO2 below.
   end;

   if (rec == EBM_R_UNIT)
      unitText = fread(fp, recSize, '*uint8')';
      header.physicalunit = strtrim(char(unitText(unitText ~= 0)));
   end

   if (rec == EBM_R_CHANNEL_NAME)
      tmp=fread(fp, recSize, 'int8');
      header.channelname=deblank(char(tmp'));
   end

   if ( rec == EBM_R_DATA )
      newdata = fread( fp, recSize/2, 'short' );
      if recnum==1
         data=newdata;
         prlen=length(data);
      else   % We have more than one record
         prev=starttime(recnum-1);
         prevend=prev+prlen/(header.samplingrate*60*60*24);
         current=starttime(recnum);
         fill=current-prevend;
         fillsamples=round(max(0, fill*60*60*24*header.samplingrate));
         filldata=zeros(fillsamples, 1);
         data=[data; filldata; newdata];
         prlen=length([filldata; newdata]);
      end
   end;

   if feof(fp)
      break
   end
   fseek( fp, recPos + recSize, 'bof' );
end


fclose( fp );

% Minimal fix: exported SpO2 (%) with gain=1 already contains percent samples.
% Do not remove 1e-9 from other non-voltage channels without validating their encoding.
if startsWith(strtrim(header.channelname),'SpO2','IgnoreCase',true) && strcmp(header.physicalunit,'%')
   assert(header.rawunitgain == 1, 'EbmRead:SpO2Gain', ...
       'Unverified SpO2 percent gain %g in %s', header.rawunitgain, fileName);
   header.unitgain = 1;
   header.outputunit = '%';
elseif strcmp(header.physicalunit,'V')
   header.outputunit = 'V';
end
header.scalingVersion = 'SpO2-percent-1';
data = data * header.unitgain;


if nargout==2
   varargout(1)={header};
end

% for i = 1:5
%     datestr(starttime(i))
% end