% 1. 파일 열기
filename = 'Data.txt';
fid = fopen(filename, 'r');

if fid == -1
    error('파일을 열 수 없습니다. Data.txt 파일이 현재 폴더에 있는지 확인해주세요.');
end

% 2. 데이터 읽기 (textscan 사용 - 가장 안정적인 방법)
% 형식: 문자열(시간) 1개 + 실수(데이터) 18개
% HeaderLines: 7줄 건너뜀
% CommentStyle: '#'으로 시작하는 줄 무시
formatSpec = ['%s', repmat('%f', 1, 18)]; 
C = textscan(fid, formatSpec, 'Delimiter', ',', 'HeaderLines', 7, 'CommentStyle', '#');

fclose(fid);

% 3. 데이터 정리
% 시간 데이터 추출
raw_time = C{1};

% 전체 숫자 데이터 매트릭스 (18개 컬럼)
data_values = [C{2:end}]; 

% 시간 형식 변환
try
    % 형식: 2026-01-16 163828.867 (yyyy-MM-dd HHmmss.SSS)
    t = datetime(raw_time, 'InputFormat', 'yyyy-MM-dd HHmmss.SSS');
catch
    warning('시간 변환 실패. 인덱스를 X축으로 사용합니다.');
    t = (1:length(raw_time))';
end

% 변수별 데이터 분리
% j1~j6      : 1~6열
% prox1~prox4: 7~10열
% raw1~raw4  : 11~14열 (추가됨)
% tof1~tof4  : 15~18열

j_data    = data_values(:, 1:6);
prox_data = data_values(:, 7:10);
raw_data  = data_values(:, 11:14); % Raw 데이터 추출
tof_data  = data_values(:, 15:18);

% 4. 데이터 Plotting
figure('Name', 'RB10 Robot Data Analysis (Full)', 'Color', 'w');
tl = tiledlayout(4,1); % 4개의 그래프를 세로로 배치
title(tl, 'Robot Data Visualization');

% (1) Joint Angles
nexttile;
plot(t, j_data, 'LineWidth', 1.5);
title('Joint Angles (j1-j6)');
ylabel('Angle (deg)');
legend({'j1', 'j2', 'j3', 'j4', 'j5', 'j6'}, 'Location', 'bestoutside');
grid on;

% (2) Proximity Sensors
nexttile;
plot(t, prox_data, 'LineWidth', 1.5);
title('Proximity Sensors');
ylabel('Value');
legend({'prox1', 'prox2', 'prox3', 'prox4'}, 'Location', 'bestoutside');
grid on;

% (3) Raw Sensors (추가된 부분)
nexttile;
plot(t, raw_data, 'LineWidth', 1.5);
title('Raw Sensors');
ylabel('Raw Value');
legend({'raw1', 'raw2', 'raw3', 'raw4'}, 'Location', 'bestoutside');
grid on;

% (4) ToF Sensors
nexttile;
plot(t, tof_data, 'LineWidth', 1.5);
title('ToF Sensors');
xlabel('Time');
ylabel('Distance (mm)');
legend({'tof1', 'tof2', 'tof3', 'tof4'}, 'Location', 'bestoutside');
grid on;