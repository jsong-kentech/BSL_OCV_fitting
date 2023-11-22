clc;clear;close all
load ('OCV_fit.mat') 

% 가중치 적용 최적화
window_size = 200;
width = 6;     % Width in inches
height = 6;    % Height in inches
alw = 2;    % AxesLineWidth
fsz = 20;      % Fontsize
lw = 2;      % LineWidth
msz = 16;       % MarkerSize


pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); 
set(gca, 'FontSize', fsz, 'LineWidth', alw); 



x_values = OCV2(:,1);
y_values = OCV2(:,2);
OCV_mov = OCV2(:,1);

dvdq = diff(y_values) ./ diff(x_values);
dvdq = [dvdq; dvdq(end)];
dvdq_mov = movmean(dvdq, window_size);


%  주어진 x와 y 데이터 
y_values = dvdq_mov(:,1);

%  x 범위 설정
x_min = 0.1; 
x_max = 0.2; 
x_min2 = 0.8; 
x_max2 = 0.9; 


% 첫 번째 x 범위 내에서 가장 작은 y 값 찾기
x_in_range = x_values(x_values >= x_min & x_values <= x_max); 
y_in_range = y_values(x_values >= x_min & x_values <= x_max); 
[min_y1, min_index1] = min(y_in_range); 
corresponding_x1 = x_in_range(min_index1); 
min1 = find(OCV2(:,1) == corresponding_x1);


% 두 번째 x 범위 내에서 가장 작은 y 값 찾기
x_in_range2 = x_values(x_values >= x_min2 & x_values <= x_max2); 
y_in_range2 = y_values(x_values >= x_min2 & x_values <= x_max2); 
[min_y2, min_index2] = min(y_in_range2);    
corresponding_x2 = x_in_range2(min_index2); 
min2 = find(OCV2(:,1) == corresponding_x2);


% 결과 출력
fprintf('첫 번째 x 범위 [%.2f, %.2f] 내에서 가장 작은 y 값: %.2f\n', x_min, x_max, min_y1);
fprintf('해당 y 값에 대응하는 x 값: %.2f\n', corresponding_x1);
fprintf('두 번째 x 범위 [%.2f, %.2f] 내에서 가장 작은 y 값: %.2f\n', x_min2, x_max2, min_y2);
fprintf('해당 y 값에 대응하는 x 값: %.2f\n', corresponding_x2);


figure(1)
x3 = OCV2(:,1);
w1 = zeros(size(dvdq_mov));
start_index = min1; 
end_index = min2;
w1(start_index:end_index) = dvdq_mov(start_index:end_index);
%w1 = dvdq_mov;
plot(x3,w1,'-g','LineWidth',lw,'MarkerSize',msz);
xlabel('SOC');
ylabel('Weight');
xlim([0 1]);



Cap = OCV2(:, 1);
Q_cell = abs(Cap(end));
x_guess = [0,Q_cell,1,Q_cell];
x_lb = [0,Q_cell*0.5,0,Q_cell*0.5];
x_ub = [1,Q_cell*2,1,Q_cell*2];


results = struct();

counter = 1;


w1_values = {w1,dvdq_mov};
w1_values = cell2mat(w1_values);


% selected_values를 weights_range에 추가


for k = 1
        w1 = w1_values(:,k) .* ones(size(OCV2(:,1)));


      for j =[0:1:20]
         w2 = j .* ones(size(OCV2(:,1)));

         %for j = 1:2
           % w2 = w1_values(:,j) .* ones(size(OCV2(:,1)));
        
        

%fmincon을 사용하여 최적화 수행  
options = optimoptions(@fmincon,'MaxIterations',5000,'StepTolerance',1e-15,'ConstraintTolerance', 1e-15, 'OptimalityTolerance', 1e-15);

problem = createOptimProblem('fmincon', 'objective', @(x) OCV_dvdq_model_07(x,OCP_n1,OCP_p1,OCV2,w1,w2), ...
            'x0', x_guess, 'lb', x_lb, 'ub', x_ub, 'options', options);
ms = MultiStart('Display','iter','UseParallel',true,'FunctionTolerance',1e-15,'XTolerance',1e-15);

[x_id,f_val, exitflag, output] = run(ms,problem,100);
[cost_hat,OCV_hat] = OCV_dvdq_model_07 (x_id,OCP_n1,OCP_p1,OCV2,w1,w2);



% %plot
figure(3)
plot(OCV2(:,1),OCV2(:,2),'b-','LineWidth',lw,'MarkerSize',msz); hold on;
plot(OCV2(:,1),OCV_hat,'r-','LineWidth',lw,'MarkerSize',msz);
xlabel('SOC');
ylabel('OCV (V)');
title('SOC vs. OCV');

%dv/dq plot
x = OCV2 (:,1);
y = OCV2 (:,2);

x_values = [];
for i = 1:(length(x)-1)
    dvdq77(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
     x_values = [x_values; x(i)];
end


y = OCV_hat (:,1);
for i = 1:(length(x) - 1)
    dvdq88(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i)); 
end


%dvdq에 이동 평균 적용
dvdq77_moving_avg = movmean(dvdq77(1:end), window_size);
dvdq88_moving_avg = movmean(dvdq88(1:end), window_size);



%dvdq plot
figure(4)
plot(x_values, dvdq77_moving_avg, 'b-', 'LineWidth', lw, 'MarkerSize', msz); hold on
plot(x_values, dvdq88_moving_avg, 'r-', 'LineWidth', lw, 'MarkerSize', msz);
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
%title('SOC vs. dV/dQ');
ylim([-1 3]);
xlim([0 1]);

figure(5)
plot(OCV2(:,1),w1,'-g','LineWidth',lw,'MarkerSize',msz);
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
xlim([0 1]);
ylim([0 2]);



%peak값 추정
y_values = dvdq77_moving_avg(1,:); % y 값 배열
y_values2 = dvdq88_moving_avg(1,:); % y1 겂 배열 

% 첫 번째 x 범위
x_min = 0.15; % 최소 x 범위
x_max = 0.25; % 최대 x 범위

% 두 번째 x 범위
x_min2 = 0.26; % 다른 최소 x 범위
x_max2 = 0.35; % 다른 최대 x 범위

%세 번쩨 x 범위
x_min3 = 0.45;
x_max3 = 0.53;

%네 번째 x 범위
x_min4 = 0.65; % 다른 최소 x 범위
x_max4 = 0.72; % 다른 최대 x 범위

%다섯 번쩨 x 범위
x_min5 = 0.73;
x_max5 = 0.8;





% 첫 번째 x 범위 내에서 가장 큰 y 값 찾기
x_in_range = x_values(x_values >= x_min & x_values <= x_max); % x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min & x_values <= x_max); % x 범위에 해당하는 y 값 추출
y_in_range2 = y_values2(x_values >= x_min & x_values <= x_max); 
[max_y1, max_index1] = max(y_in_range); % 첫 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
[max2_y1, max2_index1] = max(y_in_range2); % 첫 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
% 첫 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 
corresponding_x1 = x_in_range(max_index1); % 해당 y 값에 대응하는 x 값 찾기
corresponding_x11 = x_in_range(max2_index1); % 해당 y2 값에 대응하는 x 값 찾기
index1 = find(OCV2(:,1) == corresponding_x1);
index11 = find(OCV2(:,1) == corresponding_x11);
sim_max1 = y_values2(max2_index1);


% 두 번째 x 범위 내에서 가장 큰 y 값 찾기
x_in_range = x_values(x_values >= x_min2 & x_values <= x_max2); % 다른 x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min2 & x_values <= x_max2); % 다른 x 범위에 해당하는 y 값 추출
y_in_range2 = y_values2(x_values >= x_min2 & x_values <= x_max2); 
[max_y2, max_index2] = max(y_in_range);
[max2_y2, max2_index2] = max(y_in_range2);
% 두 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 
corresponding_x2 = x_in_range(max_index2); % 해당 y 값에 대응하는 x 값 찾기
corresponding_x22 = x_in_range(max2_index2); % 해당 y2 값에 대응하는 x 값 찾기
index2 = find(OCV2(:,1) == corresponding_x2);
index22 = find(OCV2(:,1) == corresponding_x22);
sim_max2 = y_values2(max2_index2);


% 세 번째 x 범위 내에서 가장 큰 y 값 찾기
x_in_range = x_values(x_values >= x_min3 & x_values <= x_max3); % 다른 x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min3 & x_values <= x_max3); % 다른 x 범위에 해당하는 y 값 추출
y_in_range2 = y_values2(x_values >= x_min3 & x_values <= x_max3);
[max_y3, max_index3] = max(y_in_range); % 두 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
[max2_y3, max2_index3] = max(y_in_range2); 
% 세 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 
corresponding_x3 = x_in_range(max_index3); % 해당 y 값에 대응하는 x 값 찾기
corresponding_x33 = x_in_range(max2_index3); % 해당 y2 값에 대응하는 x 값 찾기
index3 = find(OCV2(:,1) == corresponding_x3);
index33 = find(OCV2(:,1) == corresponding_x33);
sim_max3 = y_values2(max2_index3);


%네 번째 x범위 내에서 가장 큰 y값 찾기
x_in_range = x_values(x_values >= x_min4 & x_values <= x_max4); % 다른 x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min4 & x_values <= x_max4); % 다른 x 범위에 해당하는 y 값 추출
y_in_range2 = y_values2(x_values >= x_min4 & x_values <= x_max4);
[max_y4, max_index4] = max(y_in_range); % 두 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
[max2_y4, max2_index4] = max(y_in_range2); 
% 네 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 
corresponding_x4 = x_in_range(max_index4); % 해당 y 값에 대응하는 x 값 찾기
corresponding_x44 = x_in_range(max2_index4); % 해당 y2 값에 대응하는 x 값 찾기
index4 = find(OCV2(:,1) == corresponding_x4);
index44 = find(OCV2(:,1) == corresponding_x44);
sim_max4 = y_values2(max2_index4);

%다섯 번째 x범위 내에서 가장 큰 y값 찾기
x_in_range = x_values(x_values >= x_min5 & x_values <= x_max5); % 다른 x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min5 & x_values <= x_max5); % 다른 x 범위에 해당하는 y 값 추출
y_in_range2 = y_values2(x_values >= x_min5 & x_values <= x_max5);
[max_y5, max_index5] = max(y_in_range); % 두 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
[max2_y5, max2_index5] = max(y_in_range2); 
% 다섯 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 
corresponding_x5 = x_in_range(max_index5); % 해당 y 값에 대응하는 x 값 찾기
corresponding_x55 = x_in_range(max2_index5); % 해당 y2 값에 대응하는 x 값 찾기
index5 = find(OCV2(:,1) == corresponding_x5);
index55 = find(OCV2(:,1) == corresponding_x55);
sim_max5 = y_values2(max2_index5);


e1 =sqrt((max_y1-sim_max1).^2);
e2 =sqrt((max_y2-sim_max2).^2);
e3 =sqrt((max_y3-sim_max3).^2);
e4 =sqrt((max_y4-sim_max4).^2);
e5 =sqrt((max_y5-sim_max5).^2);
e6 =sqrt((corresponding_x1-corresponding_x11).^2);
e7 =sqrt((corresponding_x2-corresponding_x22).^2);
e8 =sqrt((corresponding_x3-corresponding_x33).^2);
e9 =sqrt((corresponding_x4-corresponding_x44).^2);
e10 =sqrt((corresponding_x5-corresponding_x55).^2);


%peak값의 average
error_dvdq = sqrt((max_y1-sim_max1).^2+(max_y2-sim_max2).^2+(max_y3-sim_max3).^2+(max_y4-sim_max4).^2+(max_y5-sim_max5).^2);
error_soc = sqrt(((corresponding_x1-corresponding_x11).^2 + (corresponding_x2-corresponding_x22).^2 + (corresponding_x3-corresponding_x33).^2 + (corresponding_x4-corresponding_x44).^2 + (corresponding_x5-corresponding_x55).^2) / 5);
disp(['error dvdq값: ', num2str(error_dvdq)]);
disp(['error soc값: ', num2str(error_soc)])
error_disp = ['e1: ', num2str(e1), ' e2: ', num2str(e2), ' e3: ', num2str(e3), ' e4: ', num2str(e4), ' e5: ', num2str(e5), ' e6: ', num2str(e6), ' e7: ', num2str(e7),' e8: ', num2str(e8),' e9: ', num2str(e9),' e10: ', num2str(e10)];



peak_position = ['Dpeak soc1: ', num2str(corresponding_x1), 'Fpeak soc1: ', num2str(corresponding_x11), 'Dpeak soc2: ' , num2str(corresponding_x2), 'Fpeak soc2: ', num2str(corresponding_x22), 'Dpeak soc3: ', num2str(corresponding_x3), ' Fpeak soc3: ', num2str(corresponding_x33), 'Dpeak soc4: ', num2str(corresponding_x4),'Fpeak soc4: ', num2str(corresponding_x44),'Dpeak soc5: ' , num2str(corresponding_x5), 'Fpeak soc5: ', num2str(corresponding_x55)];


disp(['Dpeak soc1', num2str(corresponding_x1)]);
disp(['Fpeak soc1', num2str(corresponding_x11)]);
disp(['Dpeak soc2', num2str(corresponding_x2)]);
disp(['Fpeak soc2', num2str(corresponding_x22)]);
disp(['Dpeak soc3', num2str(corresponding_x3)]);
disp(['Fpeak soc3', num2str(corresponding_x33)]);
disp(['Dpeak soc4', num2str(corresponding_x4)]);
disp(['Fpeak soc4', num2str(corresponding_x44)]);
disp(['Dpeak soc5', num2str(corresponding_x5)]);
disp(['Fpeak soc55', num2str(corresponding_x55)]);

figure(6)
%peak 좌표 표시
plot(x_values, dvdq77_moving_avg, 'b-', 'LineWidth', lw, 'MarkerSize', msz,'DisplayName', 'data');
hold on;
plot(x_values, dvdq88_moving_avg, 'r-', 'LineWidth', lw, 'MarkerSize', msz,'Color',[1,0,0,0.5],'DisplayName', 'fit');
ylim([0 2]);


% 별 마커와 텍스트 추가
x_star =[corresponding_x1,corresponding_x2,corresponding_x3,corresponding_x4,corresponding_x5]; 
roundedValue1 = round(x_star * 10^3) / 10^3;
y_star = dvdq77_moving_avg([index1,index2,index3,index4,index5]); 
star_handle = plot(x_star, y_star, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'MarkerSize', 12, 'DisplayName', 'Data Peak');
text(x_star, zeros(size(x_star))+0.2, arrayfun(@num2str, roundedValue1, 'UniformOutput', false), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'b','FontSize', 8); 


% 네모 마커와 텍스트 추가
x_square = [corresponding_x11,corresponding_x22,corresponding_x33,corresponding_x44,corresponding_x55]; 
roundedValue2 = round(x_square * 10^3) / 10^3; 
y_square = dvdq88_moving_avg([index11,index22,index33,index44,index55]); 
square_handle = plot(x_square, y_square, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 8, 'Color', [1, 0, 0, 0.5], 'DisplayName', 'Fit Peak');
text(x_square, zeros(size(x_square))+0.1, arrayfun(@num2str, roundedValue2, 'UniformOutput', false), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center','Color', 'r','FontSize', 8);  



% 네모표시에 대한 선
for i = 1:length(x_star)
    line([x_star(i), x_star(i)], [0, y_star(i)], 'LineStyle', ':', 'Color', 'b');
end
% 네모표시에 대한 선
for i = 1:length(x_square)
    line([x_square(i), x_square(i)], [0, y_square(i)], 'LineStyle', '--', 'Color','r');
end


legend([star_handle, square_handle], {'Data Peak', 'Fit Peak'});


% %OCP_n,OCP_p dvdq
x_1 = x_id(1,1) + (1/x_id(1,2));
y_1 = x_id(1,3) - (1/x_id(1,4));


OCP_n1(:,3) = ((OCP_n1(:,1)-x_id(1,1))/(x_1-x_id(1,1)));
OCP_p1(:,3) = ((OCP_p1(:,1)-x_id(1,3))/(y_1-x_id(1,3))); 


x = OCP_n1 (1:end,3);
y = OCP_n1 (1:end,2);


start_value = 0;
end_value = 1;



x_values3 = [];
for i = 1:(length(x) - 1)
    if x(i) >= start_value && x(i)<=end_value
    dvdq51(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));   
    x_values3 = [x_values3; x(i)];
    end
end


x = OCP_p1 (1:end,3);
y = OCP_p1 (1:end,2);


x_values4 = [];
for i = 1:(length(x) - 1)
    if x(i) >= start_value && x(i)<=end_value
    dvdq61(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));   
    x_values4 = [x_values4; x(i)];
    end
end



% dvdq51에 이동 평균 적용
idx_start = find(OCP_n1 (1:end,3) >= 0);
first_different_idx1= idx_start(1);
dvdq51 = movmean(dvdq51(first_different_idx1:end),window_size);
%x_values_moving_avg = movmean(x_values, window_size);


% dvdq61에 이동 평균 적용
idx_start = find(OCP_p1 (1:end,3) >= 0);
first_different_idx2 = idx_start(1);
dvdq61 = movmean(dvdq61(first_different_idx2:end),window_size);
%x_values2_moving_avg = movmean(x_values2, window_size);


% ocp_n,ocp_p플롯
figure(8)
min_length = min(length(x_values3), length(abs(dvdq51)));
plot(x_values3(1:min_length), abs(dvdq51(1:min_length)), 'b-', 'LineWidth', lw, 'MarkerSize', msz); hold on
min_length = min(length(x_values4), length(abs(dvdq61)));
plot(x_values4(1:min_length), abs(dvdq61(1:min_length)), 'b-', 'LineWidth', lw, 'MarkerSize', msz); hold on
ylim([0 2]);
% % 가중치 적용 최적화


% 실제 관측값과 모델의 예측 값 사이의 차이

error = OCV_hat(min1:min2)-OCV2(min1:min2,2);

% 제곱을 계산하여 평균
squared_error = error.^2;

mean_squared_error = mean(squared_error);
root_squared_error = sqrt(error.^2);
% 평균값의 제곱근을 취하여 RMSE 값
rmse_value = sqrt(mean_squared_error);
disp(['RMSE 값: ', num2str(rmse_value)]);


figure(9)
plot(OCV2(min1:min2,1),error,'DisplayName',['No. =' num2str(counter)]);
legend('show');
xlabel('SOC');
ylabel('Error(mV)');
xlim([0 1]);
xticks(0:0.1:1); 
hold on;

d_error = dvdq77_moving_avg(min1:min2) - dvdq88_moving_avg(min1:min2);
figure(10)
plot(x_values(min1:min2),d_error,'DisplayName',['No. =' num2str(counter)]);
legend('show');
xlabel('SOC');
ylabel('Error(mV)');
xlim([0 1]);
xticks(0:0.1:1); 
hold on;


        % 결과 저장
        results(counter).w1 = w1;
        results(counter).w2 = w2;
        results(counter).x_id = x_id;
        results(counter).f_val = f_val;
        results(counter).exitflag = exitflag;
        results(counter).output = output;
        results(counter).RMSE = rmse_value; 
        results(counter).Error_soc = error_soc;
        results(counter).Error_dvdq = error_dvdq;
        results(counter).Error_each = error_disp;
        results(counter).each_peak_position = peak_position; 
        results(counter).OCV_hat = OCV_hat;
        results(counter).dvdq88_moving_avg = dvdq88_moving_avg;
        results(counter).dvdq77_moving_avg = dvdq77_moving_avg;
        results(counter).x_star = x_star;
        results(counter).x_square = x_square;
        results(counter).index1 = index1;
        results(counter).index2 = index2;
        results(counter).index3 = index3;
        results(counter).index4 = index4;
        results(counter).index5 = index5;
        results(counter).index11 = index11;
        results(counter).index22 = index22;
        results(counter).index33 = index33;
        results(counter).index44 = index44;
        results(counter).index55 = index55;

        results(counter).corresponding_x1 = corresponding_x1;
        results(counter).corresponding_x2 = corresponding_x2;
        results(counter).corresponding_x3 = corresponding_x3;
        results(counter).corresponding_x4 = corresponding_x4;
        results(counter).corresponding_x5 = corresponding_x5;
        results(counter).corresponding_x11 = corresponding_x11;
        results(counter).corresponding_x22 = corresponding_x22;
        results(counter).corresponding_x33 = corresponding_x33;
        results(counter).corresponding_x44 = corresponding_x44;
        results(counter).corresponding_x55 = corresponding_x55;

        results(counter).e6 = e6;
        results(counter).e7 = e7;
        results(counter).e8 = e8;
        results(counter).e9 = e9;
        results(counter).e10 = e10;

                counter = counter + 1;

        end
end




width = 6;     % Width in inches
height = 6;    % Height in inches
alw = 2;    % AxesLineWidth
fsz = 20;      % Fontsize
lw = 2;      % LineWidth
msz = 16;       % MarkerSize

Error_soc_values = [results.Error_soc]'; 
[soc_val,soc_idx] = min(Error_soc_values);

disp(['soc 값: ', num2str(soc_val)]);
disp(['soc 인덱스: ', num2str(soc_idx)]);


maxError = max([results(soc_idx).e6,results(soc_idx).e7,results(soc_idx).e8,results(soc_idx).e9,results(soc_idx).e10]);
disp(['Max_error: ', num2str(maxError)])


% plot


figure('position', [0 0 500 400] );
% 첫 번째 그래프
subplot(2, 1, 1);

plot(OCV2(:,1),OCV2(:,2),'b-','LineWidth',lw,'MarkerSize',msz);
hold on;
plot(OCV2(:,1),results(soc_idx).OCV_hat,'r-','LineWidth',lw,'MarkerSize',msz,'Color',[1,0,0,0.5]);
xlabel('SOC');
ylabel('OCV (V)');
title('SOC vs. OCV');
yyaxis right;
ax = gca;  % 현재 축 객체 가져오기
ax.YColor = 'k';  % 검정색으로 설정
ylabel('Weight');
plot(OCV2(:,1),results(soc_idx).w2,'-g','LineWidth',lw,'MarkerSize',msz,'Color',[0,1,0,0.5]);
legend('FCC data','FCC fit','Weight','Location', 'none', 'Position', [0.2 0.85 0.1 0.05],'FontSize', 6);
xlim([0 1]);
ylim([0 20]);


%두 번째 그래프
subplot(2, 1, 2);

plot(x_values, [results(soc_idx).dvdq77_moving_avg]', 'b-', 'LineWidth', lw, 'MarkerSize', msz); hold on;
plot(x_values, [results(soc_idx).dvdq88_moving_avg]', 'r-', 'LineWidth', lw, 'MarkerSize', msz,'Color',[1,0,0,0.5]);
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
title('SOC vs. dV/dQ');
ylim([0 2]);
yyaxis right;
ax = gca; 
ax.YColor = 'k';  
ylabel('Weight');
plot(OCV2(:,1),results(soc_idx).w1,'-g','LineWidth',lw,'MarkerSize',msz,'Color',[0,1,0,0.2]);

ylim([0 2]);
xlim([0 1]);



% 별 마커와 텍스트 추가
x_star = [results(soc_idx).corresponding_x1,results(soc_idx). corresponding_x2,results(soc_idx).corresponding_x3,results(soc_idx). corresponding_x4,results(soc_idx). corresponding_x5]; 
roundedValue1 = round(x_star * 10^3) / 10^3;
y_star = results(soc_idx).dvdq77_moving_avg([results(soc_idx). index1,results(soc_idx). index2, results(soc_idx). index3, results(soc_idx). index4,results(soc_idx). index5]); 
star_handle = plot(x_star, y_star, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'MarkerSize', 12, 'DisplayName', 'Data Peak');
text(x_star, zeros(size(x_star))+0.2, arrayfun(@num2str, roundedValue1, 'UniformOutput', false), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'b','FontSize', 8); 


% 네모 마커와 텍스트 추가
x_square = [results(soc_idx). corresponding_x11,results(soc_idx).corresponding_x22,results(soc_idx). corresponding_x33,results(soc_idx). corresponding_x44,results(soc_idx). corresponding_x55]; 
roundedValue2 = round(x_square * 10^3) / 10^3; 
y_square = results(soc_idx).dvdq88_moving_avg([results(soc_idx). index11,results(soc_idx). index22,results(soc_idx). index33,results(soc_idx). index44,results(soc_idx). index55]); 
square_handle = plot(x_square, y_square, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 8, 'Color', [1, 0, 0, 0.5], 'DisplayName', 'Fit Peak');
text(x_square, zeros(size(x_square))+0.1, arrayfun(@num2str, roundedValue2, 'UniformOutput', false), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center','Color', 'r','FontSize', 8);  


% 네모표시에 대한 선
for i = 1:length(x_star)
    line([x_star(i), x_star(i)], [0, y_star(i)], 'LineStyle', ':', 'Color', 'b');
end
% 네모표시에 대한 선
for i = 1:length(x_square)
    line([x_square(i), x_square(i)], [0, y_square(i)], 'LineStyle', '--', 'Color','r');
end


legend([star_handle, square_handle], {'Data Peak', 'Fit Peak'});





function [cost,OCV_sim] = OCV_dvdq_model_07(x, OCP_n1, OCP_p1, OCV2, w1, w2)
    x_0 = x(1);
    QN = x(2);
    y_0 = x(3);
    QP = x(4);

    Cap = OCV2(:, 1);
    if (OCV2(end, 2) < OCV2(1, 2)) % Discharge OCV
        x_sto = -(Cap - Cap(1)) / QN + x_0;
        y_sto = (Cap - Cap(1)) / QP + y_0;
    else  % Charge OCV
        x_sto = (Cap - Cap(1)) / QN + x_0;
        y_sto = -(Cap - Cap(1)) / QP + y_0;
    end

    OCP_n_sim = interp1(OCP_n1(:, 1), OCP_n1(:, 2), x_sto, 'linear', 'extrap');
    OCP_p_sim = interp1(OCP_p1(:, 1), OCP_p1(:, 2), y_sto, 'linear', 'extrap');
    OCV_sim = OCP_p_sim - OCP_n_sim;

    
    % dV/dQ 값들 계산
    window_size = 200;

    x_values = OCV2(:, 1);
    y_values = OCV2(:, 2);
    y_sim_values = OCV_sim(:, 1);

    dvdq = diff(y_values) ./ diff(x_values);
    dvdq_sim = diff(y_sim_values) ./ diff(x_values);
    dvdq = [dvdq; dvdq(end)];
    dvdq_mov = movmean(dvdq, window_size);

    dvdq_sim = [dvdq_sim; dvdq_sim(end)];
    dvdq_sim_mov = movmean(dvdq_sim,window_size);

    OCV_sim_mov =  movmean(OCV_sim,window_size);

    OCV_mov = movmean(OCV2(:,2),window_size);

    cost_dvdq = sum(((dvdq_sim_mov - dvdq_mov).^2./mean(dvdq_mov)).*w1);

    % OCV 비용 계산
    cost_OCV = sum(((OCV_sim_mov - OCV_mov).^2./mean(OCV_mov)).*w2);
   
    % 비용 합산 
    cost = cost_dvdq + cost_OCV;
end
