clc;clear;close all
load ('OCV_fit.mat') 

% 가중치 적용 최적화

width = 6;     % Width in inches
height = 6;    % Height in inches
alw = 2;    % AxesLineWidth
fsz = 20;      % Fontsize
lw = 2;      % LineWidth
msz = 16;       % MarkerSize


pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %Set properties

window_size = 50;

x_values = OCV2(:,1);
y_values = OCV2(:,2);
OCV2_mov = OCV2(:,1);

dvdq = diff(y_values) ./ diff(x_values);
dvdq = [dvdq; dvdq(end)];
dvdq_mov = movmean(dvdq, window_size);



% % 주어진 x와 y 데이터
x_values = OCV2(:,1); % x 값 배열
y_values = dvdq_mov(:,1); % y 값 배열

%  x 범위 설정
x_min = 0.1; % 최소 x 범위
x_max = 0.2; % 최대 x 범위
x_min2 = 0.8; % 다른 최소 x 범위
x_max2 = 0.9; % 다른 최대 x 범위


% 첫 번째 x 범위 내에서 가장 작은 y 값 찾기
x_in_range = x_values(x_values >= x_min & x_values <= x_max); % x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min & x_values <= x_max); % x 범위에 해당하는 y 값 추출
[min_y1, min_index1] = min(y_in_range); % 첫 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
corresponding_x1 = x_in_range(min_index1); % 해당 y 값에 대응하는 x 값 찾기
index1 = find(OCV2(:,1) == corresponding_x1);


% 두 번째 x 범위 내에서 가장 작은 y 값 찾기
x_in_range2 = x_values(x_values >= x_min2 & x_values <= x_max2); % 다른 x 범위에 해당하는 x 값 추출
y_in_range2 = y_values(x_values >= x_min2 & x_values <= x_max2); % 다른 x 범위에 해당하는 y 값 추출
[min_y2, min_index2] = min(y_in_range2);    % 두 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
corresponding_x2 = x_in_range2(min_index2); % 해당 y 값에 대응하는 x 값 찾기
index2 = find(OCV2(:,1) == corresponding_x2);


% 결과 출력
fprintf('첫 번째 x 범위 [%.2f, %.2f] 내에서 가장 작은 y 값: %.2f\n', x_min, x_max, min_y1);
fprintf('해당 y 값에 대응하는 x 값: %.2f\n', corresponding_x1);
fprintf('두 번째 x 범위 [%.2f, %.2f] 내에서 가장 작은 y 값: %.2f\n', x_min2, x_max2, min_y2);
fprintf('해당 y 값에 대응하는 x 값: %.2f\n', corresponding_x2);


figure(1)
x3 = OCV2_mov(:,1);
w1 = zeros(size(dvdq_mov));
start_index = index1; 
end_index = index2;
w1(start_index:end_index) = dvdq_mov(start_index:end_index);
plot(x3,w1,'-g','LineWidth',lw,'MarkerSize',msz);
xlabel('SOC');
ylabel('Weight');
xlim([0 1]);


figure(2)
x3 = OCV2_mov(:,1);
w2 = ones(size(x3));
plot(x3,w2,'-g','LineWidth',lw,'MarkerSize',msz);
xlabel('SOC');
ylabel('Weight');
xlim([0 1]);


x_guess = [0.001,1.2,0.9,1.8];
x_lb = [0,1,0,1];
x_ub = [1,1.9,1,1.9];


%두 번째 최적화 수행
%fmincon을 사용하여 최적화 수행  
options = optimoptions(@fmincon,'MaxIterations',5000,'StepTolerance',1e-15,'ConstraintTolerance', 1e-15, 'OptimalityTolerance', 1e-15);


problem = createOptimProblem('fmincon', 'objective', @(x) OCV_dvdq_model_06(x,OCP_n1,OCP_p1,OCV2,w1,w2), ...
            'x0', x_guess, 'lb', x_lb, 'ub', x_ub, 'options', options);
 ms = MultiStart('Display', 'iter');

[x_id,f_val, exitflag, output] = run(ms,problem,100);
[cost_hat,OCV_hat] = OCV_dvdq_model_06 (x_id,OCP_n1,OCP_p1,OCV2,w1,w2);

x_0 = x_id(1);
QN = x_id(2);
y_0 = x_id(3);
QP = x_id(4);


% %plot
figure(3)
plot(OCV2(:,1),OCV2(:,2),'b-','LineWidth',lw,'MarkerSize',msz); hold on;
plot(OCV2(:,1),OCV_hat,'r-','LineWidth',lw,'MarkerSize',msz);
xlabel('SOC');
ylabel('OCV (V)');
title('SOC vs. OCV');


%dv/dq plot
x = OCV2(:,1);
y = OCV2(:,2);


x_values = [];
for i = 1:(length(x)-1)
    dvdq77(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
     x_values = [x_values; x(i)];
end


x = OCV2 (:,1);
y = OCV_hat (:,1);

x_values2 = [];
for i = 1:(length(x) - 1)
    dvdq88(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
    x_values2 = [x_values2; x(i)];
end


%dvdq에 이동 평균 적용
dvdq77_moving_avg = movmean(dvdq77(1:end), window_size);
x_values_moving_avg = movmean(x_values(1:end), window_size);

dvdq88_moving_avg = movmean(dvdq88(1:end), window_size);
x_values2_moving_avg = movmean(x_values2(1:end), window_size);


%dvdq plot
figure(4)
plot(x_values, dvdq77_moving_avg, 'b-', 'LineWidth', lw, 'MarkerSize', msz); hold on
plot(x_values2, dvdq88_moving_avg, 'r-', 'LineWidth', lw, 'MarkerSize', msz);

xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
%title('SOC vs. dV/dQ');
ylim([-1 3]);
xlim([0 1]);



% w1 = zeros(size(dvdq_sim_mov));
% greater_than_1_indices = find(dvdq_sim_mov <3);
% start_index = greater_than_1_indices(1); 
% end_index = greater_than_1_indices(end);

% w1(start_index:end_index) = dvdq_sim_mov(start_index:end_index); 
% plot(w1,'b-','LineWidth',lw,'MarkerSize',msz);

% % w(weighting) 생성
% w1 = dvdq_sim_mov;


figure(5)
plot(OCV2(:,1),w1,'-g','LineWidth',lw,'MarkerSize',msz);
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
xlim([0 1]);
ylim([0 5]);


% plot
figure('position', [0 0 500 400] );


% %%%%%%%%%%%첫 번째 그래프
subplot(2, 1, 1);

plot(OCV2(:,1),OCV2(:,2),'b-','LineWidth',lw,'MarkerSize',msz);
hold on;
plot(OCV2(:,1),OCV_hat,'r-','LineWidth',lw,'MarkerSize',msz);
xlabel('SOC');
ylabel('OCV (V)');
title('SOC vs. OCV');
yyaxis right;
ax = gca;  % 현재 축 객체 가져오기
ax.YColor = 'k';  % 검정색으로 설정
ylabel('Weight');
plot(OCV2(:,1),w2(1:end),'-g','LineWidth',lw,'MarkerSize',msz);
ylim([0 5]);
legend('FCC data','FCC fit','Weight','Location', 'none', 'Position', [0.2 0.85 0.1 0.05],'FontSize', 6);
xlim([0 1]);


%두 번째 그래프
subplot(2, 1, 2);
% subtightplot(2, 1, 2, [0.1 0.05], [0.05 0.1], [0.1 0.05]);
plot(x_values_moving_avg(1:end), dvdq77_moving_avg, 'b-', 'LineWidth', lw, 'MarkerSize', msz);
hold on;
plot(x_values2_moving_avg, dvdq88_moving_avg, 'r-', 'LineWidth', lw, 'MarkerSize', msz,'Color',[1,0,0,0.5]);
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
title('SOC vs. dV/dQ');
ylim([0 2]);
yyaxis right;
ax = gca;  % 현재 축 객체 가져오기
ax.YColor = 'k';  % 검정색으로 설정
ylabel('Weight');
plot(OCV2(:,1),w1,'-g','LineWidth',lw,'MarkerSize',msz,'Color',[0,1,0,0.3]);
ylim([0 2]);



% %OCP_n,OCP_p dvdq
x_1 = x_id(1,1) + (1/x_id(1,2));
y_1 = x_id(1,3) - (1/x_id(1,4));


OCP_n1(:,3) = ((OCP_n1(:,1)-x_id(1,1))/(x_1-x_id(1,1)));
OCP_p1(:,3) = ((OCP_p1(:,1)-x_id(1,3))/(y_1-x_id(1,3))); 


x = OCP_n1 (1:end,3);
y = OCP_n1 (1:end,2);


start_value = 0;
end_value = 1;


window_size = 50;


x_values = [];
for i = 1:(length(x) - 1)
    if x(i) >= start_value && x(i)<=end_value
    dvdq5(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));   
    x_values = [x_values; x(i)];
    end
end


x = OCP_p1 (1:end,3);
y = OCP_p1 (1:end,2);


x_values2 = [];
for i = 1:(length(x) - 1)
    if x(i) >= start_value && x(i)<=end_value
    dvdq6(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));   
%     if isnan(dvdq6)
%         % NaN 값의 앞쪽과 뒤쪽 데이터 포인트를 사용하여 내삽
%         dvdq6 = interp1(x([i-1, i+1]), y([i-1, i+1]), x(i));
%         % 내삽 결과를 저장
%         dvdq6(i) = dvdq6;
%     end
        x_values2 = [x_values2; x(i)];
    end
end


% dvdq5에 이동 평균 적용
idx_start = find(OCP_n1 (1:end,3) >= 0);
first_different_idx1= idx_start(1);
dvdq51 = movmean(dvdq5(first_different_idx1:end),window_size);
%x_values_moving_avg = movmean(x_values, window_size);


% dvdq6에 이동 평균 적용
idx_start = find(OCP_p1 (1:end,3) >= 0);
first_different_idx2 = idx_start(1);
dvdq61 = movmean(dvdq6(first_different_idx2:end),window_size);
%x_values2_moving_avg = movmean(x_values2, window_size);


% ocp_n,ocp_p플롯
figure(7)
plot(x_values, abs(dvdq51), 'b-', 'LineWidth', lw, 'MarkerSize', msz); hold on
plot(x_values2, dvdq61, 'r-', 'LineWidth', lw, 'MarkerSize', msz);
ylim([0 2]);
% % 가중치 적용 최적화


% 실제 관측값과 모델의 예측 값 사이의 차이

error = OCV_hat-OCV2(:,2);

% 제곱을 계산하여 평균
squared_error = error.^2;
mean_squared_error = mean(squared_error);

mean_squared_error = mean(squared_error);
root_squared_error = sqrt(error.^2);
% 평균값의 제곱근을 취하여 RMSE 값
rmse_value = sqrt(mean_squared_error);
disp(['RMSE 값: ', num2str(rmse_value)]);


figure(8)
plot(OCV2(:,1),error);
xlabel('SOC');
ylabel('Error(mV)');
xlim([0 1]);
xticks(0:0.1:1); 

 
% figure(10)
% plot(OCV2(1:end,1),w1(1:end),'-g','LineWidth',lw,'MarkerSize',msz);
% xlim([0 1]);
% ylim([0 5]);
% ylabel('Weight')
% xlabel('SOC')
% print('Weight','-dpng','-r300');



%peak값 추정
x_values = OCV2(:,1);     % x 값 배열
y_values = dvdq_mov(:,1); % y 값 배열
y_values2 = dvdq88_moving_avg(1,:); % y1 겂 배열 
dvdq88_moving_avg(end+1) = dvdq88_moving_avg(end);
% 첫 번째 x 범위
x_min = 0.2; % 최소 x 범위
x_max = 0.3; % 최대 x 범위

% 두 번째 x 범위
x_min2 = 0.3; % 다른 최소 x 범위
x_max2 = 0.4; % 다른 최대 x 범위

%세 번째 x 범위
x_min3 = 0.7; % 다른 최소 x 범위
x_max3 = 0.8; % 다른 최대 x 범위


% 첫 번째 x 범위 내에서 가장 큰 y 값 찾기
x_in_range = x_values(x_values >= x_min & x_values <= x_max); % x 범위에 해당하는 x 값 추출
y_in_range = y_values(x_values >= x_min & x_values <= x_max); % x 범위에 해당하는 y 값 추출
y_in_range2 = y_values2(x_values >= x_min & x_values <= x_max); 
[max_y1, max_index1] = max(y_in_range); % 첫 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
[max2_y1, max2_index1] = max(y_in_range2); % 첫 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 찾기
% 첫 번째 x 범위 내에서 최소 y 값 및 해당 인덱스 
corresponding_x1 = x_in_range(max_index1); % 해당 y 값에 대응하는 x 값 찾기
corresponding_x11 = x_in_range(max_index1); % 해당 y2 값에 대응하는 x 값 찾기
index1 = find(OCV2(:,1) == corresponding_x1);
index11 = find(OCV2(:,1) == corresponding_x11);
sim_max1 = y_values2(max_index1);


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
sim_max2 = y_values2(max_index2);


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
sim_max3 = y_values2(max_index3);


e1 =sqrt((max_y1-sim_max1).^2);
e2 =sqrt((max_y2-sim_max2).^2);
e3 =sqrt((max_y3-sim_max3).^2);
e4 =sqrt((corresponding_x1-corresponding_x11).^2);
e5 =sqrt((corresponding_x2-corresponding_x22).^2);
e6 =sqrt((corresponding_x3-corresponding_x33).^2);


%peak값의 average
error_dvdq = sqrt((max_y1-sim_max1).^2+(max_y2-sim_max2).^2+(max_y3-sim_max3).^2);
error_soc = sqrt((corresponding_x1-corresponding_x11).^2+(corresponding_x2-corresponding_x22).^2+(corresponding_x3-corresponding_x33).^2);
disp(['error dvdq값: ', num2str(error_dvdq)]);
disp(['error soc값: ', num2str(error_soc)])
eroor_disp = ['e1: ', num2str(e1), ', e2: ', num2str(e2), ', e3: ', num2str(e3), ', e4: ', num2str(e4), ', e5: ', num2str(e5), ', e6: ', num2str(e6)];
disp(eroor_disp);

% 결과 출력
% fprintf('첫 번째 x 범위 [%.2f, %.2f] 내에서 가장 큰 y 값: %.2f\n', x_min, x_max, max_y1);
% fprintf('해당 y 값에 대응하는 x 값: %.2f\n', corresponding_x1);
% 
% fprintf('두 번째 x 범위 [%.2f, %.2f] 내에서 가장 작은 y 값: %.2f\n', x_min2, x_max2, max_y2);
% fprintf('해당 y 값에 대응하는 x 값: %.2f\n', corresponding_x2);
% 
% fprintf('세 번째 x 범위 [%.2f, %.2f] 내에서 가장 작은 y 값: %.2f\n', x_min3, x_max3, max_y3);
% fprintf('해당 y 값에 대응하는 x 값: %.2f\n', corresponding_x3);


   
function [cost,OCV_sim] = OCV_dvdq_model_06(x, OCP_n1, OCP_p1, OCV2, w1, w2)
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

    window_size = 50;

    x_values = OCV2(:, 1);
    y_values = OCV2(:, 2);
    y_sim_values = OCV_sim(:, 1);

    dvdq = diff(y_values) ./ diff(x_values);
    dvdq_sim = diff(y_sim_values) ./ diff(x_values);
    dvdq = [dvdq; dvdq(end)];
    dvdq_mov = movmean(dvdq, window_size);

    dvdq_sim = [dvdq_sim; dvdq_sim(end)];
    dvdq_sim_mov = movmean(dvdq,window_size);

    OCV_sim_mov =  movmean(OCV_sim,window_size);

    OCV2_mov = movmean(OCV2,window_size);

    cost_dvdq = sum(((dvdq_sim_mov - dvdq_mov).^2/mean(dvdq_mov)).*w1');

    % OCV 비용 계산
    cost_OCV = sum((OCV_sim_mov - OCV2_mov(:,2)).^2./mean(OCV2_mov(:,2)).*w2');
   
    % 비용 합산 
    cost = sum(cost_dvdq + cost_OCV);
    
end
