
% 가중치 미적용 최적화

% data load
clc;clear;close all;
load ('OCV_fit.mat')
load('ocv1w.mat')
% 초기 추정값 
x0_2nd_opt = [0.01,1*1.2,0.9,1];

%  최적화 수행
[~,OCV_guess] =  OCV_dvdq_model_06(x0_2nd_opt,OCP_n,OCP_p,OCV);


% fmincon을 사용하여 최적화 수행
  
options = optimoptions(@fmincon,'MaxIterations',5000,'StepTolerance',1e-15,'ConstraintTolerance', 1e-15, 'OptimalityTolerance', 1e-15);
   
problem = createOptimProblem('fmincon', 'objective', @(x) OCV_dvdq_model_06(x,OCP_n,OCP_p,OCV), ...
            'x0', x0_2nd_opt, 'lb', [0,1*1.2,0.84,1*1.5], 'ub', [0.3,1*1.5,0.86,1*1.7] , 'options', options);
        ms = MultiStart('Display', 'iter');
    
        [x_id, fval, exitflag, output] = run(ms, problem, 100); 
 

[cost_hat,  OCV_hat] = OCV_dvdq_model_06(x_id,OCP_n,OCP_p,OCV);
 

% 라인,폰트 설정
width = 6;     % Width in inches
height = 6;    % Height in inches
alw = 2;    % AxesLineWidth
fsz = 20;      % Fontsize
lw = 2;      % LineWidth
msz = 16;       % MarkerSize


pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %Set properties


%plot
figure(1)
plot(OCV(:,1),OCV(:,2),'b-','LineWidth',lw,'MarkerSize',msz); hold on
plot(OCV(:,1),OCV_hat,'r-','LineWidth',lw,'MarkerSize',msz);


%dv/dq plot
figure(2) 
window_size = 50;

x = OCV (:,1);
y = OCV (:,2);


x_values = [];
for i = 1:(length(x)-1)
    dvdq1(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
    x_values = [x_values; x(i)];
end
 dvdq1(end+1) = dvdq1(end);
 x_values(end+1) = x_values(end);

 
x = OCV (:,1);
y = OCV_hat (:,1);


x_values2 = [];
for i = 1:(length(x) - 1)
    dvdq2(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i)); 
    x_values2 = [x_values2; x(i)];
end
 dvdq2(end+1) = dvdq2(end);
 x_values2(end+1) =  x_values2(end);


% dvdq 이동 평균 적용
dvdq1_moving_avg = movmean(dvdq1(1:end), window_size);
x_values_moving_avg = movmean(x_values, window_size);


dvdq2_moving_avg = movmean(dvdq2(1:end), window_size);
x_values2_moving_avg = movmean(x_values2, window_size);


% % dvdq 플롯 그리기
plot(x_values, dvdq1_moving_avg, 'b-', 'LineWidth', lw, 'MarkerSize', msz); hold on
plot(x_values2, dvdq2_moving_avg, 'r-', 'LineWidth', lw, 'MarkerSize', msz);
title('Moving avg');


%w(weighting) 생성
figure(3)
w = ones(size(dvdq1_moving_avg(1,:)));
greater_than_1_indices = find(dvdq1_moving_avg <2);
% greater_than_2_indices = find(OCV(:,1) > 0.65 & OCV(:,1) < 0.8);

greater_than_1_values = dvdq1_moving_avg(1,greater_than_1_indices);
% greater_than_2_values = OCV(greater_than_2_indices ,1);


start_index = greater_than_1_indices(1); 
end_index = greater_than_1_indices(end);


w(start_index:end_index) = dvdq1_moving_avg(start_index:end_index)+1; 


% greater_than_1_indices = find(OCV(:,1) > 0.1 & OCV(:,1) < 0.9);
% greater_than_1_values = OCV(greater_than_1_indices ,1);
% w = ones(size(OCV(:,1)));
% start_index = greater_than_1_indices(1,1); 
% end_index = greater_than_1_indices(end,1);
% w(start_index:end_index) = dvdq1_moving_avg(start_index:end_index); 

w1 ='dvdq1_moving_avg';
plot(w,'b-','LineWidth',lw,'MarkerSize',msz); hold on
save('ocv1w.mat','w','x_id','w1');



% plot
figure('position', [0 0 500 400] );
figure('Name','w적용전')


% 첫 번째 그래프
subplot(2, 1, 1);

plot(OCV(:,1),OCV(:,2),'b-','LineWidth',lw,'MarkerSize',msz);
hold on;
plot(OCV(:,1),OCV_hat,'r-','LineWidth',lw,'MarkerSize',msz);
ylabel('OCV (V)');
xlim([0 1]); 
legend('FCC data','FCC fit','Location', 'none', 'Position', [0.2 0.85 0.1 0.05],'FontSize', 6);


% 두 번째 그래프
subplot(2, 1, 2);

% subtightplot(2, 1, 2, [0.1 0.05], [0.05 0.1], [0.1 0.05]);
plot(x_values, dvdq1_moving_avg, 'b-', 'LineWidth', lw, 'MarkerSize', msz);
hold on;
plot(x_values2, dvdq2_moving_avg, 'r-', 'LineWidth', lw, 'MarkerSize', msz, 'color',[1, 0, 0, 0.6]);
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
title('SOC vs. dV/dQ');
ylim([0 2]);
print('OCV ','-dpng','-r300');





% 가중치 적용 최적화

% 두 번째 최적화를 위한 초기 추정값 (이전에 구한 x_id 값을 사용)
x_guess = x_id;

% 두 번째 최적화 수행
[~,OCV_guess] =  OCV_dvdq_model_06(x_guess,OCP_n,OCP_p,OCV,w);


% fmincon을 사용하여 최적화 수행
  
options = optimoptions(@fmincon,'MaxIterations',5000,'StepTolerance',1e-15,'ConstraintTolerance', 1e-15, 'OptimalityTolerance', 1e-15);
   
problem = createOptimProblem('fmincon', 'objective', @(x) OCV_dvdq_model_06(x,OCP_n,OCP_p,OCV,w), ...
            'x0', x_guess, 'lb', [0.0025,1*1.22,0.84,1*1.5], 'ub', [0.0029,1*1.248,0.85,1*1.63], 'options', options);
        ms = MultiStart('Display', 'iter');
    
        [x_id, fval, exitflag, output] = run(ms, problem, 100); 
 

[cost_hat,OCV_hat] = OCV_dvdq_model_06(x_id,OCP_n,OCP_p,OCV,w);
 

% plot
figure('Name','w적용후')
plot(OCV(:,1),OCV(:,2),'b-','LineWidth',lw,'MarkerSize',msz); hold on;
plot(OCV(:,1),OCV_hat,'r-','LineWidth',lw,'MarkerSize',msz);


xlabel('SOC');
ylabel('OCV (V)');
title('SOC vs. OCV (0.01C)');


yyaxis right;
ax = gca;  % 현재 축 객체 가져오기
ax.YColor = 'k';  % 검정색으로 설정
ylabel('Weight')
plot(OCV(1:end,1),w(1:end),'-g','LineWidth',lw,'MarkerSize',msz);
ylim([0 20])
legend('FCC data','FCC fit','Weight');


% dv/dq plot
figure
start_value = 0;
end_value = 1;

window_size = 50;


x = OCV(:,1);
y = OCV(:,2);


x_values = [];
for i = 1:(length(x)-1)
    dvdq77(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
    x_values = [x_values; x(i)];
end
dvdq77(end+1) = dvdq77(end);
x_values(end+1) = x_values(end);   


x = OCV (:,1);
y = OCV_hat (:,1);

x_values2 = [];
for i = 1:(length(x) - 1)
    dvdq88(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
    x_values2 = [x_values2; x(i)];   
end
dvdq88(end+1) = dvdq88(end);
x_values2(end+1) =  x_values2(end);


% dvdq에 이동 평균 적용
dvdq77_moving_avg = movmean(dvdq77(1:end), window_size);
x_values_moving_avg = movmean(x_values, window_size);

dvdq88_moving_avg = movmean(dvdq88(1:end), window_size);
x_values2_moving_avg = movmean(x_values2, window_size);

plot(x_values, dvdq77_moving_avg, 'b-', 'LineWidth', lw, 'MarkerSize', msz); hold on
plot(x_values2, dvdq88_moving_avg, 'r-', 'LineWidth', lw, 'MarkerSize', msz);


%dvdq plot
figure('Name','w생성')
w = ones(size(dvdq77_moving_avg(1,:)));
greater_than_1_indices = find(dvdq77_moving_avg <2);
% greater_than_2_indices = find(OCV(:,1) > 0.65 & OCV(:,1) < 0.8);

greater_than_1_values = dvdq77_moving_avg(1,greater_than_1_indices);
% greater_than_2_values = OCV(greater_than_2_indices ,1);


start_index = greater_than_1_indices(1); 
end_index = greater_than_1_indices(end);
% start_index2 = greater_than_2_indices(1,1);
% end_index2 = greater_than_2_indices(end,1);

w(start_index:end_index) = dvdq77_moving_avg(start_index:end_index)+1; 
plot(w,'b-','LineWidth',lw,'MarkerSize',msz);


% legend('FCC data','FCC fit')
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
title('SOC vs. dV/dQ');
ylim([-1 3]);


print('OCV fig4','-dpng','-r300');



% plot

figure('position', [0 0 500 400] );
% 첫 번째 그래프
subplot(2, 1, 1);

plot(OCV(:,1),OCV(:,2),'b-','LineWidth',lw,'MarkerSize',msz);
hold on;
plot(OCV(:,1),OCV_hat,'r-','LineWidth',lw,'MarkerSize',msz);
% xlabel('SOC');
ylabel('OCV (V)');
% title('OCV1 (0.01C)');
yyaxis right;
ax = gca;  % 현재 축 객체 가져오기
ax.YColor = 'k';  % 검정색으로 설정
ylabel('Weight')
plot(OCV(1:end,1),w(1:end),'-g','LineWidth',lw,'MarkerSize',msz);
ylim([0 20])

legend('FCC data','FCC fit','Weight','Location', 'none', 'Position', [0.2 0.85 0.1 0.05],'FontSize', 6);
xlim([0 1]);

% 두 번째 그래프
subplot(2, 1, 2);
% subtightplot(2, 1, 2, [0.1 0.05], [0.05 0.1], [0.1 0.05]);
plot(x_values, dvdq77_moving_avg, 'b-', 'LineWidth', lw, 'MarkerSize', msz);
hold on;
plot(x_values2, dvdq88_moving_avg, 'r-', 'LineWidth', lw, 'MarkerSize', msz,'Color',[1,0,0,0.5]);
xlabel('SOC');
ylabel('dV/dQ /  V (mAh)^-1');
% title('SOC vs. dV/dQ');
ylim([0 2]);
print('OCV fig9','-dpng','-r300');




%OCP_n,OCP_p dvdq
x_1 = x_id(1,1) + (1/x_id(1,2));
y_1 = x_id(1,3) - (1/x_id(1,4));


OCP_n(:,3) = ((OCP_n(:,1)-x_id(1,1))/(x_1-x_id(1,1)));
OCP_p(:,3) = ((OCP_p(:,1)-x_id(1,3))/(y_1-x_id(1,3))); 


x = OCP_n (1:end,3);
y = OCP_n (1:end,2);

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

x_values(end+1) = x_values(end);
dvdq5(end+1) = dvdq5(end);



x = OCP_p (1:end,3);
y = OCP_p (1:end,2);


x_values2 = [];
for i = 1:(length(x) - 1)
    if x(i) >= start_value && x(i)<=end_value
    dvdq6(i) = (y(i + 1) - y(i)) / (x(i + 1) - x(i));   
    if isnan(dvdq6)
        % NaN 값의 앞쪽과 뒤쪽 데이터 포인트를 사용하여 내삽
        dvdq6 = interp1(x([i-1, i+1]), y([i-1, i+1]), x(i));
        % 내삽 결과를 저장
        dvdq6(i) = dvdq6;
    end
        x_values2 = [x_values2; x(i)];
    end
end

x_values2(end+1) = x_values2(end); 
dvdq6(end+1) = dvdq6(end);



% dvdq5에 이동 평균 적용
idx_start = find(OCP_n (1:end,3) >= 0);
first_different_idx1= idx_start(1);
dvdq51 = movmean(dvdq5(first_different_idx1:end),window_size);
%x_values_moving_avg = movmean(x_values, window_size);


% dvdq6에 이동 평균 적용
idx_start = find(OCP_p (1:end,3) >= 0);
first_different_idx2 = idx_start(1);
dvdq61 = movmean(dvdq6(first_different_idx2:end),window_size);
%x_values2_moving_avg = movmean(x_values2, window_size);


% ocp_n,ocp_p플롯
plot(x_values, abs(dvdq51), 'b-', 'LineWidth', lw, 'MarkerSize', msz); hold on
plot(x_values2 , dvdq61, 'r-', 'LineWidth', lw, 'MarkerSize', msz);






% 실제 관측값과 모델의 예측 값 사이의 차이
error =  OCV_hat-OCV3(:,2);

% 제곱을 계산하여 평균
squared_error = error.^2;
mean_squared_error = mean(squared_error);
root_squared_error = sqrt(error.^2);

% 평균값의 제곱근을 취하여 RMSE 값
rmse_value = sqrt(mean_squared_error);
disp(['RMSE 값: ', num2str(rmse_value)]);

figure(8)
plot(OCV3(:,1),error);
xlabel('SOC');
ylabel('squred error')



error =  OCV_hat-OCV(:,2);

% 제곱을 계산하여 평균
squared_error = error.^2;
mean_squared_error = mean(squared_error);

mean_squared_error = mean(squared_error);
root_squared_error = sqrt(error.^2);
% 평균값의 제곱근을 취하여 RMSE 값
rmse_value = sqrt(mean_squared_error);
disp(['RMSE 값: ', num2str(rmse_value)]);


figure(9)
plot(OCV(:,1),error);
xlabel('SOC');
ylabel('Error(mV)')
print('Error','-dpng','-r300');

