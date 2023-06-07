clear;clc;close all

%% INPUTS
    % half-cell OCP data file name 
filename_OCP = 'X1288_Var_OCP_processed.mat';

    % full-cell OCV data to be examined
filename_OCV = 'C20_fresh.mat';
    
    
%% LOAD DATA
    % loading file names
load(filename_OCP) % cathode and anode ocp curves vs filling fraction.
load(filename_OCV) % OCV curves to be fitted

    % half_cell ocp (!! could be improved by averaging the three)
OCP_n = OCP.Un_all{1};
OCP_p = OCP.Up_all{1};

    % variable name: "dataList"
datalist = dataList;

%% PLOT
onoff_seedata = 1;

if onoff_seedata == 1

for i=1:size(OCP_n,2)
    figure(1); hold on; box on
    plot(OCP_n(:,1),OCP_n(:,2))
    figure(2); hold on; box on
    plot(OCP_p(:,1),OCP_p(:,2)) 
end
figure(1)
title('X1444 Anode OCP')
xlabel('x in LixC6')
ylabel('OCP [V]')
figure(2)
title('X1444 Cathode OCP')
xlabel('y in LixMO2')
ylabel('OCP [V]')


for i=1:size(datalist,1)
    Cap = datalist.C20d{i}(1:10:end,1); % [Ah] Discharged capacity
    Cap_end = Cap(end);
    Q_cell = Cap_end;
    OCV = datalist.C20d{i}(1:10:end,2); % [V] Cell Voltage     
    measurement = [Cap,OCV]; % OCV measurement matrix [q [Ah], v[V]]
    figure(3); hold on; box on
    plot(Cap,OCV)  
end
figure(3)
title('X1444 Full Cell OCV')
xlabel('Cap [Ah]')
ylabel('OCP [V]')


end

%% FITTING  FOR A SINGLE MEASUREMENT

%!! ADD: another for loop for different samples.

    % select a cell
for k_list=1:size(dataList,1)
    
    % reduce the data
Cap = datalist.C20d{k_list}(1:10:end,1); % [Ah] Discharged capacity
Cap_end = Cap(end);
Q_cell = abs(Cap_end);
OCV = datalist.C20d{k_list}(1:10:end,2); % [V] Cell Voltage     
measurement = [Cap,OCV]; % OCV measurement matrix [q [Ah], v[V]]

    % define the weighting 
w = zeros(size(Cap')); % should have the same length with the data
w(:)=1; % uniform weighting

    % prep: multi-start (tolerances)
ms = MultiStart('UseParallel',true,'FunctionTolerance',1e-15,'XTolerance',1e-15);
    
    % prep: optimset option (tolerances)
options = optimoptions(@fmincon,'MaxIterations',5000,'StepTolerance',1e-15,'ConstraintTolerance', 1e-15, 'OptimalityTolerance', 1e-15);
    
    % initial guess and lower/upper bounds
x_guess = [0,Q_cell,1,Q_cell];
x_lb = [0,Q_cell*0.5,0,Q_cell*0.5];
x_ub = [1,Q_cell*2,1,Q_cell*2]; 

    % optimization problem
        % obj = min(sum(OCV-OCV_model)^2 ; @stoichiometry_fit_vX.m
problem = createOptimProblem('fmincon','x0',x_guess,'objective',@(x) stoichiometry_fit_v3(x, OCP_n, OCP_p, measurement, w),'lb',x_lb,'ub',x_ub,'options',options);
[x_id,fval_ms,flag,outpt,allmins] = run(ms,problem,100);

    % allocate the results
x_0 = x_id(1);
QN = x_id(2);
y_0 = x_id(3);
QP = x_id(4);

if (OCV(end)<OCV(1)) % Discharge OCV
    x_sto = -(Cap - Cap(1))/QN + x_0;
    y_sto = (Cap - Cap(1))/QP + y_0;
else  % Charge OCV
    x_sto = (Cap - Cap(1))/QN + x_0;
    y_sto = -(Cap - Cap(1))/QP + y_0;
end

x_100 = x_sto(end);
y_100 = y_sto(end);

OCP_n_sim = interp1(OCP_n(:,1), OCP_n(:,2), x_sto, 'linear','extrap');
OCP_p_sim = interp1(OCP_p(:,1), OCP_p(:,2), y_sto, 'linear','extrap');

OCV_sim = OCP_p_sim - OCP_n_sim;
TotalLi = QN*x_0 + QP*y_0;

dataList.QN(k_list)=QN;
dataList.QP(k_list)=QP;
dataList.QLi(k_list)=TotalLi;
dataList.x0(k_list)=x_0;
dataList.x100(k_list)=x_100;
dataList.y0(k_list)=y_0;
dataList.y100(k_list)=y_100;
dataList.OCVerr(k_list)=norm(OCV_sim - OCV);

end


%% FIGURE SET 2
    CO = lines(8);
    figure(103);
    clf
    plot(Cap, OCV, 'o');
    hold on;
    plot(Cap, OCV_sim,'linewidth',2);
    grid on; box on;
    xlabel('Capacity (Ah)');ylabel('OCV (V)');
    legend('Experiment', 'Model');
    set(gca,'fontsize',16);
        title([strrep(dataList.testName{k_list},'_','\_'),  ' ', num2str(round(dataList.days(k_list))), ' days', ' C/150 OCP fitting'])
        
    figure(105);
    clf
    yyaxis left;
    plot(Cap, OCP_n_sim, 'linewidth',2);
    ylabel('Anode OCP (V)');
    hold on;
    yyaxis right;
    plot(Cap, OCP_p_sim,'linewidth',2);
    grid on; box on;
    xlabel('Capacity (Ah)');ylabel('Cathode OCP (V)');
    set(gca,'fontsize',20);

    figure(104);
    clf
    plot(Cap, OCV_sim - OCV);
    grid on; box on;
    xlabel('Capacity (Ah)');ylabel('OCV error (V)');
    legend('Simulation - Experiment');
    set(gca,'fontsize',20);
    
    


%% FIGURE SET 3






