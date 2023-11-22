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
