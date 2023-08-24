function [cost,OCV_sim] = OCV_dvdq_model_06(x, OCP_n, OCP_p,OCV,w)
    % OCV_hat 계산
    [~, OCV_sim] = OCV_stoichiometry_model_06(x, OCP_n, OCP_p, OCV,w);
    
    % 비용 계산
    cost_OCV = sum((OCV_sim - OCV(:,2)).^2./mean(OCV(:,2)).*w');
    
    % dV/dQ 값들 계산
    x_values = OCV(:,1);
    y_values = OCV(:,2);
    y_sim_values = OCV_sim(:,1);
    dvdq = diff(y_values) ./ diff(x_values);
    dvdq_sim = diff(y_sim_values) ./ diff(x_values);
    dvdq = [dvdq; dvdq(end)];
    dvdq_sim = [dvdq_sim; dvdq_sim(end)];

    % dv/dq를 이용한 비용 계산
    cost_dvdq = sum((dvdq_sim - dvdq).^2/mean(dvdq).*w');

    % 비용 합산 
    cost = cost_OCV + cost_dvdq;
end
