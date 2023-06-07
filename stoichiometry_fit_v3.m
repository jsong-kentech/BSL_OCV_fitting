%% Fit QP, QN and x0, y0 (0% DOD stoichiometry) with the weight matrix w

function cost = stoichiometry_fit_v3(x, OCP_n, OCP_p, OCV, w)
    
    x_0 = x(1);
    QN = x(2);
    y_0 = x(3);
    QP = x(4);
    
    Cap = OCV(:,1);
    
    if (OCV(end,2)<OCV(1,2)) % Discharge OCV
        x_sto = -(Cap - Cap(1))/QN + x_0;
        y_sto = (Cap - Cap(1))/QP + y_0;
    else  % Charge OCV
        x_sto = (Cap - Cap(1))/QN + x_0;
        y_sto = -(Cap - Cap(1))/QP + y_0;
    end
    
    OCP_n_sim = interp1(OCP_n(:,1), OCP_n(:,2), x_sto, 'linear','extrap');
    OCP_p_sim = interp1(OCP_p(:,1), OCP_p(:,2), y_sto, 'linear','extrap');
    
    OCV_sim = OCP_p_sim - OCP_n_sim;
    
    cost = sum((OCV_sim - OCV(:,2)).^2.*w')*1e3;
%     cost = sum((OCV_sim - OCV(:,2)).^2);
end