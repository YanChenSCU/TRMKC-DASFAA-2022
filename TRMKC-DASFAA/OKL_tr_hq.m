function [K, H, w, objHistory, objHistory2] = OKL_tr_hq(Ks, nCluster, w_type, is_adaptive_delta)
global t0 delta2_0;

if ~exist('w_type', 'var')
    w_type = 'WCIM';
end

if ~exist('is_adaptive_delta', 'var')
    is_adaptive_delta = false;
end
[nSmp, ~, nKernel] = size(Ks);

%****************************
% init w
%****************************
switch w_type
    case 'WCIM'
        s = ones(nKernel, 1)/nKernel;
        w = 1./s;
end

%****************************
% init K
%****************************
K = zeros(nSmp);
for iKernel = 1:nKernel
    K = K + w(iKernel) * Ks(:, :, iKernel);
end
e = zeros(nKernel, 1);
for iKernel = 1:nKernel
    E = K -  Ks(:, :, iKernel);
    e(iKernel) = sum(sum(E.^2));
end
t0=1;
delta2_0 = t0 * median(e);

%****************************
% init H
%****************************
%H = ncut(K, nCluster);
[eigvec, eigval] = eigs(K, nCluster, 'La');
H = eigvec;

converges_1 = false;
myeps = 1e-6;
max_iter1 = 20;
iter1 = 0;
objHistory = [];
while ~converges_1
    
    %**************************
    % update K, given w and H
    %**************************
    converges_2 = false;
    objHistory2 = [];
    iter2 = 0;
    max_iter2 = 1;
    while ~converges_2
        HHT = H * H';
        f1 = sum(sum(K .* HHT));
        switch w_type
            case 'WCIM'
                f2 = compute_f2_cim2wls(Ks, w, K, is_adaptive_delta);
        end
        
        lambda_2 = f1/f2;
        
        objHistory2 = [objHistory2 ; lambda_2]; %#ok
        switch w_type
          case 'WCIM'
                e = zeros(nKernel, 1);
                for iKernel = 1:nKernel
                    E = K -  Ks(:, :, iKernel);
                    e(iKernel) = sum(sum(E.^2));
                end
                if is_adaptive_delta
                    delta2 = t0 * median(e);
                else
                    delta2 = delta2_0;
                end
                g = exp(-e/delta2);
                g = g/delta2_0;
                w2 = g./s;
                K = update_K_psd(Ks, w2, HHT, lambda_2);
        end

        iter2 = iter2 + 1;
        if (  iter2 >= max_iter2 || ((iter2 > 0) && abs( objHistory2(end) - objHistory2(end-1))/abs(objHistory2(end)) < myeps))
            converges_2 = true;
        end
    end
    
    %**************************
    % update K, given w
    %**************************
    
    switch w_type
        case 'WCIM'
            e = zeros(nKernel, 1);
            for iKernel = 1:nKernel
                E = K -  Ks(:, :, iKernel);
                e(iKernel) = sum(sum(E.^2));
            end
            if is_adaptive_delta
                delta2 = t0 * median(e);
            else
                delta2 = delta2_0;
            end
            q = 1 - exp(-e/delta2);
            s = sqrt(q);
            s = s/sum(s);
            w = 1./s;
    end
    
    %**************************
    % update K, given H
    %**************************
    [eigvec, eigval] = eigs(K, nCluster, 'La');
    H = eigvec;
   
    f1 = trace(H' * K * H);
    switch w_type
        case 'WCIM'
            f2 = compute_f2_cim(Ks, w, K, is_adaptive_delta);
    end
    lambda = f1/f2;
    
    iter1 = iter1 + 1;
    objHistory = [objHistory; lambda]; %#ok
    
    
    if ((iter1 > 5) && abs(objHistory(end) - objHistory(end-1))/abs(objHistory(end)) < myeps) || iter1 > max_iter1
        converges_1 = true;
    end
    
end
end

function K = update_K_psd(Ks, w, HHT, lambda_2)
[nSmp,~,nKernel] = size(Ks);
B1 = zeros(nSmp);
for iKernel = 1:nKernel
    B1 = B1 + (w(iKernel) / sum(w)) * Ks(:, :, iKernel);
end
B2 = (1/(2 * lambda_2 * sum(w))) * HHT;

B = B1 + B2;
[V, D] = eig(B);
diagD = diag(D);
diagD(diagD<eps)=0;
K = V*diag(diagD)*V';
K = (K+K')/2;
end



function f2 = compute_f2_cim(Ks, w, K, is_adaptive_delta)
global t0 delta2_0
% t0 = 1;
[~,~,nKernel] = size(Ks);
e = zeros(nKernel, 1);
for iKernel = 1:nKernel
    E = K -  Ks(:, :, iKernel);
    e(iKernel) = sum(sum(E.^2));
end
if is_adaptive_delta
    delta2 = t0 * median(e);
else
    delta2 = delta2_0;
end
loss = 1 - exp(-e/delta2);
f2 = sum(w.*loss);
end

function f2 = compute_f2_cim2wls(Ks, w, K, is_adaptive_delta)
global t0 delta2_0
% t0 = 1;
[~,~,nKernel] = size(Ks);
e = zeros(nKernel, 1);
for iKernel = 1:nKernel
    E = K -  Ks(:, :, iKernel);
    e(iKernel) = sum(sum(E.^2));
end
if is_adaptive_delta
    delta2 = t0 * median(e);
else
    delta2 = delta2_0;
end
g = exp(-e/delta2);
wg = g.*w;
f2 = sum(wg.*e);
f2 = f2 / delta2_0;
end