function [label, K, H, w, objHistory, objHistory2] = our_method(Ks, nCluster, w_type, is_adaptive_delta)
[nSmp, ~, nKernel] = size(Ks);
if ~exist('is_adaptive_delta', 'var')
    is_adaptive_delta = false;
end
% assert(min(Ks(:)) >=0);
%*************************************************
% Step 1: Multiple base kernel refinement
%*************************************************
% Extract neighbor kernel from original kernel as page 3 in [1]
% [1] Multiple Kernel Clustering With Neighbor-Kernel Subspace
% Segmentation, TNNLS 2020.
if ~exist('nn', 'var')
    nn = round(5 * log10(nSmp));
end
Ks_n_n = zeros(nSmp, nSmp, nKernel);
for iKernel = 1:nKernel
    Ki = Ks(:, :, iKernel);
    A_Ki = genarateNeighborhood(Ki, nn);
    Ki_n = KernelGeneration(Ki, A_Ki);
%     Ks_n_n(:, :, iKernel) = Ki_n;
    Di = 1./sqrt(max(sum(Ki_n, 2), eps));% 
    Ki_n_n = (Di * Di') .* Ki_n;
    Ki_n_n = (Ki_n_n + Ki_n_n')/2;
    Ks_n_n(:, :, iKernel) = Ki_n_n;
end
% clear Ks Ki A_Ki Ki_n Di Ki_n_n;

%*************************************************
% Step 2: Consensus kernel learning via trace ratio maximization
%*************************************************
[K, H, w, objHistory, objHistory2] = OKL_tr_hq(Ks_n_n, nCluster, w_type, is_adaptive_delta);

%*************************************************
% Step 3: Extract label from subspace 
%*************************************************
H = bsxfun(@rdivide, H, sqrt(max( sum(H.^2,2), eps)));  %
label = kmeans(H, nCluster,  'replicates', 10, 'display', 'off');
end