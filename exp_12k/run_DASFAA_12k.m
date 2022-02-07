%

clear;
clc;
data_path = fullfile(pwd, '..',  filesep, "data", filesep);
addpath(data_path);
dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};
datasetCandi = {'CSTR_476n_1000d_4c_tfidf_uni.mat'};

exp_n = 'TRMKC-DASFAA';
addpath(fullfile([pwd, '\..\lib']));
addpath(fullfile([pwd, '\..\TRMKC-DASFAA']));
for i1 = 1: length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    dir_name = [pwd, filesep, exp_n, filesep, data_name];
    try
        if ~exist(dir_name, 'dir')
            mkdir(dir_name);
        end
        prefix_mdcs = dir_name;
    catch
        disp(['create dir: ',dir_name, 'failed, check the authorization']);
    end
    clear X Y y;
    load(data_name);
    if exist('y', 'var')
        Y = y;
    end
    if size(X, 1) ~= size(Y, 1)
        Y = Y';
    end
    assert(size(X, 1) == size(Y, 1));
    %X = my_preprocessing(X);
    nSmp = size(X, 1);
    nCluster = length(unique(Y));
    
    fname2 = fullfile(prefix_mdcs, [data_name, '_12kAllFea_TRMKC.mat']);
    if ~exist(fname2, 'file') 
        Xs = cell(1,1);
        Xs{1} = X;
        Ks = Xs_to_Ks_12k(Xs); % use knorm normalization
        Ks2 = Ks{1,1};
        Ks = Ks2;
        clear Ks2 Xs;
        nKernel = size(Ks, 3);

        
        %*********************************************************************
        % OurMethod-2021-du-2021
        %*********************************************************************
        tic;
        w_type = 'WCIM';
        is_adaptive_delta = false;
        [label, K, H, w_WCIM, objHistory_WCIM, objHistory2_WCIM] = TRMKC_DASFAA(Ks, nCluster, w_type,is_adaptive_delta);
        t2 = toc;
        result_10 = my_eval_y(label, Y);
        result_WCIM = [result_10(:)', t2];
        

        r_aio = [result_WCIM];
        save(fname2, 'result_WCIM', 'w_WCIM', 'objHistory_WCIM', 'objHistory2_WCIM');
        disp([data_name(1:end-4), ' has been completed!']);
    end
end
rmpath(fullfile([pwd, '\..\TRMKC-DASFAA']));
rmpath(fullfile([pwd, '\..\lib']));