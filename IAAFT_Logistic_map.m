%% ExSEnt on Logistic Map with IAAFT surrogate testing — PARALLEL VERSION
%
% Requires:
%   compute_Sampentropies(x, lambda, m, alpha)
%     -> [hd, ha, hda, ~, ~, ~, ~, ~]  (8 outputs)

clear; clc; close all;

%% ========================= SETTINGS =====================================

% ExSEnt parameters
m      = 2;
alpha  = 0.2;
lambda = 0.01;

% Surrogate settings
Nsurr         = 1000;
IAAFT_maxIter = 100;
IAAFT_tol     = 1e-4;
MIN_VALID_SURR = 100;
MIN_EXTREMA    = 1000;

prefix = 'Logistic_IAAFT_ExSEnt';

% Logistic map simulation settings
discard_n = 500;
keep_n    = 9500;

% Fixed r values
r_vec = [3.600, 3.970];% Selected values based on Fig. 3 in the ms. Change to check other regimes
nR    = numel(r_vec);

x0 = 0.5;%initial state the same a Fig. 3

metricNames = {'HD','HA','HDA'};
nMet        = numel(metricNames);

rng(1);

% --- Resilience / progress settings -------------------------------------
BATCH_SIZE  = 100;%10;    % restart pool every N surrogates; lower = safer
PRINT_EVERY = 100;%10;    % print progress line every N surrogates
MAX_WORKERS = 10;%5;     % cap workers to avoid RAM saturation

%% ========================= SIGNAL GENERATION ============================

fprintf('\n===== SIGNAL GENERATION =====\n');

scan_cache = cell(nR,1);
extCount   = zeros(nR,1);

for iR = 1:nR
    r_val = r_vec(iR);
    xk    = logistic_map(x0, r_val, discard_n, keep_n);
    sx    = std(xk);
    if sx == 0
        error('Signal at r=%.4f has zero variance.', r_val);
    end
    xk = (xk - mean(xk)) / sx;
    scan_cache{iR} = xk;

    [~,pk] = findpeaks( xk,'MinPeakProminence',0.05);
    [~,tr] = findpeaks(-xk,'MinPeakProminence',0.05);
    extCount(iR) = numel(pk) + numel(tr);
    fprintf('  r=%.3f  extrema=%d\n', r_val, extCount(iR));

    if extCount(iR) < MIN_EXTREMA
        warning('r=%.3f has only %d extrema (< MIN_EXTREMA=%d). Proceeding anyway.', ...
            r_val, extCount(iR), MIN_EXTREMA);
    end
end

% Extrema bar chart
figure('Color','w');
bar(r_vec, extCount, 0.4, 'FaceColor',[0.2 0.4 0.7]); hold on;
yline(MIN_EXTREMA,'r--','LineWidth',1.5, ...
    'DisplayName',sprintf('MIN\\_EXTREMA=%d',MIN_EXTREMA));
xticks(r_vec);
xticklabels(arrayfun(@(r) sprintf('%.3f',r), r_vec, 'UniformOutput',false));
xlabel('r  (logistic map parameter)'); ylabel('Extrema count');
title(sprintf('Logistic map extrema counts  (keep\\_n=%d)',keep_n));
legend('Location','best'); box on;
saveas(gcf,sprintf('%s_regime_scan.png',prefix)); close(gcf);

% Signal inspection plot
figure('Color','w');
for iR = 1:nR
    subplot(nR,1,iR);
    xplot = scan_cache{iR};
    nn    = (0:numel(xplot)-1)';
    plot(nn(1:min(500,end)), xplot(1:min(500,end)),'k','LineWidth',0.7);
    ylabel(sprintf('r=%.3f',r_vec(iR))); box on;
end
xlabel('Iteration n');
sgtitle('Selected logistic map regimes');
saveas(gcf,sprintf('%s_selected_regimes.png',prefix)); close(gcf);

fprintf('Using lambda=%.4f (fixed)\n', lambda);

%% ========================= STORAGE ======================================

E_orig       = nan(nR, nMet);
E_surr       = nan(nR, Nsurr, nMet);
Z_surr       = nan(nR, nMet);
P_two_sided  = nan(nR, nMet);
IAAFT_nIter  = nan(nR, Nsurr);
IAAFT_ampErr = nan(nR, Nsurr);
signals            = cell(nR,1);
surrogates_example = cell(nR,1);

%% ========================= ORIGINAL ExSEnt ==============================

fprintf('\n===== ORIGINAL ExSEnt =====\n');

% Serial — only nR calls, negligible time
for iR = 1:nR
    signals{iR} = scan_cache{iR};
    try
        [hd,ha,hda,~,~,~,~,~] = ...
            compute_Sampentropies(scan_cache{iR}, lambda, m, alpha);
        E_orig(iR,:) = [hd, ha, hda];
        fprintf('  r=%.3f  HD=%.4f  HA=%.4f  HDA=%.4f\n', ...
            r_vec(iR), hd, ha, hda);
    catch ME
        warning('Original ExSEnt failed for r=%.4f: %s', r_vec(iR), ME.message);
    end
end

%% ========================= SURROGATE LOOP ===============================
%
% For each r-value, surrogates are split into batches of BATCH_SIZE.
% Each batch runs a fresh parfor with a fresh pool — this prevents the
% per-worker memory accumulation that causes "lost connection to worker"
% crashes.  A checkpoint is saved after every batch so a crash can be
% resumed by simply re-running the script.

fprintf('\n===== SURROGATE COMPUTATION =====\n');

for iR = 1:nR

    r_val = r_vec(iR);
    fprintf('\n[%d/%d]  r=%.4f — %d surrogates in batches of %d\n', ...
        iR, nR, r_val, Nsurr, BATCH_SIZE);

    if all(isnan(E_orig(iR,:)))
        warning('Original ExSEnt NaN for r=%.4f — skipping surrogates.', r_val);
        continue;
    end

    x_loc = scan_cache{iR};   % plain array — parfor broadcast safe

    % Pre-allocate
    nIter_tmp  = nan(Nsurr,1);
    ampErr_tmp = nan(Nsurr,1);
    E_tmp      = nan(Nsurr,nMet);

    % --- Resume: load checkpoint if it exists ----------------------------
    chk_file = sprintf('%s_chk_r%d.mat', prefix, iR);
    ss_done  = 0;
    if isfile(chk_file)
        fprintf('  Checkpoint found — resuming.\n');
        chk = load(chk_file, 'nIter_tmp','ampErr_tmp','E_tmp','ss_done');
        nIter_tmp  = chk.nIter_tmp;
        ampErr_tmp = chk.ampErr_tmp;
        E_tmp      = chk.E_tmp;
        ss_done    = chk.ss_done;
        fprintf('  Resuming from surrogate %d / %d\n', ss_done+1, Nsurr);
    end

    % --- Batch loop ------------------------------------------------------
    t_total     = tic;
    batch_edges = [ss_done+1 : BATCH_SIZE : Nsurr, Nsurr+1];

    for bIdx = 1 : numel(batch_edges)-1

        b_start = batch_edges(bIdx);
        b_end   = batch_edges(bIdx+1) - 1;
        b_size  = b_end - b_start + 1;

        fprintf('  Batch %d/%d  (surr %d–%d)  starting pool ...\n', ...
            bIdx, numel(batch_edges)-1, b_start, b_end);

        % Fresh pool — clears all worker memory from previous batch
        p = gcp('nocreate');
        if ~isempty(p), delete(p); pause(3); end
        parpool('local', MAX_WORKERS);

        % Slice inputs for this batch
        b_nIter  = nan(b_size,1);
        b_ampErr = nan(b_size,1);
        b_E      = nan(b_size,nMet);

        % ----- parfor over this batch ------------------------------------
parfor ss = 1:b_size

    row = nan(1,nMet);
    nIt = NaN;
    aEr = NaN;

    try
        [xs_loc, info_loc] = iaaft_surrogate(x_loc, IAAFT_maxIter, IAAFT_tol);
        xs_loc = xs_loc(:);

        nIt = info_loc.nIter;
        aEr = info_loc.finalAmpErr;

        [hd_s,ha_s,hda_s,~,~,~,~,~] = ...
            compute_Sampentropies(xs_loc, lambda, m, alpha);

        row = [hd_s, ha_s, hda_s];

    catch ME
        % Do not rethrow inside parfor.
        % Failed surrogate remains NaN.
        fprintf('Surrogate %d failed: %s\n', ss, ME.message);
    end

    b_nIter(ss)  = nIt;
    b_ampErr(ss) = aEr;
    b_E(ss,:)    = row;
end
        % ----- end parfor ------------------------------------------------

        % Write batch results into full arrays
        nIter_tmp(b_start:b_end)  = b_nIter;
        ampErr_tmp(b_start:b_end) = b_ampErr;
        E_tmp(b_start:b_end,:)    = b_E;
        ss_done = b_end;

        % Shut pool down cleanly before next batch
        p = gcp('nocreate');
        if ~isempty(p), delete(p); pause(3); end

        % Progress
        elapsed = toc(t_total);
        rate    = ss_done / elapsed;
        eta     = (Nsurr - ss_done) / rate;
        fprintf('  %d/%d surrogates done  |  %.1f surr/min  |  ETA %.0f s\n', ...
            ss_done, Nsurr, rate*60, eta);

        % Checkpoint
        save(chk_file, 'nIter_tmp','ampErr_tmp','E_tmp','ss_done', '-v7.3');
    end

    % Store into main arrays
    IAAFT_nIter(iR,:)  = nIter_tmp;
    IAAFT_ampErr(iR,:) = ampErr_tmp;
    E_surr(iR,:,:)     = E_tmp;

    % Example surrogate for plotting
    rng(42 + iR);
    [xs_ex,~] = iaaft_surrogate(x_loc, IAAFT_maxIter, IAAFT_tol);
    surrogates_example{iR} = xs_ex(:);

    fprintf('  ampErr: mean=%.2e  max=%.2e | iter: mean=%.0f  max=%.0f\n', ...
        mean(ampErr_tmp,'omitnan'), max(ampErr_tmp,[],'omitnan'), ...
        mean(nIter_tmp, 'omitnan'), max(nIter_tmp, [],'omitnan'));

    % --- Statistics ------------------------------------------------------
    for iMet = 1:nMet
        svals = E_tmp(~isnan(E_tmp(:,iMet)), iMet);
        nv    = numel(svals);

        if nv < MIN_VALID_SURR
            warning('Too few valid surrogates (n=%d) for r=%.4f, %s.', ...
                nv, r_val, metricNames{iMet});
            continue;
        end

        mu_s = mean(svals);
        sd_s = std(svals, 0);
        if sd_s > 0
            Z_surr(iR,iMet) = (E_orig(iR,iMet) - mu_s) / sd_s;
        end

        p_upper = (1 + sum(svals >= E_orig(iR,iMet))) / (1 + nv);
        p_lower = (1 + sum(svals <= E_orig(iR,iMet))) / (1 + nv);
        P_two_sided(iR,iMet) = min(1, 2*min(p_upper, p_lower));
    end

    % Delete checkpoint now that this r-value is complete
    if isfile(chk_file), delete(chk_file); end
end

%% ========================= SUMMARY TABLE ================================

rows = table();
for iR = 1:nR
    for iMet = 1:nMet
        sv = squeeze(E_surr(iR,:,iMet));
        sv = sv(~isnan(sv));

        r_row = table();
        r_row.r_val        = r_vec(iR);
        r_row.metric       = string(metricNames{iMet});
        r_row.N_valid_surr = numel(sv);
        r_row.E_original   = E_orig(iR,iMet);
        r_row.E_IAAFT_mean = mean(sv);
        r_row.E_IAAFT_sd   = std(sv,0);
        r_row.Z_IAAFT      = Z_surr(iR,iMet);
        r_row.P_two_sided  = P_two_sided(iR,iMet);
        r_row.ampErr_mean  = mean(IAAFT_ampErr(iR,:),'omitnan');
        r_row.ampErr_max   = max( IAAFT_ampErr(iR,:),[],'omitnan');
        r_row.nIter_mean   = mean(IAAFT_nIter(iR,:), 'omitnan');
        r_row.nIter_max    = max( IAAFT_nIter(iR,:), [],'omitnan');
        rows = [rows; r_row]; %#ok<AGROW>
    end
end
disp(rows);

%% ========================= SAVE =========================================

Results.settings = struct('discard_n',discard_n,'keep_n',keep_n, ...
    'r_vec',r_vec,'Nsurr',Nsurr,'m',m,'alpha',alpha,'lambda',lambda, ...
    'IAAFT_maxIter',IAAFT_maxIter,'IAAFT_tol',IAAFT_tol, ...
    'MIN_EXTREMA',MIN_EXTREMA);
Results.signals            = signals;
Results.surrogates_example = surrogates_example;
Results.E_orig             = E_orig;
Results.E_surr             = E_surr;
Results.Z_surr             = Z_surr;
Results.P_two_sided        = P_two_sided;
Results.IAAFT_nIter        = IAAFT_nIter;
Results.IAAFT_ampErr       = IAAFT_ampErr;
Results.summaryTable       = rows;

save(sprintf('%s_results.mat',  prefix), 'Results',  '-v7.3');
writetable(rows, sprintf('%s_summary.csv', prefix));
save(sprintf('%s_workspace.mat', prefix), '-v7.3');
fprintf('\nResults and full workspace saved: %s\n', prefix);

%% ========================= PLOTS ========================================

%% -- 1. Original vs surrogate per metric --
for iMet = 1:nMet
    smean = squeeze(mean(E_surr(:,:,iMet), 2, 'omitnan'));
    q25   = squeeze(prctile(E_surr(:,:,iMet), 25, 2));
    q75   = squeeze(prctile(E_surr(:,:,iMet), 75, 2));
    siqr  = q75 - q25;

    figure('Color','w'); hold on; box on;
    errorbar(r_vec, smean, siqr, '-o', 'LineWidth',1.5, 'MarkerSize',8, ...
        'DisplayName', sprintf('IAAFT mean \\pm IQR  (N=%d)', Nsurr));
    plot(r_vec, E_orig(:,iMet), 's-', 'LineWidth',2, 'MarkerSize',8, ...
        'DisplayName','Original logistic map');
    xticks(r_vec);
    xticklabels(arrayfun(@(r) sprintf('%.3f',r), r_vec, 'UniformOutput',false));
    xlabel('r  (logistic map parameter)'); ylabel(metricNames{iMet});
    legend('Location','best');
    title(sprintf('%s: Logistic map vs IAAFT surrogates', metricNames{iMet}));
    saveas(gcf, sprintf('%s_%s_vs_surrogate.png', prefix, metricNames{iMet}));
    close(gcf);
end

%% -- 2. Z-scores --
figure('Color','w'); hold on; box on;
for iMet = 1:nMet
    plot(r_vec, Z_surr(:,iMet), 'o-', 'LineWidth',2, 'MarkerSize',8, ...
        'DisplayName', metricNames{iMet});
end
yline(0,    'k--', 'HandleVisibility','off');
yline( 1.96,'r:',  'DisplayName','\pm1.96');
yline(-1.96,'r:',  'HandleVisibility','off');
xticks(r_vec);
xticklabels(arrayfun(@(r) sprintf('%.3f',r), r_vec, 'UniformOutput',false));
xlabel('r  (logistic map parameter)'); ylabel('Z  (IAAFT-corrected)');
legend('Location','best');
title('Surrogate-corrected ExSEnt Z-scores — Logistic map');
saveas(gcf, sprintf('%s_Z_values.png', prefix)); close(gcf);

%% -- 3. IAAFT quality: spectral amplitude error --
figure('Color','w');
for iR = 1:nR
    subplot(1,nR,iR);
    histogram(IAAFT_ampErr(iR,:), 20, 'FaceColor',[0.2 0.4 0.7]);
    xlabel('Spectral amp. error'); ylabel('Count');
    title(sprintf('r=%.3f  mean=%.2e', r_vec(iR), ...
        mean(IAAFT_ampErr(iR,:),'omitnan'))); box on;
end
sgtitle('IAAFT convergence: spectral amplitude mismatch');
saveas(gcf, sprintf('%s_quality_ampErr.png', prefix)); close(gcf);

%% -- 4. IAAFT quality: iteration count --
figure('Color','w');
for iR = 1:nR
    subplot(1,nR,iR);
    histogram(IAAFT_nIter(iR,:), 'FaceColor',[0.7 0.3 0.2]);
    xlabel('Iterations'); ylabel('Count');
    title(sprintf('r=%.3f  max=%d', r_vec(iR), ...
        max(IAAFT_nIter(iR,:),[],'omitnan'))); box on;
end
sgtitle('IAAFT convergence: iteration count');
saveas(gcf, sprintf('%s_quality_nIter.png', prefix)); close(gcf);

%% -- 5. Surrogate diagnostics: PSD, amplitude PDF, ACF --
nfft   = min(4096, keep_n);
maxLag = min(2000, round(keep_n/10));
nBins  = 80;

for iR = 1:nR
    if isempty(signals{iR}) || isempty(surrogates_example{iR}), continue; end
    x  = signals{iR};
    xs = surrogates_example{iR};

    figure('Color','w','Position',[100 100 1200 350]);

    subplot(1,3,1);
    [Px, fx  ] = periodogram(x,  hanning(numel(x)),  nfft);
    [Pxs,fxs ] = periodogram(xs, hanning(numel(xs)), nfft);
    semilogy(fx,  Px,  'k',   'LineWidth',1.2, 'DisplayName','Original'); hold on;
    semilogy(fxs, Pxs, '--',  'Color',[0.8 0.2 0.2], ...
        'LineWidth',1, 'DisplayName','IAAFT');
    xlabel('Frequency (cycles/sample)'); ylabel('PSD');
    legend('Location','best'); title('PSD'); box on;

    subplot(1,3,2);
    edges = linspace(min([x;xs]), max([x;xs]), nBins+1);
    [cx, ex] = histcounts(x,  edges, 'Normalization','pdf');
    [cxs,~]  = histcounts(xs, edges, 'Normalization','pdf');
    stairs(ex(1:end-1), cx,  'k',  'LineWidth',1.2); hold on;
    stairs(ex(1:end-1), cxs, '--', 'Color',[0.8 0.2 0.2], 'LineWidth',1);
    xlabel('Amplitude (z-scored)'); ylabel('PDF');
    legend({'Original','IAAFT'},'Location','best');
    title('Amplitude distribution'); box on;

    subplot(1,3,3);
    [acx, lags] = xcorr(x,  maxLag, 'normalized');
    [acxs,~   ] = xcorr(xs, maxLag, 'normalized');
    plot(lags, acx,  'k',  'LineWidth',1.2); hold on;
    plot(lags, acxs, '--', 'Color',[0.8 0.2 0.2], 'LineWidth',1);
    xlabel('Lag (samples)'); ylabel('ACF');
    legend({'Original','IAAFT'},'Location','best');
    title('Autocorrelation'); box on;

    sgtitle(sprintf('IAAFT diagnostics  –  r=%.3f', r_vec(iR)));
    saveas(gcf, sprintf('%s_diagnostics_r%.3f.png', prefix, r_vec(iR))); close(gcf);
end

%% -- 6. Time-series example --
nShow = min(500, keep_n);
for iR = 1:nR
    if isempty(signals{iR}) || isempty(surrogates_example{iR}), continue; end
    x  = signals{iR};
    xs = surrogates_example{iR};
    nn = (0:numel(x)-1)';

    figure('Color','w');
    subplot(2,1,1);
    plot(nn(1:nShow), x(1:nShow),  'k', 'LineWidth',0.8);
    ylabel('x_n');
    title(sprintf('Original logistic map,  r=%.3f', r_vec(iR))); box on;
    subplot(2,1,2);
    plot(nn(1:nShow), xs(1:nShow), 'Color',[0.2 0.4 0.8], 'LineWidth',0.8);
    xlabel('Iteration n'); ylabel('x_n (IAAFT)');
    title('IAAFT surrogate (example #1)'); box on;
    saveas(gcf, sprintf('%s_timeseries_r%.3f.png', prefix, r_vec(iR))); close(gcf);
end

%% -- 7. Bifurcation diagram --
r_bif         = linspace(2.8, 4.0, 1400);
n_bif_discard = 500;
n_bif_keep    = 200;
nBif          = numel(r_bif);
bif_r_cell    = cell(nBif,1);
bif_x_cell    = cell(nBif,1);

parpool('local', MAX_WORKERS);
parfor k = 1:nBif
    xb            = logistic_map(0.4, r_bif(k), n_bif_discard, n_bif_keep);
    bif_r_cell{k} = r_bif(k) * ones(n_bif_keep,1);
    bif_x_cell{k} = xb;
end
delete(gcp('nocreate'));

bif_r = vertcat(bif_r_cell{:});
bif_x = vertcat(bif_x_cell{:});

figure('Color','w','Position',[100 100 900 400]);
plot(bif_r, bif_x, '.k', 'MarkerSize',0.4); hold on;
for ii = 1:nR
    xline(r_vec(ii), 'r-', 'LineWidth',1.5, ...
        'Label', sprintf('r=%.3f',r_vec(ii)), ...
        'LabelVerticalAlignment','bottom');
end
xlabel('r'); ylabel('x_n (attractor)');
title('Logistic map bifurcation diagram with selected r values');
xlim([r_bif(1) r_bif(end)]); box on;
saveas(gcf, sprintf('%s_bifurcation.png', prefix)); close(gcf);

fprintf('\nDone.\n');

%% ========================================================================
%%  LOCAL FUNCTIONS
%% ========================================================================

function x = logistic_map(x0, r, n_discard, n_keep)
    xc = x0;
    for k = 1:n_discard
        xc = r * xc * (1 - xc);
    end
    x = zeros(n_keep,1);
    for k = 1:n_keep
        xc   = r * xc * (1 - xc);
        x(k) = xc;
    end
end

function [xs, info] = iaaft_surrogate(x, maxIter, tol)
    if nargin<2||isempty(maxIter), maxIter=100;  end
    if nargin<3||isempty(tol),     tol    =1e-4; end

    x        = x(:);
    N        = numel(x);
    x_sorted = sort(x);
    amp_orig = abs(fft(x));
    xs           = x(randperm(N));
    amp_err      = Inf;
    amp_err_prev = Inf;

    for it = 1:maxIter
        phase       = angle(fft(xs));
        y           = real(ifft(amp_orig .* exp(1i*phase)));
        [~,ord]     = sort(y);
        xs_new      = zeros(N,1);
        xs_new(ord) = x_sorted;
        amp_err     = norm(abs(fft(xs_new)) - amp_orig) / norm(amp_orig);
        xs          = xs_new;
        if amp_err < tol,                          break; end
        if abs(amp_err_prev - amp_err) < tol*0.01, break; end
        amp_err_prev = amp_err;
    end

    info.nIter       = it;
    info.finalAmpErr = amp_err;
end