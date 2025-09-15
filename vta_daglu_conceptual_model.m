% VTA DA / DA+Glu conceptual population model for reward and shock conditioning
% -----------------------------------------------------------------------------
% Purpose
%   Compact, phenomenological simulation illustrating how two VTA populations
%   (dopamine-only and dopamine+glutamate) could differentially bias action
%   selection in appetitive vs aversive contexts.
%
% What this IS
%   • A minimal neural *population* model: two transmitter-driven inputs feed a
%     single action-accumulation variable with a thresholded readout.
%   • State variables (input “gains”) increment at cue/feedback events and decay
%     slowly over time, producing acquisition-like changes across days.
%   • A tool to generate qualitative patterns and simple KO contrasts that align
%     with our recordings/behavior at a coarse level.
%
% Outputs (written to ./data/)
%   • params_YYYYMMDD_HHMMSS.mat  and  params_YYYYMMDD_HHMMSS.json
%   • figure{1..3}_YYYYMMDD_HHMMSS.png / .fig   (reward hits/latency; shock freezing)
%   • Console: permutation-test summaries (see “Statistics” below).
%
% How to run
%   • Run this script as-is (MATLAB R2018b+; jsonencode available in R2016b+).
%   • Key knobs:
%       - Reward:  params.h (threshold)
%       - Shock:   params_shock.h (threshold), .c (noise), .taup (decay)
%       - Stimuli: wdrive, wdrive_shock (8-element drive vectors)
%       - KOs:     apply_knockout(P,'glutamate') or apply_knockout(P,'dopamine',0.5)
%   • Reproducibility: rng(42)
%
% Statistics
%   • Primary curve comparisons use a permutation test on Euclidean (L2) distance
%     between per-day group means (unit of permutation = blocks).
%   • Terminal performance uses the mean over the LAST 5 DAYS with a permutation
%     test on |Δ mean|. P-values use the “add-one” correction: (b+1)/(N+1).
%
% Interpretation notes
%   • Units are arbitrary. Curves showing day-to-day change reflect stateful gains
%     with decay and event-driven increments; these illustrate trends and are not
%     evidence of a particular learning rule or fitted dynamics.
%
% Figure mapping
%   • Produces panels corresponding to Supplementary Fig. 8 (behavioral summaries
%     and KO contrasts).
%
% File: vta_daglu_conceptual_model.m
% -----------------------------------------------------------------------------

% dopglu5_learning.m  (compressed)
% Conceptual DA/GLU population model with KO sims, plotting, and permutation tests

close all; rng(42);

% ------------ GLOBALS ------------
NPERM = 10000;                      % permutations
OUTDIR = fullfile('data');          % save here in current folder
if ~exist(OUTDIR,'dir'); mkdir(OUTDIR); end
TS = datestr(now,'yyyymmdd_HHMMSS');% timestamp for filenames

% ------------ BASE PARAMS ------------
params = create_base_params();
params.freeze_mode = 'binary';      % reward = binary decision

% ------------ REWARD CONTEXT ------------
wdrive = [13; 10; 8; 0; 11; 0; 6; 0];
params.h = 0.6;                               % faster early hits
params.Istim = create_input_drive(wdrive, params);
[choice_rewcont, del_rewcont, ~] = simulate_behavior(params, wdrive, false, false);

% KO: Glutamate in reward
params_ko = apply_knockout(params, 'glutamate');
[choice_rewknock, del_rewknock, ~] = simulate_behavior(params_ko, wdrive, false, false);

% ------------ SHOCK CONTEXT ------------
params_shock = params;
params_shock.freeze_mode = 'fractional';
params_shock.taup = 1000; params_shock.h = 2.5; params_shock.c = 0.5;
wdrive_shock = [14; 10; 6; 0; -2; 0; 0; 0];
params_shock.Istim = create_input_drive(wdrive_shock, params_shock);
[choice_shockcont, del_shockcont, ~] = simulate_behavior(params_shock, wdrive_shock, true, false);

% KO: 50% Dopamine in shock
params_shock_ko = apply_knockout(params_shock, 'dopamine', 0.5);
[choice_shockknock, del_shockknock, ~] = simulate_behavior(params_shock_ko, wdrive_shock, true, false);

% ------------ PLOTTING ------------
plot_reward_behavior(choice_rewcont, del_rewcont, 'k');
plot_reward_behavior(choice_rewknock, del_rewknock, 'g');
figure(1); legend('Reward Control', 'Reward KO (Glu)', 'Location','best');
figure(2); legend('Reward Control', 'Reward KO (Glu)', 'Location','best');

plot_freezing_behavior(choice_shockcont, del_shockcont, 'k');
plot_freezing_behavior(choice_shockknock, del_shockknock, 'b');
figure(3); legend('Shock Control', 'Shock KO (DA)', 'Location','best');

% ------------ STATS (Euclidean/last-5) ------------
perm_euclid_curves(choice_rewcont,  choice_rewknock,  'Reward CS+ hits (curve across days)', NPERM);
perm_euclid_curves(del_rewcont,     del_rewknock,     'Reward latency (curve across days)', NPERM);
perm_euclid_curves(choice_shockcont,choice_shockknock,'Shock freeze fraction (curve across days)', NPERM);

perm_euclid_lastK(choice_rewcont, choice_rewknock, 2, ...
    'Reward CS+ hits (last 2 days)', NPERM);

% ------------ SAVE ------------
all_params = struct('params',params, 'params_ko',params_ko, ...
    'params_shock',params_shock, 'params_shock_ko',params_shock_ko, ...
    'wdrive_reward',wdrive, 'wdrive_shock',wdrive_shock);

save(fullfile(OUTDIR, ['params_' TS '.mat']), '-struct', 'all_params');

fid = fopen(fullfile(OUTDIR, ['params_' TS '.json']), 'w');
fwrite(fid, jsonencode(all_params)); fclose(fid);

save_all_figs(OUTDIR, TS);
fprintf('Saved outputs to %s (stamp %s)\n', OUTDIR, TS);

% ================== FUNCTIONS ==================

function Istim = create_input_drive(wdrive, P)
    normf = P.taup * (1 - exp(-(P.Ts + P.Td) / P.taup)) / exp(-P.Td / P.taup);
    Istim = (normf / P.pr) * wdrive / P.taup;
end

function P = create_base_params()
    P = struct('Ts',10,'Td',30,'taua',10,'taup',25000, ...
               'taus',10,'dt',0.01,'h',0.8,'c',0.05,'pr',1.0, ...
               'Nblocks',10,'days',20,'tperday',20, ...
               'wg',0.4,'wd',0.3,'swg',0.3,'swd',0.3,'swnd',0.2);
    P.Ntrials = P.days * P.tperday;
    P.nt_trial = round((P.Ts + P.Td) / P.dt);
    P.nt = round(P.Ntrials * (P.Ts + P.Td) / P.dt);
    P.timey = linspace(0, P.Ntrials * (P.Ts + P.Td), P.nt);
end

function P = apply_knockout(P, type, strength)
    if nargin < 3, strength = 0; end
    switch type
        case 'glutamate', P.wg = P.wg * strength;
        case 'dopamine',  P.wd = P.wd * strength;
    end
end

function [choice_all, latency_all, traces] = simulate_behavior(P, wtrain, shock, return_trace)
    N = P.Nblocks; D = P.days; T = P.tperday;
    choice_all = zeros(N, D); latency_all = zeros(N, D); traces = struct();
    for b = 1:N
        [c, d, tr] = run_single_block(P, wtrain, shock, return_trace && b==1);
        % Reshape trials → [T x D], vectorized per-day stats
        C = reshape(c, T, D); L = reshape(d, T, D);
        choice_all(b,:)  = mean(C, 1);
        L(C==0) = NaN;                      % ignore non-hits for latency
        latency_all(b,:) = mean(L, 1, 'omitnan');
        if return_trace && b==1, traces = tr; end
    end
end

function [choices, delays, traces] = run_single_block(P, wtrain, shock, return_trace)
    nt = P.nt_trial; N = P.Ntrials; dt = P.dt; h = P.h;
    wrs = zeros(8, N*nt);
    Istim = create_input_drive(wtrain, P);  % already divided by taup above
    choices = zeros(N,1); delays = zeros(N,1); traces = struct();

    for k = 1:N
        gds = zeros(nt,1); nds = zeros(nt,1);
        sbd = 0; sbg = 0; snd = 0; ya = zeros(nt,1); del_time = NaN;

        for j = 1:nt
            idx = (k-1)*nt + j;

            % transmitter updates
            if j == 1
                gds(j+1) = gds(j) + wrs(1,idx);
                nds(j+1) = nds(j) + wrs(5,idx);
            elseif j < round(P.Ts/dt)
                gds(j+1) = gds(j) + (gds(j) - wrs(2,idx)) * (exp(-dt/P.taua)-1);
                nds(j+1) = nds(j) + (nds(j) - wrs(6,idx)) * (exp(-dt/P.taua)-1);
            else
                gds(j+1) = gds(j) * exp(-dt/P.taua);
                nds(j+1) = nds(j) * exp(-dt/P.taua);
            end

            % passive decay
            wrs(:,idx+1) = wrs(:,idx) * exp(-dt/P.taup);

            % stimulus at CS end
            if j == round(P.Ts/dt)
                if rand < P.pr || k == N
                    wrs(:,idx+1) = wrs(:,idx+1) + Istim;
                    gds(j+1) = gds(j) + wrs(3,idx);
                    nds(j+1) = nds(j) + wrs(7,idx);
                else
                    gds(j+1) = gds(j) + wrs(4,idx);
                    nds(j+1) = nds(j) + wrs(8,idx);
                end
            end

            % synaptic filtering
            sbd = sbd + dt*(P.swd * gds(j) - sbd)/P.taus;
            sbg = sbg + dt*(P.swg * gds(j) - sbg)/P.taus;
            snd = snd + dt*(P.swnd* nds(j) - snd)/P.taus;

            % action accumulation
            cs_end = round(P.Ts/dt);
            if (ya(j) < h && j < cs_end) || (shock && isfield(P,'freeze_mode') && strcmp(P.freeze_mode,'fractional') && j < cs_end)
                ya(j+1) = ya(j) + dt*(P.wd*(sbd+snd) + P.wg*sbg) + sqrt(dt*P.c)*randn;
                del_time = j*dt;
            else
                ya(j+1) = ya(j);
            end
        end

        % === Behavioral readout ===
        cs_idx = 1 : round(P.Ts / dt);
        if isfield(P,'freeze_mode') && strcmp(P.freeze_mode,'fractional') && shock
            freeze_fraction = mean(ya(cs_idx) > h);
            choices(k) = freeze_fraction;
            if freeze_fraction > 0
                delays(k) = del_time;
            else
                delays(k) = NaN;
            end
        else
            hit = any(ya(cs_idx) > h);
            choices(k) = hit;
            if hit
                delays(k) = del_time;
            else
                delays(k) = NaN;
            end
        end

        % traces (optional)
        if return_trace && k==N
            gds = gds(1:end-1); nds = nds(1:end-1);
            traces.u_gd = gds; traces.u_d = nds;
            traces.s_g = 0.3*gds; traces.s_d = 0.3*nds;
        end
    end
end

% ---------- Plotters ----------
function plot_reward_behavior(choice_mat, latency_mat, color)
    D = size(choice_mat,2);
    cd = mean(choice_mat); cd_err = std(choice_mat)/sqrt(size(choice_mat,1));
    ld = mean(latency_mat,'omitnan'); ld_err = std(latency_mat,'omitnan')/sqrt(size(latency_mat,1));
    figure(1); hold on; errorbar(1:D, cd, cd_err, [color '-o'], 'LineWidth', 5);
    set(gca,'fontsize',30); xlabel('Day'); ylabel('CS+ Hits');
    figure(2); hold on; errorbar(1:D, ld, ld_err, [color '-o'], 'LineWidth', 5);
    set(gca,'fontsize',30); xlabel('Day'); ylabel('CS+ Hit Latency (s)');
end

function plot_freezing_behavior(freeze_mat, latency_mat, color)
    D = size(freeze_mat,2);
    fd = mean(freeze_mat); fd_err = std(freeze_mat)/sqrt(size(freeze_mat,1));
    figure(3); hold on; errorbar(1:D, fd, fd_err, [color '-o'], 'LineWidth', 5);
    set(gca,'fontsize',30); xlabel('Day'); ylabel('CS+ Freeze Fraction');
end

% ---------- Permutation tests ----------
function perm_euclid_curves(A, B, label, nperm)
    mA = mean(A,1,'omitnan'); mB = mean(B,1,'omitnan');
    stat_obs = norm(mA - mB, 2);
    AB = [A;B]; nA = size(A,1); nTot = size(AB,1);
    stats = zeros(nperm,1);
    for i=1:nperm
        idx = randperm(nTot);
        mAp = mean(AB(idx(1:nA),:),1,'omitnan');
        mBp = mean(AB(idx(nA+1:end),:),1,'omitnan');
        stats(i) = norm(mAp - mBp, 2);
    end
    p = (sum(stats >= stat_obs) + 1) / (nperm + 1);   % add-one correction
    fprintf('%s: Euclidean distance = %.4f, p = %.4g (nPerm=%d)\n', label, stat_obs, p, nperm);
end

function perm_euclid_lastK(A, B, K, label, nperm)
    a = mean(A(:,end-K+1:end),2,'omitnan');
    b = mean(B(:,end-K+1:end),2,'omitnan');
    stat_obs = abs(mean(a,'omitnan') - mean(b,'omitnan'));  % 1D Euclidean
    combo = [a;b]; n = numel(a); stats = zeros(nperm,1);
    for i=1:nperm
        idx = randperm(numel(combo));
        ap = combo(idx(1:n)); bp = combo(idx(n+1:end));
        stats(i) = abs(mean(ap,'omitnan') - mean(bp,'omitnan'));
    end
    p = (sum(stats >= stat_obs) + 1) / (nperm + 1);
    fprintf('%s: |mean diff| = %.4f, p = %.4g (nPerm=%d)\n', label, stat_obs, p, nperm);
end

% ---------- Saving ----------
function save_all_figs(outdir, ts)
    figs = findall(0,'Type','figure');
    for k = 1:numel(figs)
        f = figs(k);
        set(f,'PaperPositionMode','auto');
        fn_png = fullfile(outdir, sprintf('figure%d_%s.png', get(f,'Number'), ts));
        fn_fig = fullfile(outdir, sprintf('figure%d_%s.fig', get(f,'Number'), ts));
        print(f, fn_png, '-dpng','-r300'); savefig(f, fn_fig);
    end
end
