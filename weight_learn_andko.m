%
% weight_learn_andko.m
%
% code for learning the weights associated with stimulus for conditioned
% reward condition and then the shock condition.
%
% we assume there is a single action possible, selected
% based on feedback either being shock or reward.
%
%

% model parameters
taup = 2500;     % plasticity parameter
Ts = 10;        % stimulus time
Td = 30;        % decay time
pr = 0.5;       % probability of reward

% first focus on target trained activity weight parameters for reward
delgdc_train = 13;  % increment from a cue to dop-glu
ssgdc_train = 10;   % steady state after cue for dop-glu
delgdr_train = 8;   % increment after reward to dop-glu
delgdn_train = 0;   % increment from no reward to dop-glu
delndc_train = 11;  % increment after cue for nonglu-dop
ssndc_train = 0;    % steady state after cue for nonglu-dop
delndr_train = 6;   % increment from a reward to nonglu-dop
delndn_train = 0;   % increment from no reward to nonglu-dop
taua = 10;          % time constant of activity
rt = 10;            % shock or reward time

% next, the dopamine and glutamate outputs
swg = 0.3;
swd = 0.3;
swnd = 0.3;
taus = 10;

% last, let's focus on the drive of these to action selection
wg = 0.2; % weight given to glutamate trace
wd = 0.3; % weight given to dopamine trace
c = 0.05; % noise amplitude
h = 0.8;  % threshold value (single, determines latency)

% simulation parameters
Nblocks = 10;
days = 20;   % number of days
tperday = 20;       % number of trials per day
dpts = Nblocks*tperday;
Ntrials = days*tperday;
T = (Ts+Td)*Ntrials;
dt = 0.01;
nt_trial = round((Ts+Td)/dt);
nt = round(T/dt)+1;
timey = linspace(0,T,nt);

% activity variables
gds = zeros(nt_trial,1);  % glutamate-dopamine neuron activity
nds = zeros(nt_trial,1);  % no glu-dopamine neuron activity

% synaptic variables
sbd = zeros(nt_trial,1);  % dopamine output of glu-dop neuron
sbg = zeros(nt_trial,1);  % glutamate output of glu-dop neuron
snd = zeros(nt_trial,1);  % dopamine output of noglu-dop neuron

choices = zeros(Ntrials,1);
del_times = zeros(Ntrials,1);

% now define all empty weight time series
wrs = zeros(8,nt);  wrs_trials = zeros(8,Ntrials);

% the associated input that will bring them to the appropriate trained val
wtrain = [delgdc_train; ssgdc_train; delgdr_train; delgdn_train;
    delndc_train; ssndc_train; delndr_train; delndn_train];
Istim = taup*(1-exp(-(Ts+Td)/taup))/exp(-Td/taup)/pr*wtrain;
Istim/taup

%
% REWARD TASK
%

for k = 1:Ntrials

    del_time = 0;
    ya = zeros(nt_trial,1);   % action selection variable (ddm)

    for j=1:nt_trial, ind = (k-1)*nt_trial+j;

        if j==1
            gds(j+1)=gds(j)+wrs(1,ind);
            nds(j+1)=nds(j)+wrs(5,ind);
        end

        if j>1 & j<round(Ts/dt)
            gds(j+1)=gds(j)+(gds(j)-wrs(2,ind))*(exp(-dt/taua)-1);
            nds(j+1)=nds(j)+(nds(j)-wrs(6,ind))*(exp(-dt/taua)-1);
        end
        
        wrs(:,ind+1) = wrs(:,ind)*exp(-dt/taup);

        if j==round(Ts/dt)
            if rand<pr
                wrs(:,ind+1) = wrs(:,ind+1)+Istim/taup;
                gds(j+1)=gds(j)+wrs(3,ind);
                nds(j+1)=nds(j)+wrs(7,ind);
            else
                gds(j+1)=gds(j)+wrs(4,ind);
                nds(j+1)=nds(j)+wrs(8,ind);
            end
        end

        if j>round(Ts/dt)
            gds(j+1)=gds(j)*exp(-dt/taua);
            nds(j+1)=nds(j)*exp(-dt/taua);
        end

        sbd(j+1) = sbd(j)+dt*(swd*gds(j)-sbd(j))/taus;
        sbg(j+1) = sbg(j)+dt*(swg*gds(j)-sbg(j))/taus;
        snd(j+1) = snd(j)+dt*(swnd*nds(j)-snd(j))/taus;

        if ya(j)<h & j<round(Ts/dt)
            ya(j+1) = ya(j)+dt*(wd*(sbd(j)+snd(j))+wg*sbg(j))+sqrt(dt*c)*randn;
            del_time = j*dt;
        else
            ya(j+1) = ya(j);
            if ya(j)>h, choices(k)=1; del_times(k) = del_time; end
        end

    end

    wrs_trials(:,k)=wrs(:,ind+1);

end

tplot = dt*[-round(5/dt):nt_trial+1];

snd_plot = [zeros(round(5/dt)+1,1);snd];
sbd_plot = [zeros(round(5/dt)+1,1);sbd];
sbg_plot = [zeros(round(5/dt)+1,1);sbg];
figure(20), hold on, plot(tplot,snd_plot,'color',[16 0 238]/256,'linewidth',5);
hold on, plot(tplot,sbd_plot,'color',[116 249 250]/256,'linewidth',5);
hold on, plot(tplot,sbg_plot,'color',[29 117 0]/256,'linewidth',5);
set(gca,'fontsize',30);
xlabel('time (s)','fontsize',30,'interpreter','latex');
ylabel('activity (a.u.)','fontsize',30,'interpreter','latex');

% plot weight learning across days
% figure(1), hold on,
% plot([1:Ntrials]/day,wtrain(1)*ones(1,Ntrials),'--','color',[16 0 238]/256,'LineWidth',4);
% plot([1:Ntrials]/day,wtrain(3)*ones(1,Ntrials),'--','color',[16 0 238]/256,'LineWidth',4);
% plot([1:Ntrials]/day,wtrain(5)*ones(1,Ntrials),'--','color',[116 249 250]/256,'LineWidth',4);
% plot([1:Ntrials]/day,wtrain(7)*ones(1,Ntrials),'--','color',[116 249 250]/256,'LineWidth',4);
% plot([1:Ntrials]/day,wrs_trials(1,:),'-','color',[16 0 238]/256,'linewidth',4);
% plot([1:Ntrials]/day,wrs_trials(3,:),'-','color',[16 0 238]/256,'linewidth',4);
% plot([1:Ntrials]/day,wrs_trials(5,:),'-','color',[116 249 250]/256,'linewidth',4);
% plot([1:Ntrials]/day,wrs_trials(7,:),'-','color',[116 249 250]/256,'linewidth',4);
% set(gca,'fontsize',30);
% xlabel('day','fontsize',30,'interpreter','latex');
% ylabel('weight (a.u.)','fontsize',30,'interpreter','latex');

% now we will rerun these experiments another Nblocks-1 more times to get
% now make a matrix to contain data per data along the row
choice_data = zeros(dpts,days);
del_data = zeros(dpts,days);
t_day = tinv(0.975, dpts-1); % 0.975 corresponds to 95% confidence int
choice_rewcont = zeros(Nblocks,days);
del_rewcont = zeros(Nblocks,days);

for j=1:days
    choice_data(1:tperday,j) = choices((j-1)*tperday+1:j*tperday);
    del_data(1:tperday,j) = del_times((j-1)*tperday+1:j*tperday);
end    


% and now let us repeat the sims Nblock-1 more times to get the rest of the data
for m=2:Nblocks

    gds = zeros(nt_trial,1);  % glutamate-dopamine neuron activity
    nds = zeros(nt_trial,1);  % no glu-dopamine neuron activity
    sbd = zeros(nt_trial,1);  % dopamine output of glu-dop neuron
    sbg = zeros(nt_trial,1);  % glutamate output of glu-dop neuron
    snd = zeros(nt_trial,1);  % dopamine output of noglu-dop neuron 
    choices = zeros(Ntrials,1);
    del_times = zeros(Ntrials,1);
    wrs = zeros(8,nt);

    for k = 1:Ntrials, del_time = 0; ya = zeros(nt_trial,1);

        for j=1:nt_trial, ind = (k-1)*nt_trial+j;

            if j==1, gds(j+1)=gds(j)+wrs(1,ind); nds(j+1)=nds(j)+wrs(5,ind); end

            if j>1 & j<round(Ts/dt)
                gds(j+1)=gds(j)+(gds(j)-wrs(2,ind))*(exp(-dt/taua)-1);
                nds(j+1)=nds(j)+(nds(j)-wrs(6,ind))*(exp(-dt/taua)-1);
            end
        
            wrs(:,ind+1) = wrs(:,ind)*exp(-dt/taup);

            if j==round(Ts/dt)
                if rand<pr
                    wrs(:,ind+1) = wrs(:,ind+1)+Istim/taup;
                    gds(j+1)=gds(j)+wrs(3,ind);
                    nds(j+1)=nds(j)+wrs(7,ind);
                else
                    gds(j+1)=gds(j)+wrs(4,ind);
                    nds(j+1)=nds(j)+wrs(8,ind);
                end
            end

            if j>round(Ts/dt)
                gds(j+1)=gds(j)*exp(-dt/taua);
                nds(j+1)=nds(j)*exp(-dt/taua);
            end

            sbd(j+1) = sbd(j)+dt*(swd*gds(j)-sbd(j))/taus;
            sbg(j+1) = sbg(j)+dt*(swg*gds(j)-sbg(j))/taus;
            snd(j+1) = snd(j)+dt*(swnd*nds(j)-snd(j))/taus;

            if ya(j)<h & j<round(Ts/dt)
                ya(j+1) = ya(j)+dt*(wd*(sbd(j)+snd(j))+wg*sbg(j))+sqrt(dt*c)*randn;
                del_time = j*dt;
            else
                ya(j+1) = ya(j);
                if ya(j)>h, choices(k)=1; del_times(k) = del_time; end
            end

        end

    end

    for j=1:days
        choice_data((m-1)*tperday+1:m*tperday,j) = choices((j-1)*tperday+1:j*tperday);
        del_data((m-1)*tperday+1:m*tperday,j) = del_times((j-1)*tperday+1:j*tperday);
    end

end

choice_day = mean(choice_data);
choice_merr = t_day*std(choice_data)/sqrt(dpts);
schoice = sum(choice_data);
del_day = sum(del_data)./schoice;
del_std = zeros(1,days);
del_serr = zeros(1,days);
del_merr = zeros(1,days);
for j=1:days
    del_std(j) = std(del_data(choice_data(:,j)==1,j));
    del_serr(j) = del_std(j)/sqrt(schoice(j));
    del_t = tinv(0.975,schoice(j)-1);
    del_merr(j) = del_t*del_serr(j);
end

for k=1:Nblocks, for j=1:days
    choice_curr = choice_data((k-1)*tperday+1:k*tperday,j);
    choice_rewcont(k,j) = mean(choice_curr);
    del_curr = del_data((k-1)*tperday+1:k*tperday,j);
    del_rewcont(k,j) = mean(del_curr(choice_curr==1));
end
end

%
% plot choices across days and latency
%
figure(2), hold on, plot([1:days],choice_day,'k-o','linewidth',5);
hold on, errorbar([1:days],choice_day,choice_merr,'k','LineWidth',5);
set(gca,'fontsize',30);
xlabel('day','fontsize',30,'interpreter','latex');
ylabel('CS+ Hits','fontsize',30,'interpreter','latex');

figure(3), hold on, plot([1:days],del_day,'k-o','linewidth',5);
hold on, errorbar([1:days],del_day,del_merr,'k','LineWidth',5);
set(gca,'fontsize',30);
xlabel('day','fontsize',30,'interpreter','latex');
ylabel('CS+ Hit Latency (s)','fontsize',30,'interpreter','latex');
pause(1e-15);

%
% KNOCKOUT EXPERIMENTS FOR REWARD TASK
%
% now perform the knockout experiments
wg = 0; % weight given to glutamate trace

choice_rewknock = zeros(Nblocks,days);
del_rewknock = zeros(Nblocks,days);

% and now let us repeat the sims Nblock-1 more times to get the rest of the data
for m=1:Nblocks

    gds = zeros(nt_trial,1);  % glutamate-dopamine neuron activity
    nds = zeros(nt_trial,1);  % no glu-dopamine neuron activity
    sbd = zeros(nt_trial,1);  % dopamine output of glu-dop neuron
    sbg = zeros(nt_trial,1);  % glutamate output of glu-dop neuron
    snd = zeros(nt_trial,1);  % dopamine output of noglu-dop neuron 
    choices = zeros(Ntrials,1);
    del_times = zeros(Ntrials,1);
    wrs = zeros(8,nt);

    for k = 1:Ntrials, del_time = 0; ya = zeros(nt_trial,1);

        for j=1:nt_trial, ind = (k-1)*nt_trial+j;

            if j==1, gds(j+1)=gds(j)+wrs(1,ind); nds(j+1)=nds(j)+wrs(5,ind); end

            if j>1 & j<round(Ts/dt)
                gds(j+1)=gds(j)+(gds(j)-wrs(2,ind))*(exp(-dt/taua)-1);
                nds(j+1)=nds(j)+(nds(j)-wrs(6,ind))*(exp(-dt/taua)-1);
            end
        
            wrs(:,ind+1) = wrs(:,ind)*exp(-dt/taup);

            if j==round(Ts/dt)
                if rand<pr
                    wrs(:,ind+1) = wrs(:,ind+1)+Istim/taup;
                    gds(j+1)=gds(j)+wrs(3,ind);
                    nds(j+1)=nds(j)+wrs(7,ind);
                else
                    gds(j+1)=gds(j)+wrs(4,ind);
                    nds(j+1)=nds(j)+wrs(8,ind);
                end
            end

            if j>round(Ts/dt)
                gds(j+1)=gds(j)*exp(-dt/taua);
                nds(j+1)=nds(j)*exp(-dt/taua);
            end

            sbd(j+1) = sbd(j)+dt*(swd*gds(j)-sbd(j))/taus;
            sbg(j+1) = sbg(j)+dt*(swg*gds(j)-sbg(j))/taus;
            snd(j+1) = snd(j)+dt*(swnd*nds(j)-snd(j))/taus;

            if ya(j)<h & j<round(Ts/dt)
                ya(j+1) = ya(j)+dt*(wd*(sbd(j)+snd(j))+wg*sbg(j))+sqrt(dt*c)*randn;
                del_time = j*dt;
            else
                ya(j+1) = ya(j);
                if ya(j)>h, choices(k)=1; del_times(k) = del_time; end
            end

        end

    end

    for j=1:days
        choice_data((m-1)*tperday+1:m*tperday,j) = choices((j-1)*tperday+1:j*tperday);
        del_data((m-1)*tperday+1:m*tperday,j) = del_times((j-1)*tperday+1:j*tperday);
    end

end

choice_day = mean(choice_data);
choice_merr = t_day*std(choice_data)/sqrt(dpts);
schoice = sum(choice_data);
del_day = sum(del_data)./schoice;
del_std = zeros(1,days);
del_serr = zeros(1,days);
del_merr = zeros(1,days);
for j=1:days
    del_std(j) = std(del_data(choice_data(:,j)==1,j));
    del_serr(j) = del_std(j)/sqrt(schoice(j));
    del_t = tinv(0.975,schoice(j)-1);
    del_merr(j) = del_t*del_serr(j);
end

for k=1:Nblocks, for j=1:days
    choice_curr = choice_data((k-1)*tperday+1:k*tperday,j);
    choice_rewknock(k,j) = mean(choice_curr);
    del_curr = del_data((k-1)*tperday+1:k*tperday,j);
    del_rewknock(k,j) = mean(del_curr(choice_curr==1));
end
end

%
% plot choices across days and latency
%
figure(2), hold on, plot([1:days],choice_day,'g-o','linewidth',5);
hold on, errorbar([1:days],choice_day,choice_merr,'g','LineWidth',5);
set(gca,'fontsize',30);
xlabel('day','fontsize',30,'interpreter','latex');
ylabel('CS+ Hits','fontsize',30,'interpreter','latex');

figure(3), hold on, plot([1:days],del_day,'g-o','linewidth',5);
hold on, errorbar([1:days],del_day,del_merr,'g','LineWidth',5);
set(gca,'fontsize',30);
xlabel('day','fontsize',30,'interpreter','latex');
ylabel('CS+ Hit Latency (s)','fontsize',30,'interpreter','latex');
pause(1e-15);

%
% SHOCK EXPERIMENTS
%

% next focus on target trained activity weight parameters for shock
delgdc_train = 14;   % increment from a shock-cue to dop-glu
ssgdsc_train = 10;    % steady state after cue for dop-glu
delgds_train = 6;     % increment after shock to dop-glu
delndc_train = -2;   % increment after shock-cue for nonglu-dop
ssndsc_train = 0;     % steady state after cue for nonglu-dop
delnds_train = 0;     % increment from a reward to nonglu-dop

% last, let's focus on the drive of these to action selection
wg = 0.6; % weight given to glutamate trace
wd = 0.05; % weight given to dopamine trace


c = 0.05; % noise amplitude
h = 8.5;  % threshold value (single, determines latency)
taup = 250;


% activity variables
gds = zeros(nt_trial,1);  % glutamate-dopamine neuron activity
nds = zeros(nt_trial,1);  % no glu-dopamine neuron activity

% synaptic variables
sbd = zeros(nt_trial,1);  % dopamine output of glu-dop neuron
sbg = zeros(nt_trial,1);  % glutamate output of glu-dop neuron
snd = zeros(nt_trial,1);  % dopamine output of noglu-dop neuron

% action selection variable
choices = zeros(Ntrials,1);
del_times = zeros(Ntrials,1);

% now define all empty weight time series
wrs = zeros(6,nt);  wrs_trials = zeros(6,Ntrials);

% the associated input that will bring them to the appropriate trained val
wtrain = [delgdc_train; ssgdc_train; delgds_train;
    delndc_train; ssndc_train; delnds_train;];
Istim = taup*(1-exp(-(Ts+Td)/taup))/exp(-Td/taup)*wtrain;
Istim/taup

for k = 1:Ntrials

    ya = zeros(nt_trial,1);   % action selection variable (ddm)

    for j=1:nt_trial, ind = (k-1)*nt_trial+j;

        if j==1
            gds(j+1)=gds(j)+wrs(1,ind);
            nds(j+1)=nds(j)+wrs(4,ind);
        end

        if j>1 & j<round(Ts/dt)
            gds(j+1)=gds(j)+(gds(j)-wrs(2,ind))*(exp(-dt/taua)-1);
            nds(j+1)=nds(j)+(nds(j)-wrs(5,ind))*(exp(-dt/taua)-1);
        end
        
        wrs(:,ind+1) = wrs(:,ind)*exp(-dt/taup);

        if j==round(Ts/dt)
            wrs(:,ind+1) = wrs(:,ind+1)+Istim/taup;
            gds(j+1)=gds(j)+wrs(3,ind);
            nds(j+1)=nds(j)+wrs(6,ind);
        end

        if j>round(Ts/dt)
            gds(j+1)=gds(j)*exp(-dt/taua);
            nds(j+1)=nds(j)*exp(-dt/taua);
        end
        
        sbd(j+1) = max(sbd(j)+dt*(swd*gds(j)-sbd(j))/taus,0);
        sbg(j+1) = max(sbg(j)+dt*(swg*gds(j)-sbg(j))/taus,0);
        snd(j+1) = max(snd(j)+dt*(swnd*nds(j)-snd(j))/taus,0);

        if ya(j)<h & j<round(Ts/dt)
            ya(j+1) = ya(j)+dt*(wd*(sbd(j)+snd(j))+wg*sbg(j))+sqrt(dt*c)*randn;
            del_time = j*dt;
        else
            ya(j+1) = ya(j);
            if ya(j)>h, choices(k)=1; del_times(k) = del_time; end
        end

    end

    wrs_trials(:,k)=wrs(:,ind+1);

end


tplot = [-round(5/dt):nt_trial+1]*dt;
gds_plot = [zeros(round(5/dt)+1,1);gds];
nds_plot = [zeros(round(5/dt)+1,1);nds];
figure(11), hold on, plot(tplot,gds_plot,'color',[116 249 250]/256,'linewidth',5);
hold on, plot(tplot,nds_plot,'color',[16 0 238]/256,'linewidth',5);
set(gca,'fontsize',30);
xlabel('time (s)','fontsize',30,'interpreter','latex');
ylabel('activity','fontsize',30,'interpreter','latex');


snd_plot = [zeros(round(5/dt)+1,1);snd];
sbd_plot = [zeros(round(5/dt)+1,1);sbd];
sbg_plot = [zeros(round(5/dt)+1,1);sbg];
figure(21), hold on, plot(tplot,snd_plot,'color',[16 0 238]/256,'linewidth',5);
hold on, plot(tplot,sbd_plot,'color',[116 249 250]/256,'linewidth',5);
hold on, plot(tplot,sbg_plot,'color',[29 117 0]/256,'linewidth',5);
set(gca,'fontsize',30);
xlabel('time (s)','fontsize',30,'interpreter','latex');
ylabel('activity','fontsize',30,'interpreter','latex');
pause(1e-15);

% now make a matrix to contain data per data along the row
choice_data = zeros(dpts,days);
del_data = zeros(dpts,days);
t_day = tinv(0.975, dpts-1); % 0.975 corresponds to 95% confidence int
choice_shockcont = zeros(Nblocks,days);
del_shockcont = zeros(Nblocks,days);

for j=1:days
    choice_data(1:tperday,j) = choices((j-1)*tperday+1:j*tperday);
    del_data(1:tperday,j) = del_times((j-1)*tperday+1:j*tperday);
end

% and now let us repeat the sims Nblock-1 more times to get the rest of the data
for m=2:Nblocks

    gds = zeros(nt_trial,1);  % glutamate-dopamine neuron activity
    nds = zeros(nt_trial,1);  % no glu-dopamine neuron activity
    sbd = zeros(nt_trial,1);  % dopamine output of glu-dop neuron
    sbg = zeros(nt_trial,1);  % glutamate output of glu-dop neuron
    snd = zeros(nt_trial,1);  % dopamine output of noglu-dop neuron 
    choices = zeros(Ntrials,1);
    del_times = zeros(Ntrials,1);
    wrs = zeros(6,nt);

    for k = 1:Ntrials, del_time = 0; ya = zeros(nt_trial,1);

        for j=1:nt_trial, ind = (k-1)*nt_trial+j;

            if j==1, gds(j+1)=gds(j)+wrs(1,ind); nds(j+1)=nds(j)+wrs(4,ind); end

            if j>1 & j<round(Ts/dt)
                gds(j+1)=gds(j)+(gds(j)-wrs(2,ind))*(exp(-dt/taua)-1);
                nds(j+1)=nds(j)+(nds(j)-wrs(5,ind))*(exp(-dt/taua)-1);
            end
        
            wrs(:,ind+1) = wrs(:,ind)*exp(-dt/taup);

            if j==round(Ts/dt)
                wrs(:,ind+1) = wrs(:,ind+1)+Istim/taup;
                gds(j+1)=gds(j)+wrs(3,ind);
                nds(j+1)=nds(j)+wrs(6,ind);
            end

            if j>round(Ts/dt)
                gds(j+1)=gds(j)*exp(-dt/taua);
                nds(j+1)=nds(j)*exp(-dt/taua);
            end

            sbd(j+1) = sbd(j)+dt*(swd*gds(j)-sbd(j))/taus;
            sbg(j+1) = sbg(j)+dt*(swg*gds(j)-sbg(j))/taus;
            snd(j+1) = snd(j)+dt*(swnd*nds(j)-snd(j))/taus;

            if ya(j)<h & j<round(Ts/dt)
                ya(j+1) = ya(j)+dt*(wd*(sbd(j)+snd(j))+wg*sbg(j))+sqrt(dt*c)*randn;
                del_time = j*dt;
            else
                ya(j+1) = ya(j);
                if ya(j)>h, choices(k)=1; del_times(k) = del_time; end
            end

        end

    end

    for j=1:days
        choice_data((m-1)*tperday+1:m*tperday,j) = choices((j-1)*tperday+1:j*tperday);
        del_data((m-1)*tperday+1:m*tperday,j) = del_times((j-1)*tperday+1:j*tperday);
    end

end

choice_day = mean(choice_data);
choice_merr = t_day*std(choice_data)/sqrt(dpts);
schoice = sum(choice_data);
del_day = sum(del_data)./schoice;
del_std = zeros(1,days);
del_serr = zeros(1,days);
del_merr = zeros(1,days);
for j=1:days
    del_std(j) = std(del_data(choice_data(:,j)==1,j));
    del_serr(j) = del_std(j)/sqrt(schoice(j));
    del_t = tinv(0.975,schoice(j)-1);
    del_merr(j) = del_t*del_serr(j);
end

for k=1:Nblocks, for j=1:days
    choice_curr = choice_data((k-1)*tperday+1:k*tperday,j);
    choice_shockcont(k,j) = mean(choice_curr);
    del_curr = del_data((k-1)*tperday+1:k*tperday,j);
    del_shockcont(k,j) = mean(del_curr(choice_curr==1));
end
end


figure(5), hold on, plot([1:days],choice_day,'k-o','linewidth',5);
hold on, errorbar([1:days],choice_day,choice_merr,'k','LineWidth',5);
set(gca,'fontsize',30);
xlabel('day','fontsize',30,'interpreter','latex');
ylabel('CS+ Freeze','fontsize',30,'interpreter','latex');

figure(6), hold on, plot([1:days],del_day,'k-o','linewidth',5);
hold on, errorbar([1:days],del_day,del_merr,'k-','LineWidth',5);
set(gca,'fontsize',30);
xlabel('day','fontsize',30,'interpreter','latex');
ylabel('CS+ Freeze Latency (s)','fontsize',30,'interpreter','latex');
pause(1e-15);


wd = 0.0; % dopamine knockout
choice_shockknock = zeros(Nblocks,days);
del_shockknock = zeros(Nblocks,days);

% and now let us repeat the sims Nblock-1 more times to get the rest of the data
for m=1:Nblocks

    gds = zeros(nt_trial,1);  % glutamate-dopamine neuron activity
    nds = zeros(nt_trial,1);  % no glu-dopamine neuron activity
    sbd = zeros(nt_trial,1);  % dopamine output of glu-dop neuron
    sbg = zeros(nt_trial,1);  % glutamate output of glu-dop neuron
    snd = zeros(nt_trial,1);  % dopamine output of noglu-dop neuron 
    choices = zeros(Ntrials,1);
    del_times = zeros(Ntrials,1);
    wrs = zeros(6,nt);

    for k = 1:Ntrials, del_time = 0; ya = zeros(nt_trial,1);

        for j=1:nt_trial, ind = (k-1)*nt_trial+j;

            if j==1, gds(j+1)=gds(j)+wrs(1,ind); nds(j+1)=nds(j)+wrs(4,ind); end

            if j>1 & j<round(Ts/dt)
                gds(j+1)=gds(j)+(gds(j)-wrs(2,ind))*(exp(-dt/taua)-1);
                nds(j+1)=nds(j)+(nds(j)-wrs(5,ind))*(exp(-dt/taua)-1);
            end
        
            wrs(:,ind+1) = wrs(:,ind)*exp(-dt/taup);

            if j==round(Ts/dt)
                wrs(:,ind+1) = wrs(:,ind+1)+Istim/taup;
                gds(j+1)=gds(j)+wrs(3,ind);
                nds(j+1)=nds(j)+wrs(6,ind);
            end

            if j>round(Ts/dt)
                gds(j+1)=gds(j)*exp(-dt/taua);
                nds(j+1)=nds(j)*exp(-dt/taua);
            end

            sbd(j+1) = sbd(j)+dt*(swd*gds(j)-sbd(j))/taus;
            sbg(j+1) = sbg(j)+dt*(swg*gds(j)-sbg(j))/taus;
            snd(j+1) = snd(j)+dt*(swnd*nds(j)-snd(j))/taus;

            if ya(j)<h & j<round(Ts/dt)
                ya(j+1) = ya(j)+dt*(wd*(sbd(j)+snd(j))+wg*sbg(j))+sqrt(dt*c)*randn;
                del_time = j*dt;
            else
                ya(j+1) = ya(j);
                if ya(j)>h, choices(k)=1; del_times(k) = del_time; end
            end

        end

    end

    for j=1:days
        choice_data((m-1)*tperday+1:m*tperday,j) = choices((j-1)*tperday+1:j*tperday);
        del_data((m-1)*tperday+1:m*tperday,j) = del_times((j-1)*tperday+1:j*tperday);
    end

end

choice_day = mean(choice_data);
choice_merr = t_day*std(choice_data)/sqrt(dpts);
schoice = sum(choice_data);
del_day = sum(del_data)./schoice;
del_std = zeros(1,days);
del_serr = zeros(1,days);
del_merr = zeros(1,days);
for j=1:days
    del_std(j) = std(del_data(choice_data(:,j)==1,j));
    del_serr(j) = del_std(j)/sqrt(schoice(j));
    del_t = tinv(0.975,schoice(j)-1);
    del_merr(j) = del_t*del_serr(j);
end

for k=1:Nblocks, for j=1:days
    choice_curr = choice_data((k-1)*tperday+1:k*tperday,j);
    choice_shockknock(k,j) = mean(choice_curr);
    del_curr = del_data((k-1)*tperday+1:k*tperday,j);
    del_shockknock(k,j) = mean(del_curr(choice_curr==1));
end
end

figure(5), hold on, plot([1:days],choice_day,'b-o','linewidth',5);
hold on, errorbar([1:days],choice_day,choice_merr,'b','LineWidth',5);
set(gca,'fontsize',30);
xlabel('day','fontsize',30,'interpreter','latex');
ylabel('CS+ Freeze','fontsize',30,'interpreter','latex');

figure(6), hold on, plot([1:days],del_day,'b-o','linewidth',5);
hold on, errorbar([1:days],del_day,del_merr,'b-','LineWidth',5);
set(gca,'fontsize',30);
xlabel('day','fontsize',30,'interpreter','latex');
ylabel('CS+ Freeze Latency (s)','fontsize',30,'interpreter','latex');

% significance of difference b/w control and knockout experiments
% first the reward choice
Dcrew = pdist2(choice_rewcont,choice_rewknock,'euclidean');
Dcrewmean = mean(Dcrew(:));

N = 1e6;   % number of times to shuffle to get p val
Dcrewshufs = zeros(N,1);
crew_sers = [choice_rewcont;choice_rewknock];
for j=1:N
    crew_sers = crew_sers(randperm(size(crew_sers, 1)), :);
    Dcrewshufm = pdist2(crew_sers(1:10,:),crew_sers(11:end,:),'euclidean');
    Dcrewshufs(j)=mean(Dcrewshufm(:));
end

pval_crew = sum(Dcrewshufs > Dcrewmean)/N;
disp(['p value for the choice in the reward block is ',num2str(pval_crew)]);

% next the reward latency
Ddrew = pdist2(del_rewcont,del_rewknock,'euclidean');
Ddrewmean = mean(Ddrew(:));

N = 1e6;   % number of times to shuffle to get p val
Ddrewshufs = zeros(N,1);
drew_sers = [del_rewcont;del_rewknock];
for j=1:N
    drew_sers = drew_sers(randperm(size(drew_sers, 1)), :);
    Ddrewshufm = pdist2(drew_sers(1:10,:),drew_sers(11:end,:),'euclidean');
    Ddrewshufs(j)=mean(Ddrewshufm(:));
end

pval_drew = sum(Ddrewshufs > Ddrewmean)/N;
disp(['p value for the latency in the reward block is ',num2str(pval_drew)]);


% next the shock choice
Dcshock = pdist2(choice_shockcont,choice_shockknock,'euclidean');
Dcshockmean = mean(Dcshock(:));

N = 1e6;   % number of times to shuffle to get p val
Dcshockshufs = zeros(N,1);
cshock_sers = [choice_shockcont;choice_shockknock];
for j=1:N
    cshock_sers = cshock_sers(randperm(size(cshock_sers, 1)), :);
    Dcshockshufm = pdist2(cshock_sers(1:10,:),cshock_sers(11:end,:),'euclidean');
    Dcshockshufs(j)=mean(Dcshockshufm(:));
end

pval_cshock = sum(Dcshockshufs > Dcshockmean)/N;
disp(['p value for the choice in the shock block is ',num2str(pval_cshock)]);

% next the shock latency
Ddshock = pdist2(del_shockcont,del_shockknock,'euclidean');
Ddshockmean = mean(Ddshock(:));

N = 1e6;   % number of times to shuffle to get p val
Ddhshockshufs = zeros(N,1);
dshock_sers = [del_shockcont;del_shockknock];
for j=1:N
    dshock_sers = dshock_sers(randperm(size(dshock_sers, 1)), :);
    Ddshockshufm = pdist2(dshock_sers(1:10,:),dshock_sers(11:end,:),'euclidean');
    Ddshockshufs(j)=mean(Ddshockshufm(:));
end

pval_dshock = sum(Ddshockshufs > Ddshockmean)/N;
disp(['p value for the latency in the shock block is ',num2str(pval_dshock)]);

