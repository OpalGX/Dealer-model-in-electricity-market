addpath('/home/gong0022/code')
read(symengine, '/home/gong0022/code/supply_function.mu')
% avesigma = zeros(10, 1);
% for i1 = 0:0.05:0.9
tic
i1=0.7
for i2=40
A=i1; % Amplitude of sin noise
sigmapd = i2;
% I = 1; % number of interation
N = 10; % number of supplier/
%deltaPs = i5; % step size of supply
% deltaPd = 1; % stef size of demand
deltaT = 1; % step size of time
tf =36000; % number of auction in one iteration
ps0 = 100; % decay length of supply curve
ps1 = 1; % the second piece
pd0 = 80; % decay length of demad curve
sigmaps = 0;
t0 = 0;
sigmat0 = 0;
D = 20; % drifting constantdd
M = 10; % moving avarage
weights=1:1:M; %weight of moving average
pc = 25;
s0=1;
s1=1;
p = zeros(tf, 1); 
% v = zeros(tf, 1); 
d = 0;gapsize = 3; 

 %sigma = zeros(I,1);
 %avep = zeros(I,1);
% for iter = 1:I
demand = zeros(tf, 1); 
p = zeros(tf, 1); 
pp = zeros(tf, 1);
ema=0;
for t = 1: tf
    ema
    s = 0;
    d = 0;
    if t == 1
        % Initialize the parameter
        ps=ps0*ones(N,1);
        pd=pd0*ones(N,1);
    
    
    elseif t <= M + 1
           ema = 0;
    else
            delP = pp - p;
            emats = tsmovavg(delP, 'w', weights, 1);
           ema = round (1000*emats(t - 2))/1000;
            
    end
        for j = 1 : N
              % ps(j) = ps(j)+ (2*(rand<0.5)-1)*deltaPs ;            
               % ps(j) = ps(j)+ (2*(rand<0.5)-1)*deltaPs + D*deltaT*ema ;
               ps(j) = lognrnd(log(ps0),sigmaps);
               % ps(j) = ps(j)+ (2*(rand<0.5)-1)*deltaPs + D*deltaT*ema ;            
               % pd(j) = pd(j)+ (2*(rand<0.5)-1)*deltaPd;
               pdr = lognrnd(log(pd0),sigmapd);
               t0r = normrnd(t0,sigmat0);
               % pd(j) = pdr;
               pd(j) = pdr*(1+A*sin(pi*(t-t0r)/12));      

               %pd(j) = max(pdr*(1+A*sin(pi*(t-t0r)/12))+ D*deltaT*ema,0.01);      
        end
        
    syms x
    for i = 1:N
            si = feval(symengine, 'supply_function', pc,ps(i),ps1,s0,s1);
            % si = feval(symengine, 'supplyFn', A,pc,k,s0);
            s = s + si;
    end
    
    for i = 1:N
            di = 2*exp(-x/pd(i));
            d = d + di;
    end
    % d = round (10*(N*(s0+s1)+s1*N*sin(t*pi/12)+wgn (1,1,0)/5))/10
    % Calculate the equilibrium price
    % eqn = s - d + 6
    eqn = s - d;
    answer = 0;
    try
       answer = vpasolve(eqn,x,[1,600]);
       
    end
    
    % x = vpasolve(func,x,120)
    sizeanswer = size(answer);
    if sizeanswer(1)==1
        p(t)=answer;
        %v(t)= subs(d,answer);
    else
        p(t)=0;
        eqn
    end
    
    
    
    % Calculate the average supply parameter
    
    if t>1
        pp(t-1) = p(t);
    end
 %   p
%    pp
% end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot price
f = figure();
plot(p);
axis([0,tf,0,100])
xlabel('Time')
ylabel('Price')
% filename = '/home/gong0022/output/filenametest.jpg';
filename = sprintf('/home/gong0022/output/Mar23_price_ts_pd0 _ % d_sigmapd _ % .2f_sigmaps _ 0_A _ % .2f_pc _ % d_ % d_run.jpg', pd0,sigmapd,sigmaps,pc,tf);
saveas(f, filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%[pks,locs,w,pr] = findpeaks(p);
% nty = prctile(pr,90);
% maxpr = max(pr);
%[counts,centers] = hist(pr,ceil(maxpr)); % use constant bin size
% h = figure;
% hist(pr)
% filename = sprintf('prominence_pd0 _ % d_sigmapd _ % d_sigmaps _ % .2f_A _ % .2f_pc _ % d_ % d_run.png', pd0,sigmapd,sigmaps,A, pc,tf);
% saveas(h, filename, 'jpg')
% close(h)
% locgap = find(conv(double(counts==0),ones(1,gapsize),'valid')==gapsize); %// find n zeros
% deal with 
% if numel(locgap)== 0
%    firstgap = 0;
%    num_spike = 0;
% else 
% firstgap = locgap(1);
%[m,n] = size(counts);
% num_spike = sum(counts(firstgap:n));
% end
% peakrate = num_spike/tf;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot volume
% f = figure()
% plot(p);
% axis([0,tf,0,100])
% xlabel('Time')
% ylabel('Price')
% filename = sprintf('volume_pd0 _ % d_sigmapd _ % d_sigmaps _ 0_A _ % .2f_pc _ % d_ % d_run.png', pd0,sigmapd,sigmaps,pc,tf);
% saveas(f, filename)
% close(f)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% save peakrate info
% openfile= sprintf('spike_rate _Mar11 _pd0 _ % d_ps0 _ % d_sigmat0 _ % d_sigmaps _ % d_gapsize _ % d_ % d_run.txt', pd0, ps0, sigmat0, sigmaps, gapsize, tf);
% fid = fopen(openfile, 'a')
% fprintf(fid, '% .2f % .2f % .4f \r \n', i1, i2, peakrate);
% fclose(fid);

% save top 10% percentile data
% openfile= sprintf('prominence_ 90_percentile _Mar11 _pd0 _ % d_ps0 _ % d_sigmat0 _ % d_sigmaps _ % d_gapsize _ % d_ % d_run.txt', pd0, ps0, sigmat0, sigmaps, gapsize, tf);
% fid = fopen(openfile, 'a')
% fprintf(fid, '% .2f % .2f % .4f \r \n', i1, i2, nty);
% fclose(fid);


% save standard deviation
% openfile= sprintf('standard deviation_Mar11 _pd0 _ % d_ps0 _ % d_sigmat0 _ % d_sigmaps _ % d_gapsize _ % d_ % d_run.txt', pd0, ps0, sigmat0, sigmaps, gapsize, tf);
% fid = fopen(openfile, 'a')
% fprintf(fid, '% .2f % .2f % .4f % .4f \r \n', i1, i2, std(p), std(pr));
% fclose(fid);


        % end
    % end
    
% plot price
% f = figure()
% plot(p);
% axis([0,tf,0,100])
% filename = sprintf('price_ps1=% d.png', r);
% saveas(f, filename)
% close(f)

% plot distribution
% xRange = 0:0.5:100;                % # Range of integers to compute a probability for
% H = hist(p,xRange);        % # Bin the data
% f = semilogy(xRange,H./numel(p));  % # Plot the probabilities for each integer
% xlabel('Price');
% ylabel('Probability');
% filename = sprintf('price_histogram _A _ %0 .2f_sigmapd _ % d_ % d_run.png', A, sigmapd, tf);
% saveas(f, filename)

% plot volumn distribution
% xRange = 0:0.2:100;                % # Range of integers to compute a probability for
% H = hist(v,xRange);        % # Bin the data
% f = semilogy(xRange,H./numel(p));  % # Plot the probabilities for each integer
% xlabel('Volumn');
% ylabel('Probability');
% filename = sprintf('volumn_histogram _ % d_run.png', tf);
% saveas(f, filename)


% plot(demand)
% axis([0,tf,0,30])
% sigma(iter) = std(p);
% avep(iter) = mean(p);

% msigma = mean(sigma)
% mp = mean(avep)
% avesigma(r) = msigma
filename = sprintf('/home/gong0022/output/April12_drift_10_simulation_sigmapd _  %.2f_A _  %.2f_sigmat0=0_%d_run.mat', i2, i1,tf);
save(filename, 'pd0', 'ps0', 'sigmat0', 'sigmaps', 'gapsize','tf','p', 'f','sigmat0','M','D');
%close(f)


runtime=toc
save(filename,'runtime','-append')
end
