function hysteresis(M)

filename = sprintf('C:/Users/Xue/Documents/MATLAB/cluster output/April12_drift_10_simulation_sigmapd _  40.00_A _  0.70_sigmat0=0_36000_run.mat')
%filename = sprintf('C:/Users/Xue/Documents/MATLAB/cluster output/Mar29_simulation_sigmapd _  40.00_A _  0.70_sigmat0=0_36000_run.mat')
load(filename,'p')
read(symengine, 'C:\Users\Xue\Documents\MATLAB\FYP Code\supply_function.mu')
p=p(1:3000);
sum(p(:)==0);
v=zeros(size(p));
peaks=zeros(size(p));


[pks,locs,w,pr] = findpeaks(p,'MinPeakProminence',10);
si = feval(symengine, 'supply_function', 25,100,1,1,1);
s= 10*si;

for i=1:size(p)
    v(i)= subs(s,p(i));
end
moveavev = tsmovavg(v, 's', M, 1)

figure1=figure(1);
locs1 = locs((pr>=20)&(pr<=30));
peaks1=peaks;
peaks1(ceil(locs1))=1;
movaverate1 = tsmovavg(peaks1, 's', M, 1);
plot(moveavev,movaverate1);
xlabel('Average Demand');
ylabel('Average Spike Rate');


figure2=figure(2);
locs2 = locs((pr>=60)&(pr<=70));
peaks2=peaks;
peaks2(ceil(locs2))=1;
movaverate2 = tsmovavg(peaks2, 's', M, 1);
plot(moveavev,movaverate2);
xlabel('Average Demand');
ylabel('Average Spike Rate');


figure3=figure(3);
locs3 = locs((pr>=100)&(pr<=150));
peaks3=peaks;
peaks3(ceil(locs3))=1;
movaverate3 = tsmovavg(peaks3, 's', M, 1);
plot(moveavev,movaverate3);
xlabel('Average Demand');
ylabel('Average Spike Rate');


figure4=figure(4)
peaks(ceil(locs))=1;
movaverate = tsmovavg(peaks, 's', M, 1)
moveavev = tsmovavg(v, 's', M, 1)
plot(moveavev,movaverate)
xlabel('Average Demand')
ylabel('Average Spike Rate')


end
