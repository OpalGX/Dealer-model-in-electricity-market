function phase_transition(sigma_pd, A)
filename = sprintf('C:/Users/Xue/Documents/MATLAB/cluster output/April12_drift_10_simulation_sigmapd _  %.2f_A _  %.2f_sigmat0=0_10000_run.mat', sigma_pd, A)
%filename = sprintf('C:/Users/Xue/Documents/MATLAB/cluster output/April8_simulation_sigmapd _  40.00_A _  0.70_sigmat0=0_100000_run.mat')
load(filename,'p')

figure3=figure(3);
plot(p);

[pks,locs,w,pr] = findpeaks(p);
[n,xout] = hist(pr,ceil(max(pr)))
x=transpose(xout)
figure1=figure(1);
exclude=[];
%try
exclude= (((xout>20)&(xout<30))|((xout>60)&(xout<70))|((xout>100)&(xout<150)))
%end
exclude
figure1=figure(1);
f2=fit(x,transpose(100*n/sum(n)),'exp1','Exclude', exclude);
plot(f2,xout,100*n/sum(n));
xlabel('Prominence(Euro/KWh)')
ylabel('Frequency(%)')

figure2=figure(2);
x=1:1:ceil(max(pr));
pr2=100*n/sum(n)-transpose(f2(x));
plot(x, pr2);
xlabel('Prominence(Euro/KWh)')
ylabel('Frequency(%)')
[height,location] = findpeaks(pr2);

pk1=0;pk2=0;pk3=0;
try
pk1=max(pr2(20:30));
end
try
pk2=max(pr2(60:70));
end
try
pk3=max(pr2(100:150));
end

filename = sprintf('C:/Users/Xue/Documents/MATLAB/phase_transition_2/drift_10_sigmapd_%.2f_A_%.2f_sigmat0_0.mat', sigma_pd, A);
save(filename, 'A','sigma_pd','height','location','f2','figure1','figure2','figure3', 'pr2','pk1','pk2','pk3')
close(figure1)
close(figure2)

end