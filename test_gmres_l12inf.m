% Generate a random problem
n = 40;
A = randn(n,n);
b = randn(n,1);

% Set options and call the method
options.norm = 'l2';
options.norm = 'linf';
options.norm = 'l1';

[x,flag,resnorm,iter,X,R,V,H,history] = gmres_l12inf(A,b,[],[],[],[],options);

% Produce a plot
figure(1); clf, hold on
plot(history.gamma_l1,'bo-','LineWidth',2);
plot(history.gamma_l2,'r*-','LineWidth',2);
plot(history.gamma_linf,'ks-','LineWidth',2);
set(gca,'YScale','log');
legend('|r|_1','|r|_2','|r|_inf');

title(sprintf('Various residual norms, going after %s',options.norm));
xlabel('iter');
grid on


