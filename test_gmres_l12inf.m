% Generate a random problem
n = 20;
A = randn(n,n);
b = randn(n,1);

% Generate a problem showing that no strict 
% decrease may happen until the last iterate for l1
% A = full(spdiags([ones(n,1), ones(n,1)], 0:1,n,n));
% b = zeros(n,1); b(end) = 1;

% Generate a problem showing that no strict 
% decrease may happen until the last iterate for linf
% A = full(spdiags([ones(n,1), [1:n]'], 0:1,n,n));
% b = zeros(n,1); b(end) = 1;


% Set options 
options.norm = 'l2';
options.norm = 'linf';
options.norm = 'l1';

% Set the LP solver (effective only if options.norm is 'l1' of 'inf)
options.solver = 'own';
options.solver = 'linprog';

% Call the method
[x,flag,resnorm,iter,X,R,V,H,LAMBDA,history] = gmres_l12inf(A,b,[],[],[],[],options);

% Produce a plot of the residual norms over iteration number
figure(1); clf, hold on
plot(history.gamma_l1,'bo-','LineWidth',2);
plot(history.gamma_l2,'r*-','LineWidth',2);
plot(history.gamma_linf,'ks-','LineWidth',2);
set(gca,'YScale','log');
legend('|r|_1','|r|_2','|r|_inf');
title(sprintf('Various residual norms, going after %s',options.norm));
xlabel('iter');
grid on

% Produce a plot showing the sparsity pattern of the residual vector
figure(2); clf
spy(round(R,3));
title('Residual sparsity pattern over iterations');
xlabel('iter');

% Produce a plot showing where the residual vector attains its maximum 
figure(3); clf
if (~isempty(LAMBDA))
	spy(round(LAMBDA,4));
else
	[ix,jx] = find(abs(abs(R) - repmat(max(abs(R)),length(x),1)) < 1e-5);
	tmp = sparse(ix,jx,1);
	spy(tmp);
end
title('Coordinates where the residual attains its maximum');
xlabel('iter');

% Produce a plot confirming that V'*A'*lambda is strictly lower triangular
figure(4); clf
if (~isempty(LAMBDA))
	spy(round(V'*A'*LAMBDA,4));
	title('Verifying the strict lower triangular property');
	xlabel('iter');
end

% Produce a plot showing the number of inner LP iterations, if available
figure(5); clf; hold on
if (~isempty(history.lpiter))
	plot(history.lpiter,'bo');
	plot(mean(history.lpiter)* ones(1,iter-1),'r');
	axis([0, iter, 1, max(history.lpiter)+1]);
	title('Number of inner LP iterations');
	xlabel('iter');
end
grid on

