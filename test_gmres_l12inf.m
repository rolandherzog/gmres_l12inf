% Set problem dimension
n = 2;

% Output some information
fprintf('-------------------------------------------------------------\n');
fprintf(' Problem size is n = %d.\n',n);

% Generate a random problem
% A = randn(n,n);
% b = randn(n,1);
% fprintf(' Random problem.\n');


% Generate a (symmetric) problem with sparse rhs
% (in the basis of eigenvectors)
% A = spdiags(randn(n,1),0,n,n);
% Q = randn(n,n); Q = orth(Q);
% A = Q * A * Q';
% b = randn(n,1);
% b(abs(b) > 0.5) = 0;
% fprintf(' Symmetric problem with %d-sparse rhs in eigenvector basis.\n',nnz(b));
% b = Q * b;


% Generate a (symmetric) problem with sparse solution
% (in the basis of eigenvectors)
% A = spdiags(randn(n,1),0,n,n);
% Q = randn(n,n); Q = orth(Q);
% A = Q * A * Q';
% xstar = randn(n,1);
% xstar(abs(xstar) > 0.5) = 0;
% fprintf(' Symmetric problem with %d-sparse solution in eigenvector basis.\n',nnz(xstar));
% xstar = Q * xstar;
% b = A * xstar;


% Generate a problem showing that no strict 
% decrease may happen until the last iterate for l1
% A = full(spdiags([ones(n,1), ones(n,1)], 0:1,n,n));
% b = zeros(n,1); b(end) = 1;
% fprintf(' Problem designed for l1-total stagnation.\n');


% Generate a problem showing that no strict 
% decrease may happen until the last iterate for linf
% TODO: This example seems broken
% A = full(spdiags([ones(n,1), [1:n]'], 0:1,n,n));
% b = zeros(n,1); b(end) = 1;
% fprintf(' Problem designed for linf-total stagnation.\n');


% Generate a problem with a non-symmetric, nearly orthogonal
% matrix
U = randn(n,n); U = orth(U);
V = randn(n,n); V = orth(V);
S = 1 + 0.01 * randn(n,1);
A = U * diag(S) * V';
b = randn(n,1);
fprintf(' Non-symmetric problem with singular values close to 1.\n');


% Generate a problem with a few distinct eigenvalues
% U = randn(n,n); 
% D = randn(n,1);
% D = round(D,0); D(D==0) = 1;
% A = U * diag(D) * inv(U);
% b = randn(n,1);
% fprintf(' Non-symmetric problem with %d distinct eigenvalues.\n',length(unique(D)));


% Set options 
options.norm = 'l2';
options.norm = 'linf';
options.norm = 'l1';
options.preserve_zero_residual_components = 0;  % only meaningful when options.norm == 'l1'
options.zero_residual_threshold = 1e-6;         % only meaningful when options.norm == 'l1' and options.preserve_zero_residual_components == 1

% Output some information
fprintf(' Residual is minimized w.r.t. %s norm.\n',options.norm);
if (strcmpi(options.norm,'l1'))
	if (options.preserve_zero_residual_components)
		fprintf(' Zero residual components WILL be preserved.\n');
	else
		fprintf(' Zero residual components will NOT be preserved.\n');
	end
end

% Set the LP solver (effective only if options.norm is 'l1' of 'inf)
options.solver = 'own';
options.solver = 'linprog';
if (strcmpi(options.norm,'l1') || strcmpi(options.norm,'linf'))
	fprintf(' Using %s LP solver.\n',options.solver);
end

% Call the method
rtol = 1e-12;
atol = 0;
[x,flag,resnorm,iter,X,R,V,H,LAMBDA,history] = gmres_l12inf(A,b,rtol,atol,[],[],options);
fprintf(' gmres_l12inf stopped at iteration %d with exitflag %d.\n',iter,flag);
fprintf('-------------------------------------------------------------\n');

% Produce a plot of the residual norms over iteration number
figure(1); clf, hold on
plot(history.gamma_l1,'bo-','LineWidth',2);
plot(history.gamma_l2,'r*-','LineWidth',2);
plot(history.gamma_linf,'ks-','LineWidth',2);
set(gca,'YScale','log');
legend('|r|_1','|r|_2','|r|_inf','Location','southwest');
title(sprintf('Various residual norms, going after %s',options.norm));
xlabel('iter');
grid on

% Produce a plot showing the sparsity pattern of the residual vector
figure(2); clf
spy(round(R,round(-log10(options.zero_residual_threshold))));
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

% Produce a plot showing the objective function |b - A x|_*
% in case n = 2
figure(6); clf; hold on
if (n == 2)
	xmin = min(X(1,:)); xmax = max(X(1,:)); 
	xmin = xmin - 0.3 * (xmax - xmin); 
	xmax = xmax + 0.3 * (xmax - xmin);
	ymin = min(X(2,:)); ymax = max(X(2,:));
	ymin = ymin - 0.3 * (ymax - ymin); 
	ymax = ymax + 0.3 * (ymax - ymin);
	nx = 41;
	[XX,YY] = meshgrid(linspace(xmin,xmax,nx),linspace(ymin,ymax,nx));
	ZZ = repmat(b,1,length(XX(:))) - A*[XX(:)'; YY(:)'];
	if (strcmpi(options.norm,'l1'))
		f = sum(abs(ZZ));
	elseif (strcmpi(options.norm,'l2'))
		f = sqrt(sum(ZZ.^2));
	elseif (strcmpi(options.norm,'linf'))
		f = max(abs(ZZ));
	end
	f = reshape(f,nx,nx);
	contour(XX,YY,f);
	plot(X(1,:),X(2,:),'r-o','MarkerSize',10,'MarkerFaceColor','r');
	title('Level lines of the objective, and iterates');
	axis equal
end

