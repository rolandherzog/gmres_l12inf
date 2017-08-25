% Set problem dimension
n = 10;

% Set problem type (see below)
problem.type = 5;

% Output some information
fprintf('-------------------------------------------------------------\n');
fprintf(' Problem size is n = %d.\n',n);
fprintf(' Problem type is %d.\n',problem.type);

if (problem.type == 1)

	% Generate a random problem
	A = randn(n,n);
	b = randn(n,1);
	fprintf(' Random problem.\n');

elseif (problem.type == 2)

	% Generate a (symmetric) problem with sparse rhs
	% (in the basis of eigenvectors)
	A = spdiags(randn(n,1),0,n,n);
	Q = randn(n,n); Q = orth(Q);
	A = Q * A * Q';
	b = randn(n,1);
	b(abs(b) > 0.5) = 0;
	fprintf(' Symmetric problem with %d-sparse rhs in eigenvector basis.\n',nnz(b));
	b = Q * b;

elseif (problem.type == 3)

	% Generate a (symmetric) problem with sparse solution
	% (in the basis of eigenvectors)
	A = spdiags(randn(n,1),0,n,n);
	Q = randn(n,n); Q = orth(Q);
	A = Q * A * Q';
	xstar = randn(n,1);
	xstar(abs(xstar) > 0.5) = 0;
	fprintf(' Symmetric problem with %d-sparse solution in eigenvector basis.\n',nnz(xstar));
	xstar = Q * xstar;
	b = A * xstar;

elseif (problem.type == 4)

	% Generate a problem showing that no strict 
	% decrease may happen until the last iterate for l1
	A = full(spdiags([ones(n,1), ones(n,1)], 0:1,n,n));
	b = zeros(n,1); b(end) = 1;
	fprintf(' Problem designed for l1-total stagnation.\n');

elseif (problem.type == 5)

	% Generate a problem showing that no strict 
	% decrease may happen until the last iterate for linf
	% A = full(spdiags([ones(n,1), [1:n]'], 0:1,n,n));
	A = full(spdiags([ones(n,1), ones(n,1)], 0:1,n,n));
	b = (-1).^[1:n]';
	fprintf(' Problem designed for linf-total stagnation.\n');

elseif (problem.type == 6)

	% Generate a problem with a non-symmetric, nearly orthogonal
	% matrix
	U = randn(n,n); U = orth(U);
	V = randn(n,n); V = orth(V);
	S = 1 + 0.01 * randn(n,1);
	A = U * diag(S) * V';
	b = randn(n,1);
	fprintf(' Non-symmetric problem with singular values close to 1.\n');

elseif (problem.type == 7)

	% Generate a problem with a few distinct eigenvalues
	U = randn(n,n); 
	D = randn(n,1);
	D = round(D,0); D(D==0) = 1;
	A = U * diag(D) * inv(U);
	b = randn(n,1);
	fprintf(' Non-symmetric problem with %d distinct eigenvalues.\n',length(unique(D)));

elseif (problem.type == 8)

	% Generate a block-diagonal problem
	A = diag(1:n);
	m = round(n/2);
	[V,D] = eig(A(1:m,1:m));
	b = V(:,1);
	b = [b; randn(n-m,1)];
	fprintf(' Diagonal problem with upper portion trivial to solve.\n');

elseif (problem.type == 9)

	% Generate discrete ill-posed problem deriv2 as used in 
	% Clason, Jin, Kunisch: A semismooth Newton method for $L^1$ data fitting 
	% with automatic choice of regularization parameters and noise calibration
	% Code taken from l1fitting_test.m accompanying the paper, available at
	% https://www.uni-due.de/~adf040p/codes/l1fitting.zip. It uses
	% deriv2 from Regularization Tools: http://www2.imm.dtu.dk/~pch/Regutools/
	addpath('./regtools');
	d_mag = 1.0;             % magnitude of noise
	d_per = 0.3;             % percentage of corrupted data points
	[A,ye,xe] = deriv2(n,3); % operator A, exact data ye, reference solution xe
	drnd = rand(n,1); ind = find(drnd<d_per); imprnd = zeros(n,1); 
	imprnd(ind) = randn(length(ind),1); noise = d_mag*max(abs(ye))*imprnd; % add impulse noise to data
	b = ye + noise;  % noisy data
	% Consider additional l1 regularization as well
	lambda = 2e-2;
	A = [A; lambda*eye(n)];
	Pinv = pinv(A);
	Pinv = A';
	A = Pinv * A;
	b = Pinv * [b; zeros(n,1)];
	fprintf(' Ill-posed l1-data fitting problem deriv2 from Clason, Jin, Kunisch.\n');

elseif (problem.type == 10)

	% Convection diffusion example from Chen
	% [beta; gamma] is the convection direction
	beta = 0.5;
	gamma = 0.5;
	nn = round(sqrt(n));
	h=1./(nn+1);
	a=4; b=-1-gamma; c=-1-beta; d=-1+beta; e=-1+gamma;
	I=speye(nn);
	t1=spdiags([c*ones(nn,1),2*ones(nn,1),d*ones(nn,1)],-1:1,nn,nn);
	t2=spdiags([b*ones(nn,1),2*ones(nn,1),e*ones(nn,1)],-1:1,nn,nn);
	A=kron(I,t1)+kron(t2,I);
	n = nn^2;
	b = randn(n,1);

else

	fprintf(' Unknown problem type. Exiting.\n')
	return

end

% PRepare some plots
figure(1); clf, hold on

% Loop over all possible norms
% for norm_string = { 'l1', 'l2', 'linf' }
% for norm_string = { 'linf', 'l2', 'l1' }
for norm_string = { 'linf' }

	% Set options
	options.norm = norm_string{1};
	options.preserve_zero_residual_components = 0;  % only meaningful when options.norm == 'l1'
	options.zero_residual_threshold = 1e-6;         % only meaningful when options.norm == 'l1' and options.preserve_zero_residual_components == 1

	% Output some information
	fprintf('-------------------------------------------------------------\n');
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
	rtol = 1e-6;
	atol = [];
	maxiter = [];
	[x,flag,resnorm,iter,X,R,V,H,LAMBDA,history] = gmres_l12inf(A,b,rtol,atol,maxiter,[],options);
	fprintf(' gmres_l12inf stopped at iteration %d with exitflag %d.\n',iter,flag);

	% Produce a plot of the residual norms over iteration number
	figure(1);
	if (strcmpi(options.norm,'l1'))
		plot(history.gamma_l1,'bo-','LineWidth',2);
	elseif (strcmpi(options.norm,'l2'))
		plot(history.gamma_l2,'r*-','LineWidth',2);
	elseif (strcmpi(options.norm,'linf'))
		plot(history.gamma_linf,'ks-','LineWidth',2);
	end
	set(gca,'YScale','log');

end % for = { 'l1', 'l2', 'linf' }
fprintf('-------------------------------------------------------------\n');

% Finalize some figures
figure(1);
legend('|r|_1','|r|_2','|r|_inf','Location','southwest');
title(sprintf('Various residual norms when going after each one'));
xlabel('iter');
grid on


% Produce a plot showing the sparsity pattern of the residual vector
% figure(2); clf
% spy(round(R,round(-log10(options.zero_residual_threshold))));
% title('Residual sparsity pattern over iterations');
% xlabel('iter');

% Produce a plot showing where the residual vector attains its maximum 
% figure(3); clf
% if (~isempty(LAMBDA))
% 	spy(round(LAMBDA,4));
% else
% 	[ix,jx] = find(abs(abs(R) - repmat(max(abs(R)),length(x),1)) < 1e-5);
% 	tmp = sparse(ix,jx,1);
% 	spy(tmp);
% end
% title('Coordinates where the residual attains its maximum');
% xlabel('iter');

% Produce a plot confirming that V'*A'*lambda is strictly lower triangular
% figure(4); clf
% if (~isempty(LAMBDA))
% 	spy(round(V'*A'*LAMBDA,4));
% 	title('Verifying the strict lower triangular property');
% 	xlabel('iter');
% end

% Produce a plot showing the number of inner LP iterations, if available
% figure(5); clf; hold on
% if (~isempty(history.lpiter))
% 	plot(history.lpiter,'bo');
% 	plot(mean(history.lpiter)* ones(1,iter-1),'r');
% 	axis([0, iter, 1, max(history.lpiter)+1]);
% 	title('Number of inner LP iterations');
% 	xlabel('iter');
% end
% grid on

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

