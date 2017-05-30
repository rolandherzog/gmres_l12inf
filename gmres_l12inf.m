function [x,flag,resnorm,iter,X,R,V,H,LAMBDA,history] = gmres_l12inf(A,b,rtol,atol,maxiter,x0,options)
% GMRES   Generalized Minimum Residual Method 
%         with l1, l2, or linf norm residual minimization.
%
% This method solves the square linear system
% A x = b in an iterative way by
% * building a 2-orthonormal basis of the Krylov subspaces K_k(A;r0)
% * finding x_k \in x_0 + K_k(A;r0) which minimizes |b - A x_k|_*
%   where |.|_* is either the l1, l2, or linf norm.
%
% To select the norm, set options.norm to 'l1', 'l2', 'linf' respectively.
%
% The method stops as soon as as |rk|_* <= rtol |r0|_* or |rk|_* <= atol holds.
%
% No preconditioning is currently applied.
% 
% The output components are:
% x       - approximate solution at final iterate
% flag    - 0 (converged), -1 (negative entries in H), 1 (max # of iterations reached)
% resnorm - |r|_* at final iterate
% iter    - # of iterations performed
% X       - full iterate history matrix
% R       - full residual history matrix
% V       - full Krylov basis vector matrix
% H       - Hessenberg matrix generated in Arnoldi process
% history - a history of certain iteration dependent scalars

% Determine whether the matrix A is a matrix or function
% and wrap it into a function if necessary
if (isnumeric(A))
	A = @(v) A*v;
end

% Determine the dimension of the problem
n = length(b);

% Check and assign relative tolerance
if (nargin < 3) || isempty(rtol)
	rtol = 1e-6;
end

% Check and assign absolute tolerance
if (nargin < 4) || isempty(atol)
	atol = 1e-6;
end

% Check and assign maximum # of iterations
if (nargin < 5) || isempty(maxiter)
	maxiter = 30;
end

% Check and assign initial guess
if (nargin < 6) || isempty(x0)
	x0 = zeros(n,1);
end

% Select the algorithm
if (nargin < 7) || ~isfield(options,'norm')
	options.norm = 'l1';
end


% Parse the output arguments
if (nargout >= 5), need_X = 1; end
if (nargout >= 6), need_R = 1; end
if (nargout >= 7), need_V = 1; end
if (nargout >= 8), need_H = 1; end
if (nargout >= 9), need_LAMBDA = 1; end

% Initialize counters and flags
iter = 1;
done = 0;

% Get initial residual
r0 = b - A(x0);

% Initialize the sequence of iterates and residuals
if (need_X), X = x0; end
if (need_R), R = r0;  end
if (need_LAMBDA), LAMBDA = [];  end
history = [];

% Calculate initial residual norm (in all of l1, l2, linf)
% gamma_l1 = mynorm(r,'l1');
% gamma_l2 = mynorm(r,'l2');
% gamma_linf = mynorm(r,'linf');
history.gamma_l1 = mynorm(r0,'l1');
history.gamma_l2 = mynorm(r0,'l2');
history.gamma_linf = mynorm(r0,'linf');

% Remember the initial *-norm of the residual to evaluate the relative stopping criterion
gamma = mynorm(r0,options.norm);   
gamma0 = gamma;

% Set initial Krylov space basis vector
% Note: Krylov vectors will be 2-orthonormal
V(:,1) = r0 / history.gamma_l2(end);


% Begin loop
while (~done)

	% Check residual norm for convergence
	if (gamma <= rtol * gamma0) || (gamma <= atol)
		flag = 0;
		done = 1;
	end
	
	% Perform the iteration
	if (~done)

		% Evaluate the matrix times the most recent basis vector
		Av = A(V(:,iter));
		
		% Store the preliminary basis vector
		V(:,iter+1) = Av;

		% Calculate orthogonalization coefficients 
		H(1:iter,iter) = V(:,1:iter)'*V(:,iter+1);

		% Orthogonalize the most recent basis vector
		V(:,iter+1) = V(:,iter+1) - V(:,1:iter)*H(1:iter,iter);

		% Store the 2-norm 
		H(iter+1,iter) = norm(V(:,iter+1),2);

		% Sanity check for imaginary entries in H
		if (any(imag(H(:))))
			done = 1;
			flag = -1;
		end

		% Normalize the most recent basis vector
		V(:,iter+1) = V(:,iter+1) / H(iter+1,iter);

		% Construct the current iterate by minimizing |b - A x_k|_*
		if (strcmpi(options.norm,'l1'))

			% Solve an LP to find the expansion coefficients for xk - x0
			linprogoptions = optimoptions(@linprog,'display','off');
			c = [zeros(iter,1); ones(n,1)];
			[yt,primalval,~,~,lambda] = linprog(c,[[A(V(:,1:iter)),-eye(n)];[-A(V(:,1:iter)),-eye(n)]],[r0;-r0],[],[],[],[],[],linprogoptions); 
			y = yt(1:iter); t = yt(iter+1:end); 
			x = x0 + V(:,1:iter) * y;
			% Note: The objective value primalval coincides with mynorm(r,'l1') below

		elseif (strcmpi(options.norm,'l2'))

			% Solve a least-squares problem to find the expansion coefficients for xk - x0
			e = zeros(iter+1,1); e(1) = 1;
			y = H(1:iter+1,1:iter) \ (gamma0 * e);
			x = x0 + V(:,1:iter) * y;

		elseif (strcmpi(options.norm,'linf'))

			% Solve an LP to find the expansion coefficients for xk - x0
			linprogoptions = optimoptions(@linprog,'display','off');
			c = [zeros(iter,1); 1];
			[yt,primalval,~,~,lambda] = linprog(c,[[A(V(:,1:iter)),-ones(n,1)];[-A(V(:,1:iter)),-ones(n,1)]],[r0;-r0],[],[],[],[],[],linprogoptions); 
			y = yt(1:iter); t = yt(iter+1:end);
			x = x0 + V(:,1:iter) * y;
			% Note: The objective value primalval coincides with mynorm(r,'linf') below

		end

		% Evaluate the residual and its l1-, l2-, linf-norms
		% (This could probably be made more economical.)
		r = b - A(x);
		history.gamma_l1 = [history.gamma_l1; mynorm(r,'l1')];
		history.gamma_l2 = [history.gamma_l2; mynorm(r,'l2')];
		history.gamma_linf = [history.gamma_linf; mynorm(r,'linf')];

		% Store also the *-norm of the current residual
		gamma = mynorm(r,options.norm);   

		% Store current iterate and residual, if necessary
		if (need_X), 
			X = [X x]; 
		end
		if (need_R)
			R = [R r];
		end
		if (need_LAMBDA && (strcmpi(options.norm,'l1') || strcmpi(options.norm,'linf')))
			LAMBDA = [LAMBDA lambda.ineqlin(1:n)-lambda.ineqlin(n+1:end)];
		end

		% Check for maximum # of iterations
		if (iter < maxiter)
			iter = iter + 1;
		elseif (~done)
			flag = 1;
			done = 1;
		end

	end % if (~done)

end % while (~done)

% Finalize the output
resnorm = gamma;


% Evaluate the (residual-type) norm of r
% depending on the choice of the user
function val = mynorm(r,normstring)
if (strcmpi(normstring,'l1'))
	val = norm(r,1);
elseif (strcmpi(normstring,'l2'))
	val = norm(r,2);
elseif (strcmpi(normstring,'linf'))
	val = norm(r,inf);
else
	error('Norm must be l1, l2, or linf.\n');
end


