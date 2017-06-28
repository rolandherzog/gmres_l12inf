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
% To select the solver, set 
%   options.solver to 'linprog' (effective only if options.norm is 'l1' of 'inf)
%   options.solver to 'own'     (effective only if options.norm is 'l1' of 'inf).
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
	maxiter = n+1;
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

% Initialize 'previous' LP solution (with u non-existent), and corresponding basis
% to be able to provide an initial guess to our own, customized LP solver
if (strcmpi(options.solver,'own'))
	if (strcmpi(options.norm,'l1'))
		sp = max(-2*r0,0);
		sm = max(+2*r0,0);
		t = r0 + sp;
	elseif (strcmpi(options.norm,'linf'))
		t = max(abs(r0));
		sp = max(t-r0,0);
		sm = max(t+r0,0);
	end
	tspmupm = [t; sp; sm];
	B = find(tspmupm);
	B = B(1:2*n);
end


% Initialize the sequence of iterates and residuals
if (need_X), X = x0; end
if (need_R), R = r0;  end
if (need_LAMBDA), LAMBDA = [];  end
history = [];
history.lpiter = [];

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
			if (strcmpi(options.solver,'linprog'))

				% Solve using linprog (without an initial guess)
				linprogoptions = optimoptions(@linprog,'display','off','algorithm','interior-point');
				c = [zeros(iter,1); ones(n,1)];
				Aineq = [[A(V(:,1:iter)),-eye(n)];[-A(V(:,1:iter)),-eye(n)]];
				bineq = [r0;-r0];
				[ut,primalval,exitflag,output,lambda] = linprog(c,Aineq,bineq,[],[],[],[],[],linprogoptions); 
				history.lpiter = [history.lpiter; output.iterations];
				assert(exitflag == 1,'gmres_l12inf: linprog terminated with exitflag %d.',exitflag);

				% Postprocess the solution
				u = ut(1:iter); t = ut(iter+1:end); 
				x = x0 + V(:,1:iter) * u;
				% Note: The objective value primalval coincides with mynorm(r,'l1') below

			elseif (strcmpi(options.solver,'own'))

				% Solve using our own customized LP solver (with initial guess)
				tspmupm0 = zeros(3*n+2*iter,1);
				tspmupm0(1:3*n) = tspmupm(1:3*n);                           % copy the t, sp, sm iterates
				tspmupm0(3*n+1:3*n+iter-1) = tspmupm(3*n+1:3*n+iter-1);     % copy the up iterates with 0 appended
				tspmupm0(3*n+iter+1:3*n+2*iter-1) = tspmupm(3*n+iter:end);  % copy the um iterates with 0 appended
				B0 = B;
				ix = find(B0>=3*n+iter);
				B0(ix) = B0(ix) + 1;
				[tspmupm,B,lpiter] = lp_solver(A(V(:,1:iter)),r0,tspmupm0,B0,options);
				history.lpiter = [history.lpiter; lpiter];

				% Postprocess the solution
				up = tspmupm(3*n+1:3*n+iter);
				um = tspmupm(3*n+iter+1:end);
				u = up - um;
				t = tspmupm(1:n);
				x = x0 + V(:,1:iter) * u;

			else
				error('options.solver must be ''linprog'' or ''own''.\n');
			end


		elseif (strcmpi(options.norm,'l2'))

			% Solve a least-squares problem to find the expansion coefficients for xk - x0
			e = zeros(iter+1,1); e(1) = 1;
			u = H(1:iter+1,1:iter) \ (gamma0 * e);
			x = x0 + V(:,1:iter) * u;

		elseif (strcmpi(options.norm,'linf'))

			% Solve an LP to find the expansion coefficients for xk - x0
			if (strcmpi(options.solver,'linprog'))

				% Solve using linprog (without an initial guess)
				linprogoptions = optimoptions(@linprog,'display','off','algorithm','interior-point');
				c = [zeros(iter,1); 1];
				Aineq = [[A(V(:,1:iter)),-ones(n,1)];[-A(V(:,1:iter)),-ones(n,1)]];
				bineq = [r0;-r0];
				[ut,primalval,exitflag,output,lambda] = linprog(c,Aineq,bineq,[],[],[],[],[],linprogoptions); 
				history.lpiter = [history.lpiter; output.iterations];
				assert(exitflag == 1,'gmres_l12inf: linprog terminated with exitflag %d.',exitflag);

				% Postprocess the solution
				u = ut(1:iter); t = ut(iter+1:end);
				x = x0 + V(:,1:iter) * u;
				% Note: The objective value primalval coincides with mynorm(r,'linf') below

			elseif (strcmpi(options.solver,'own'))

				% Solve using our own customized LP solver (with initial guess)
				tspmupm0 = zeros(1+2*n+2*iter,1);
				tspmupm0(1:1+2*n) = tspmupm(1:1+2*n);                             % copy the t, sp, sm iterates
				tspmupm0(1+2*n+1:1+2*n+iter-1) = tspmupm(1+2*n+1:1+2*n+iter-1);   % copy the up iterates with 0 appended
				tspmupm0(1+2*n+iter+1:1+2*n+2*iter-1) = tspmupm(1+2*n+iter:end);  % copy the um iterates with 0 appended
				B0 = B;
				ix = find(B0>=1+2*n+iter);
				B0(ix) = B0(ix) + 1;
				[tspmupm,B,lpiter] = lp_solver(A(V(:,1:iter)),r0,tspmupm0,B0,options);
				history.lpiter = [history.lpiter; lpiter];

				% Postprocess the solution
				up = tspmupm(1+2*n+1:1+2*n+iter);
				um = tspmupm(1+2*n+iter+1:end);
				u = up - um;
				t = tspmupm(1);
				x = x0 + V(:,1:iter) * u;

			else
				error('options.solver must be ''linprog'' or ''own''.\n');
			end

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
		if (need_LAMBDA && (~strcmpi(options.solver,'own')) && (strcmpi(options.norm,'l1') || strcmpi(options.norm,'linf')))
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
	error('Norm must be ''l1'', ''l2'', or ''linf''.\n');
end


