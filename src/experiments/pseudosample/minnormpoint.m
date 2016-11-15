function [x,values,indices,w,gaps,all_outputs] = minnormpoint(XX,maxiter,gap,starting_point)

% n points in R^p

[ p n ] = size(XX);

if nargin >= 3
    display = 0;
    TOL_1 = gap;
    TOL_2 = gap;
    TOL_3 = gap;
    TOL_4 = gap;
    display = 1;

else
    TOL_1 = 1e-10;
    TOL_2 = 1e-10;
    TOL_3 = 1e-10;
    TOL_4 = 1e-10;
    display = 1;
end
maxnorm = max( sum(XX.^2,1) );

if nargin < 4
    % min-norm point among all points
    [a,b] = min( sum(XX.^2,1) );

else
    b = starting_point;
end
    x = XX(:,b);
w = 1;
X = x;
indices = b; 
iter = 0;


while iter < maxiter
    if mod(iter,100)==1, iter, end
    iter = iter + 1;
    % step 1
    % a
    x = X*w;

    direction = - X*w;

    % b
    [a,b] = max(XX' * direction);
    xx = XX(:,b);
    bb = b;
    % c
    % compute values of the function
    gaps(iter) = direction' * xx - direction' * x;
    values(iter) =  -.5 * (x'*x);
    all_outputs.indices{iter} = indices;
    all_outputs.weights{iter} = w;

    if gaps(iter) < TOL_1 * maxnorm
        % stop!
        if display, fprintf('stopped at step 1c - reached upper bound on duality gap \n'); end
        break;
    end



    % d
    if min(sum( (X - repmat(xx,1,size(X,2))).^2 , 1 ) ) < TOL_2 * maxnorm;
        % stop!
        if display, fprintf('stopped at step 1d\n'); end
        break;
    end

    % e
    X = [X, xx];
    w = [w; 0];
indices = [ indices, bb];

    iterloc =1;
    while 1,
        iterloc = iterloc + 1;
        if iterloc > 100*n,
            fprintf('probably looping between step 2 and 3, exit \n');
            iterloc = 0;
            break;
        end
        % step 2
        % a
        try
            R = chol( X'*X + maxnorm );
        catch
            % not positive definite
            iterloc = 0;
             
            fprintf('not positive definite when adding new point in step 2, exit\n');
            break;
        end
        v = R \ ( R' \ ones(size(X,2),1) );
        v = v / sum(v);

        % checking
        % X'* ( X * v );

        % b
        if all( v > TOL_3),
            w = v;
            break;
            % go to step 1
        else

            % step 3
            % a
            ind = find( w - v > TOL_4);
            if isempty(ind)
                iterloc = 0;
                fprintf('can''t do line search in step 2, exit\n');
                break;
            end
            %b
            theta = min( 1, min ( w(ind)./(w(ind)-v(ind))));
            %c
            w = theta * v + (1-theta) * w;
            %d-e
            torem = find( w < TOL_3);
            w(torem) = [];
            X(:,torem) = [];
            indices(torem) = [];
            w = w / sum(w);
            % go to step 2
        end
    end
    if iterloc==0, break; end
end
iter
[a,b] = max(- XX' * x);
xx = XX(:,b);

gaps(iter+1) = direction' * xx - direction' * x;
values(iter+1) =  -.5 * (x'*x);


