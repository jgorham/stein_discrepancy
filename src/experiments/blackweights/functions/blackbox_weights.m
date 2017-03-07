function wts = blackbox_weights(x, method, model, varargin)


switch lower(method)
    case lower({'controlvar'})
        %Simpel control variable method: x should be zero-mean control variates in this case
        wts = weights_control_variate(x, model, varargin{:});
        
    case 'stein'
        % The method in black-box importance sampling
        wts = weights_stein(x, model, varargin{:});        
        
    case {'bayesian', 'bayesianmc'}
        % Oates etal method; no partition, equivalent to Bayesian MC with Stein kernel 
        wts = weights_control_functional(x, model, size(x,1), varargin{:});
        
    case {'controlfunctional_fold2'}        
        % Oates etal, use two fold partition. 
        wOates_1 = weights_control_functional(x, model, round(size(x,1)/2));
        wOates_2 = weights_control_functional(x(end:-1:1,:), model, round(size(x,1)/2));
        wts = (wOates_1 + wOates_2(end:-1:1))/2;

    case lower({'KDE-Gaussian-rot'})
        % Delyon & Portier method: https://arxiv.org/abs/1409.0733
        % IS-KDE based weights (Gaussian kernel, bandwith using rule of thumb (rot))
        %q = @(x)pdf(model, x);
        q = @(x)logpdf(model, x); 
        wts = weights_IS_KDE_Gauss(x, q,'hrange',1, 'q', 'logq'); 

    case lower({'KDE-Gaussian-search'})
        % Delyon & Portier method: https://arxiv.org/abs/1409.0733        
        % IS-KDE (Gaussian kernel, search bandwidth heuristic)
        q = @(x)logpdf(model, x);
        wts = weights_IS_KDE_Gauss(x, q, 'q', 'logq'); 

    case lower({'KDE-Epan-rot'})        
        % Delyon & Portier method: https://arxiv.org/abs/1409.0733        
        % IS-KDE based weights (Epanechnikov kernel, bandwith using rule of thumb (rot))
        q = @(x)pdf(model, x);
        wts= weights_IS_KDE(x, q,'hrange',1); 

    case lower({'KDE-Epan-search'})
        % Delyon & Portier method: https://arxiv.org/abs/1409.0733        
        % IS-KDE, (Epanechnikov kernel, search bandwidth heuristic)
        q = @(x)pdf(model, x);
        wts = weights_IS_KDE(x, q); 
         
    otherwise 
        error('wrong!');
end


wts = wts(:);
        
        
