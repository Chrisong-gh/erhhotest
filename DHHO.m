% Developed in MATLAB R2013b
% Source codes demo version 1.0
% _____________________________________________________

% Main paper:
% Harris hawks optimization: Algorithm and applications
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
% Future Generation Computer Systems,
% DOI: https://doi.org/10.1016/j.future.2019.02.028
% https://www.sciencedirect.com/science/article/pii/S0167739X18313530
% _____________________________________________________

% You can run the HHO code online at codeocean.com  https://doi.org/10.24433/CO.1455672.v1
% You can find the HHO code at https://github.com/aliasghar68/Harris-hawks-optimization-Algorithm-and-applications-.git
% _____________________________________________________

%  Author, inventor and programmer: Ali Asghar Heidari,
%  PhD research intern, Department of Computer Science, School of Computing, National University of Singapore, Singapore
%  Exceptionally Talented Ph. DC funded by Iran's National Elites Foundation (INEF), University of Tehran
%  03-03-2019

%  Researchgate: https://www.researchgate.net/profile/Ali_Asghar_Heidari

%  e-Mail: as_heidari@ut.ac.ir, aliasghar68@gmail.com,
%  e-Mail (Singapore): aliasgha@comp.nus.edu.sg, t0917038@u.nus.edu
% _____________________________________________________
%  Co-author and Advisor: Seyedali Mirjalili
%
%         e-Mail: ali.mirjalili@gmail.com
%                 seyedali.mirjalili@griffithuni.edu.au
%
%       Homepage: http://www.alimirjalili.com
% _____________________________________________________
%  Co-authors: Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, and Hui-Ling Chen

%       Homepage: http://www.evo-ml.com/2019/03/02/hho/
% _____________________________________________________
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Harris's hawk optimizer: In this algorithm, Harris' hawks try to catch the rabbit.

% T: maximum iterations, N: populatoin size, CNVG: Convergence curve
% To run HHO: [Rabbit_Energy,Rabbit_Location,CNVG]=HHO(N,T,lb,ub,dim,fobj)

function [Rabbit_Energy,Rabbit_Location,CNVG]=DHHO(N,T,lb,ub,dim,fobj)

%disp('HHO is now tackling your problem')
%tic
% initialize the location and Energy of the rabbit
Rabbit_Location=zeros(1,dim);
Rabbit_Energy=inf;

%Initialize the locations of Harris' hawks
X=initialization(N,dim,ub,lb);

% %Circle map
% a=0.5;
% b=0.2;
% for i=1:N
%     X(i+1)=mod(X(i)+b-(a/(2*pi))*sin(2*pi*X(i)),1);
% end

Xq = zeros(N,dim);

CNVG=zeros(1,T);

t=0; % Loop counter

while t<T
    for i=1:size(X,1)
        % Check boundries
        FU=X(i,:)>ub;FL=X(i,:)<lb;X(i,:)=(X(i,:).*(~(FU+FL)))+ub.*FU+lb.*FL;
        % fitness of locations
        fitness=fobj(X(i,:));
        % Update the location of Rabbit
        if fitness<Rabbit_Energy
            Rabbit_Energy=fitness;
            Rabbit_Location=X(i,:);
        end
    end
    
    E1=2*(1-(t/T)); % factor to show the decreaing energy of rabbit    eq.(3) E1=（0,2）
%     E1 = 2 ./ (1 + exp((t-T/2)/T*10));
    a = 2.5;
    sigma = randn*(sin(pi/2 * t/T).^a + cos(pi/2*t/T)-1);
    % Update the location of Harris' hawks
    for i=1:size(X,1)
        E0=2*rand()-1; %-1<E0<1
        Escaping_Energy=E1*(E0)+sigma;  % escaping energy of rabbit （-2,2）
%         E = E1*(E0);
%         Escaping_Energy = 2* (exp(E)-exp(-E))./(exp(E)+exp(-E));
        if abs(Escaping_Energy)>=1
            %% Exploration:
            % Harris' hawks perch randomly based on 2 strategy:
            
            q=rand();
            F = 0.5;
%             rand_Hawk_index = floor(N*rand()+1);  % 0-30 随机选一个
%               r1 = unidrnd(N);
%               r2 = unidrnd(N);
%               r3 = unidrnd(N);
%               r4 = unidrnd(N);
            X_r1 = X(unidrnd(N), :);
            X_r2 = X(unidrnd(N), :);
            X_r3 = X(unidrnd(N), :);
            X_r4 = X(unidrnd(N), :);
            % F_rand = fobj(X_rand);
            if q<0.5
                % perch based on other family members
%                 X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*X(i,:));
%                 X(i,:)= X_rand-rcp(lb,ub)*rand()*abs(X_rand-2*rand()*X(i,:)); 
%                 unidrnd(N) 1~N 
                X(i,:)= Rabbit_Location(1,:)+ F*(X_r1 - X_r2)+F*(X_r3 -X_r4); 
%                 X(i,:)= X_rand-rand()*abs(X_rand-2*rand()*X(i,:)); 
            elseif q>=0.5
                % perch on a random tall tree (random site inside group's home range)
%                 X(i,:)=(Rabbit_Location(1,:)-mean(X))-rand()*((ub-lb)*rand+lb);
%                 X(i,:)=(Rabbit_Location(1,:)-mean(X))-rcp(lb,ub)*rand()*((ub-lb)*rand+lb);
                X(i,:)=(Rabbit_Location(1,:)-mean(X))-rand()*((ub-lb)*rand+lb);
            end
%             if q<0.3
%                 % perch based on other family members
%                 % X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*X(i,:));
%                 X(i,:)= X_rand-rand()*abs(X_rand-2*rand()*X(i,:));  % random convergence parameter
%             elseif q>=0.6
%                 % perch on a random tall tree (random site inside group's home range)
%                 % X(i,:)=(Rabbit_Location(1,:)-mean(X))-rand()*((ub-lb)*rand+lb);
%                 X(i,:)=(Rabbit_Location(1,:)-mean(X))-rand()*((ub-lb)*rand+lb);
%             else
%                 X_rand_fan = (ub + lb) - X_rand;
%                 X(i,:)= X_rand_fan - rand()*abs(X_rand_fan-2*rand()*X(i,:));  % random convergence parameter
%             end
            
        elseif abs(Escaping_Energy)<1
            %% Exploitation:
            % Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
            
            %% phase 1: surprise pounce (seven kills)
            % surprise pounce (seven kills): multiple, short rapid dives by different hawks
            
            r=rand(); % probablity of each event
            
            if r>=0.5 && abs(Escaping_Energy)<0.5 % Hard besiege
                X(i,:)=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X(i,:));
            end
            
            if r>=0.5 && abs(Escaping_Energy)>=0.5  % Soft besiege
                Jump_strength=2*(1-rand()); % random jump strength of the rabbit
                X(i,:)=(Rabbit_Location-X(i,:))-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:));
            end
            
            %% phase 2: performing team rapid dives (leapfrog movements)
            if r<0.5 && abs(Escaping_Energy)>=0.5 % Soft besiege % rabbit try to escape by many zigzag deceptive motions
                
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:));
                
                if fobj(X1)<fobj(X(i,:)) % improved move?
                    X(i,:)=X1;
                else % hawks perform levy-based short rapid dives around the rabbit
                    X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:))+rand(1,dim).*Levy(dim);
                    if (fobj(X2)<fobj(X(i,:))) % improved move?
                        X(i,:)=X2;
                    end
                end
            end
            
            if r<0.5 && abs(Escaping_Energy)<0.5 % Hard besiege % rabbit try to escape by many zigzag deceptive motions
                % hawks try to decrease their average location with the rabbit
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X));
                
                if fobj(X1)<fobj(X(i,:)) % improved move?
                    X(i,:)=X1;
                else % Perform levy-based short rapid dives around the rabbit
                    X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X))+rand(1,dim).*Levy(dim);
                    if (fobj(X2)<fobj(X(i,:))) % improved move?
                        X(i,:)=X2;
                    end
                end
            end
            
            % 准反向  Quasi-Opposition-Based Learning
%             CS = (ub + lb)/2;
%             MP = (ub + lb) - X(i,:);
%             if MP > CS
%                 Xq(i,:) = CS +rand() * (MP - CS);
%             else
%                 Xq(i,:) = MP + rand() * (CS - MP);
%             end
%             if fobj(Xq(i,:))<fobj(X(i,:))
%                 X(i,:) = Xq(i,:);
%             end
%             % 准反向
%           
             % 准反射  Quasi-reflection-Based Learning
%             CS = (ub + lb)/2;
%             MP = X(i,:);
%             if MP > CS
%                 Xq(i,:) = CS +rand() * (MP - CS);
%             else
%                 Xq(i,:) = MP + rand() * (CS - MP);
%             end
%             if fobj(Xq(i,:))<fobj(X(i,:))
%                 X(i,:) = Xq(i,:);
%             end
            % 准反射
        end
    end
    t=t+1;
    CNVG(t)=Rabbit_Energy;
    %    Print the progress every 100 iterations
    %    if mod(t,100)==0
    %        display(['At iteration ', num2str(t), ' the best fitness is ', num2str(Rabbit_Energy)]);
    %    end
end
%toc
end

% ___________________________________
function o=Levy(d)
    beta=1.5;
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
    o=step;
end
 % random convergence parameter
function r = rcp(lb,ub)
    if size(lb,2)==1
        Aend = lb;
        Astart = ub;
    else
        Aend = lb(1,1);
        Astart = ub(1,1);
    end
    

    sigma = 0;
%     r = Aend + (Astart-Aend)* abs(normrnd(0,1)) + sigma * rand();
    r = abs(Aend + (Astart-Aend)* abs(normrnd(0,1)) )+ sigma * rand();
    
%     E .*2.* abs(normrnd(0,1,1,501))
%     r = abs(Aend+(Astart-Aend)*normrnd(0,1)+ sigma * rand());
%     r =abs(E.* (exp(E)-exp(-E))./(exp(E)+exp(-E)));
%     r =abs(E.* (exp(E)-exp(-E))./(exp(E)+exp(-E)));
   %r =(Astart-Aend)*normrnd(0,1);
end

function r = tanh_abs(E)
    
  %    r = Aend + (Astart-Aend)*normrnd(0,1) + sigma * rand();
%     r = abs(Aend+(Astart-Aend)*normrnd(0,1)+ sigma * rand());
%     r =abs(E.* (exp(E)-exp(-E))./(exp(E)+exp(-E)));
%     r =abs((exp(E)-exp(-E))./(exp(E)+exp(-E)));
    r=2* ((exp(E)-exp(-E))./(exp(E)+exp(-E))+1);
   %r =(Astart-Aend)*normrnd(0,1);
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end

