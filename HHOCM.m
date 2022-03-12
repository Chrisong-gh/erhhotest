
function [Rabbit_Energy,Rabbit_Location,CNVG]=HHOCM(N,T,lb,ub,dim,fobj)


% tic
% initialize the location and Energy of the rabbit
Rabbit_Location=zeros(1,dim);
Rabbit_Energy=inf;

%Initialize the locations of Harris' hawks
X=initialization(N,dim,ub,lb);  %30*30
% logistic map 
X_Logistic=zeros(N,dim);
for i =1 : N
    %     X_D(i,:) = X(i,:) + 10*rand*(rand*(LB+UB-X(i,:))-X(i,:));
    X_Logistic(i,:) = 4*X(i,:).*(1-X(i,:));
    if fobj(X_Logistic(i,:))<fobj(X(i,:))
        X(i,:) = X_Logistic(i,:);
    end
end
CNVG=zeros(1,T);    %收敛曲线

t=0; % Loop counter

while t<T
    
    %% 常规操作
    for i=1:size(X,1)
        % Check boundries
        FU=X(i,:)>ub;
        FL=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(FU+FL)))+ub.*FU+lb.*FL;
        % fitness of locations
        fitness=fobj(X(i,:));
        % Update the location of Rabbit
        if fitness<Rabbit_Energy
            Rabbit_Energy=fitness;
            Rabbit_Location=X(i,:);
        end
    end
    
    %% 算法主要迭代与特�?
    E1=2*(1-(t/T)); % factor to show the decreaing energy of rabbit
    
    %mutation operators
    zeta = t/T;
    
    % Update the location of Harris' hawks
    for i=1:size(X,1)
        E0=2*rand()-1; %-1<E0<1
        Escaping_Energy=E1*(E0);  % escaping energy of rabbit
        
        
        
        if abs(Escaping_Energy)>=1
            %% Exploration:
            % Harris' hawks perch randomly based on 2 strategy:
            
            q=rand();
            %random index by hawks
            rand_Hawk_index = floor(N*rand()+1);
            
            X_rand = X(rand_Hawk_index, :);
            if q<0.5
                % perch based on other family members
                X(i,:)=X_rand-rand()*abs(X_rand-2*rand()*X(i,:));
            elseif q>=0.5
                % perch on a random tall tree (random site inside group's home range)
                X(i,:)=(Rabbit_Location(1,:)-mean(X))-rand()*((ub-lb)*rand+lb);
            end
            
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
                %initial mutation
                  X1_mutation = X(i,:);
                  X2_mutation = X(i,:);
%                 X1_mutation=X_rand;
%                 X2_mutation=X_rand;
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:));% Y
                %mutation operators
                if rand >= zeta
                    X1_mutation = Rabbit_Location;
                    X1_fit = fobj(X1_mutation);
                else
                    X1_mutation = X1;
                    X1_fit = fobj(X1_mutation);
                end
                
                X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:))+rand(1,dim).*Levy(dim);% Z
                %mutation operators
                if rand >= zeta
                    X2_mutation = Rabbit_Location;
                    X2_fit = fobj(X2_mutation);
                else
                    X2_mutation = X2;
                    X2_fit = fobj(X2_mutation);
                end
                
                %                 if fobj(X1_mutation)<fobj(X(i,:)) % improved move?
                %                     X(i,:)=X1_mutation;
                %                 else % hawks perform levy-based short rapid dives around the rabbit
                %                     X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X(i,:))+rand(1,dim).*Levy(dim);% Z
                %                     %mutation operators
                %                     if rand >= zeta
                %                         X2_mutation = Rabbit_Location;
                %                     else
                %                         X2_mutation = X2;
                %                     end
                %                     if (fobj(X2_mutation)<fobj(X(i,:))), % improved move?
                %                         X(i,:)=X2_mutation;
                %                     end
                %                 end
                tau = rand;
                
                %crossover
                
                X3_crossover = X1_mutation + tau*(X2_mutation-X1_mutation);
                X3_fitness = fobj(X3_crossover);
                %selection
                
                X1_mutation_fitness = fobj(X1_mutation);
                X2_mutation_fitness = fobj(X2_mutation);
                min_fitness = X1_mutation_fitness;
                best_location = X1_mutation;
                if min_fitness > X2_mutation_fitness
                    min_fitness = X2_mutation_fitness;
                    best_location = X2_mutation;
                end
                if min_fitness > X3_fitness
                    min_fitness = X3_fitness;
                    best_location = X3_crossover;
                end
                
                X(i,:) = best_location;
                
            end
            
            if r<0.5 && abs(Escaping_Energy)<0.5, % Hard besiege % rabbit try to escape by many zigzag deceptive motions
                % hawks try to decrease their average location with the rabbit
                rho=rand;
                X1_mutation = X(i,:);
                X2_mutation = X(i,:);
                Jump_strength=2*(1-rand());
                X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X));% Y
                if rho >= zeta
                    X1_mutation = Rabbit_Location;
                else
                    X1_mutation = X1;
                end
                X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X))+rand(1,dim).*Levy(dim);% Z
                if rand >= zeta
                    X2_mutation = Rabbit_Location;
                else
                    X2_mutation = X2;
                end
                tau = rand;
                
                %crossover
                
                X3_crossover = X1_mutation + tau*(X2_mutation-X1_mutation);
                X3_fitness = fobj(X3_crossover);
                %selection
                
                X1_mutation_fitness = fobj(X1_mutation);
                X2_mutation_fitness = fobj(X2_mutation);
                
                min_fitness = X1_mutation_fitness;
                best_location = X1_mutation;
                if min_fitness > X2_mutation_fitness
                    min_fitness = X2_mutation_fitness;
                    best_location = X2_mutation;
                end
                if min_fitness > X3_fitness
                    min_fitness = X3_fitness;
                    best_location = X3_crossover;
                end
                
                X(i,:) = best_location;
                %                 if fobj(X1)<fobj(X(i,:)) % improved move?
                %                     X(i,:)=X1;
                %                 else % Perform levy-based short rapid dives around the rabbit
                %                     X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-mean(X))+rand(1,dim).*Levy(dim);% Z
                %                     if (fobj(X2)<fobj(X(i,:))), % improved move?
                %                         X(i,:)=X2;
                %                     end
                %                 end
            end
            %%
        end
    end
    t=t+1;
    CNVG(t)=Rabbit_Energy;
    %    Print the progress every 100 iterations
    %    if mod(t,100)==0
    %        display(['At iteration ', num2str(t), ' the best fitness is ', num2str(Rabbit_Energy)]);
    %    end
end

end

% ___________________________________
function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end

