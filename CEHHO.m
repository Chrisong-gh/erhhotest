%% [1]������,��ͳ,�����,л��.���羫Ӣ����˹ӥ�Ż��㷨[J/OL].�����Ӧ��:1-10[2021-01-29].http://kns.cnki.net/kcms/detail/51.1307.TP.20210114.0947.032.html.
%% ���羫Ӣ����˹ӥ�Ż��㷨
function [Rabbit_Energy,Rabbit_Location,CNVG]=CEHHO(N,T,lb,ub,dim,fobj)
% initialize the location and Energy of the rabbit
Rabbit_Location=zeros(1,dim);
Rabbit_Energy=inf;

%Initialize the locations of Harris' hawks
X=initialization(N,dim,ub,lb);

%%����������Ⱥ
XYS = X;
%����������Ⱥ��Ӧ��ֵ
for i=1:size(X,1)
    fitnessYS(i)=fobj(X(i,:));
end
CNVG=zeros(1,T);

t=0; % Loop counter
mfold = mean(fitnessYS);%������Ⱥƽ��ֵ
while t<T
    for i=1:size(X,1)
        % Check boundries
        FU=X(i,:)>ub;FL=X(i,:)<lb;X(i,:)=(X(i,:).*(~(FU+FL)))+ub.*FU+lb.*FL;
        % fitness of locations
        fitness(i)=fobj(X(i,:));
        %����������Ⱥ
        if fitness(i)<fitnessYS(i)
            fitnessYS(i) = fitness(i);
            XYS(i,:) = X(i,:);
        end
        % Update the location of Rabbit
        if  fitness(i)<Rabbit_Energy
            Rabbit_Energy= fitness(i);
            Rabbit_Location=X(i,:);
            indexBest = i;
        end    
    end
    %% �Ľ��㣺��Ӣ�ȼ��ƶ�
    [fitnessTemp, index]= sort(fitness);%����
    %ȡǰ3λ���������Ž�
    S = sum(fitnessTemp(1:3));%ǰ3λ��Ӧ��ֵ��
    Temp = X(index(1),:).*fitnessTemp(1)./S + X(index(2),:).*fitnessTemp(2)./S +X(index(3),:).*fitnessTemp(3)./S;
    %Խ�紦��
    FU=Temp>ub;FL=Temp<lb;Temp=(Temp.*(~(FU+FL)))+ub.*FU+lb.*FL;
    fitATemp = fobj(Temp);
    if (fitATemp<Rabbit_Energy)
        Rabbit_Location = Temp;
        Rabbit_Energy = fitATemp;
        X(indexBest,:) = Temp;
        fitness(indexBest) = fitATemp; 
        XYS(indexBest,:) = Temp;
        fitnessYS(indexBest) = fitATemp;
    end
    %% �Ľ���:��˹������߲���
    mf = mean(fitnessYS);%��ǰ������Ⱥƽ��ֵ
    if t>0 %�ڶ��ε�����ʼ�ж�
        if abs(mf - mfold)<10E-5 %���ǰ�����μ���û�䣬�����ø�˹������߲���
          for i = 1:N
            indexR = randi(N);
            delta = cos(pi*(t/T)^2/2).*(X(i,:) - XYS(indexR,:));
            Temp = delta.*randn();
            %Խ�紦��
             FU=Temp>ub;FL=Temp<lb;Temp=(Temp.*(~(FU+FL)))+ub.*FU+lb.*FL;
             X(i,:) = Temp;
          end
        end
    end
    mfold = mf;
    %% �Ľ��㣺 �����������������²���
    E1 = 2*(1-(t/T)^(1/3))^(1/3); %������ʽ��17��
    % Update the location of Harris' hawks
    for i=1:size(X,1)
        E0=2*rand()-1; %-1<E0<1
        Escaping_Energy=E1*(E0);  % escaping energy of rabbit
        
        if abs(Escaping_Energy)>=1
            %% Exploration:
            % Harris' hawks perch randomly based on 2 strategy:
            
            q=rand();
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
            
          %% �Ľ��㣺��r����tent����ӳ����иĽ�
            R =Tent(10);
            r = R(1);          
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
                    if (fobj(X2)<fobj(X(i,:))) % improved move
                        X(i,:)=X2;
                    end
                end
            end
            %%
        end
    end
    t=t+1;
    CNVG(t)=Rabbit_Energy;
end
end

%Levy����
function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end

