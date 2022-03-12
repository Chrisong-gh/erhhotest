%  Sine Cosine Algorithm (SCA)  
%
%  Source codes demo version 1.0                                                                      
%                                                                                                     
%  Developed in MATLAB R2011b(7.13)                                                                   
%                                                                                                     
%  Author and programmer: Seyedali Mirjalili                                                          
%                                                                                                     
%         e-Mail: ali.mirjalili@gmail.com                                                             
%                 seyedali.mirjalili@griffithuni.edu.au                                               
%                                                                                                     
%       Homepage: http://www.alimirjalili.com                                                         
%                                                                                                     
%  Main paper:                                                                                        
%  S. Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems
%  Knowledge-Based Systems, DOI: http://dx.doi.org/10.1016/j.knosys.2015.12.022
%_______________________________________________________________________________________________
% You can simply define your cost function in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of iterations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single numbers

% To run SCA: [Best_score,Best_pos,cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%______________________________________________________________________________________________


function [Destination_fitness,Destination_position,Convergence_curve]=SCA(N,Max_iteration,lb,ub,dim,fobj)

% display('SCA is optimizing your problem');

%Initialize the set of random solutions初始化随机解集
X=initialization(N,dim,ub,lb);

Destination_position=zeros(1,dim);%终点位置
Destination_fitness=inf;%终点适应度值

Convergence_curve=zeros(1,Max_iteration);%收敛曲线
Objective_values = zeros(1,size(X,1));%目标值

% Calculate the fitness of the first set and find the best one计算第一组的适合度并找出最佳的一组
for i=1:size(X,1)
    Objective_values(1,i)=fobj(X(i,:));
    if i==1
        Destination_position=X(i,:);%终点位置
        Destination_fitness=Objective_values(1,i);%目标值给终点适应度值
    elseif Objective_values(1,i)<Destination_fitness%终点适应度值大于目标值时
        Destination_position=X(i,:);
        Destination_fitness=Objective_values(1,i);
    end
    
    All_objective_values(1,i)=Objective_values(1,i);%目标值给所有的目标值进行初始化随机解
end

%Main loop主循环
t=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness从第二次迭代开始，因为第一次迭代专门用于计算适应度
while t<=Max_iteration
    
    % Eq. (3.4)
    a = 2;
   
    r1=a-t*((a)/Max_iteration); % r1 decreases linearly from a to 0   %r1从a线性减少到0自适应调整正余弦函数的范围
    
    % Update the position of solutions with respect to destination更新解决方案相对于终点的位置
    for i=1:size(X,1) % in i-th solution在第i个解决方案中
        for j=1:size(X,2) % in j-th dimensionin 在j维中
            
            % Update r2, r3, and r4 for Eq. (3.3)%更新r2、r3和r4
            r2=(2*pi)*rand();%随机数{0，2pi}
            r3=2*rand;%随机数
            r4=rand();%随机数
            
            % Eq. (3.3)
            if r4<0.5
                % Eq. (3.1)
                X(i,j)= X(i,j)+(r1*sin(r2)*abs(r3*Destination_position(j)-X(i,j)));%正弦
            else
                % Eq. (3.2)
                X(i,j)= X(i,j)+(r1*cos(r2)*abs(r3*Destination_position(j)-X(i,j)));%余弦
            end
            
        end
    end
    
    for i=1:size(X,1)
         
        % Check if solutions go outside the search spaceand bring them back%检查解决方案是否超出搜索范围并将其带回
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % Calculate the objective values计算目标值
        Objective_values(1,i)=fobj(X(i,:));
        
        % Update the destination if there is a better solution如果有更好的解决方案，更新终点值
        if Objective_values(1,i)<Destination_fitness
            Destination_position=X(i,:);%更新终点位置
            Destination_fitness=Objective_values(1,i);%更新终点适应度函数值
        end
    end
    
    Convergence_curve(t)=Destination_fitness;%收敛曲线绘制
    
    % Display the iteration and best optimum obtained so far %显示迭代和迄今为止获得的最佳结果
%     if mod(t,50)==0%如果迭代次数除以50取余数为0
%         display(['At iteration ', num2str(t), ' the optimum is ', num2str(Destination_fitness)]);%迭代时的最佳值是：
%     end
    
    % Increase the iteration counter%迭代累加
    t=t+1;
end