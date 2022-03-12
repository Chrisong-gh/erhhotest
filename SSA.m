%_________________________________________________________________________________
%  Salp Swarm Algorithm (SSA) source codes version 1.0
%
%  Developed in MATLAB R2016a
%
%  Author and programmer: Seyedali Mirjalili
%
%         e-Mail: ali.mirjalili@gmail.com
%                 seyedali.mirjalili@griffithuni.edu.au
%
%       Homepage: http://www.alimirjalili.com
%
%   Main paper:
%   S. Mirjalili, A.H. Gandomi, S.Z. Mirjalili, S. Saremi, H. Faris, S.M. Mirjalili,
%   Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems
%   Advances in Engineering Software
%   DOI: http://dx.doi.org/10.1016/j.advengsoft.2017.07.002
%____________________________________________________________________________________
%0
function [FoodFitness,FoodPosition,Convergence_curve]=SSA(N,Max_iter,lb,ub,dim,fobj)
%���������ʳ����Ӧ�Ⱥ���������ʳ��λ�ã��������ߣ�=���루Ⱥ��Ŀ�����������������ޣ����ޣ�ά�ȣ�������ʽ��
if size(ub,1)==1    %sizeȡub������
    ub=ones(dim,1)*ub;%ones����ȫһ����dim��1��
    lb=ones(dim,1)*lb;
end
Convergence_curve = zeros(1,Max_iter);%zeros����ȫ�����һ�У�max_iter��


%1
%Initialize the positions of salps  ��ʼ���׺���λ��
SalpPositions=initialization(N,dim,ub,lb);%��ʼ�����׺���Ⱥ��Ŀ��ά�ȣ������ޣ�ȷ��Ⱥ��λ��
FoodPosition=zeros(1,dim);%ʳ��λ�ö���
FoodFitness=inf;                              %ʳ����Ӧ�Ⱥ���ֵ



%2
%calculate the fitness of initial salps            �����ʼ�����Ӧ�Ⱥ���ֵ
for i=1:size(SalpPositions,1)    %ѭ��
    SalpFitness(1,i)=fobj(SalpPositions(i,:));%��λ�ü������Ӧ��ֵ
end

[sorted_salps_fitness,sorted_indexes]=sort(SalpFitness);%sort�����������У�(��)���������Ӧ��ֵ�������� 
% sorted_indexes�����������У�����ʾsorted_salps_fitness�е�Ԫ����SalpFitness��Ԫ�صĶ�Ӧ��

for newindex=1:N   %�µ���������
    Sorted_salps(newindex,:)=SalpPositions(sorted_indexes(newindex),:);
end

FoodPosition=Sorted_salps(1,:);%�ź������Ⱥȷ��ʳ��λ��
FoodFitness=sorted_salps_fitness(1);%�ź������Ⱥ����Ӧ��ֵ��Ϊʳ�����Ӧ��ֵ

%3
%Main loop.(��ѭ����.........................................................................................................................
l=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness of salps
%�ӵڶ��ε�����ʼ����Ϊ��һ�ε���ר�����ڼ���salps����Ӧ��
while l<Max_iter+1
    
    c1 = 2*exp(-(4*l/Max_iter)^2); % Eq. (3.2) in the paper
    
    for i=1:size(SalpPositions,1)
        
        SalpPositions= SalpPositions';%��ת��
        
        if i<=N/2
            for j=1:1:dim
                c2=rand();
                c3=rand();
                % Eq. (3.1) in the paper 
                if c3<0.5 
                    SalpPositions(j,i)=FoodPosition(j)+c1*((ub(j)-lb(j))*c2+lb(j));
                else
                    SalpPositions(j,i)=FoodPosition(j)-c1*((ub(j)-lb(j))*c2+lb(j));%jά�е�i��salp��λ�� 
                end
              
            end
            
        elseif i>N/2 && i<N+1   %  &������   �����ж�i>N/2�����ֵΪ�٣��Ϳ����ж��������ʽ��ֵΪ�٣��Ͳ���Ҫ���ж�i<N+1��ֵ.
            point1=SalpPositions(:,i-1);
            point2=SalpPositions(:,i);
            
            SalpPositions(:,i)=(point2+point1)/2; % Eq. (3.4) in the paper
        end
        
        SalpPositions= SalpPositions';%��ת��
    end
    
    for i=1:size(SalpPositions,1)  %�߽��޶�
        Tp=SalpPositions(i,:)>ub(1,:);
        Tm=SalpPositions(i,:)<lb(1,:);
        SalpPositions(i,:)=(SalpPositions(i,:).*(~(Tp+Tm)))+ub(1,:).*Tp+lb(1,:).*Tm;  %���salppositions�г����߽������ �����ϱ߽� 5 �� -5
        %   ~��ʾ��
        SalpFitness(1,i)=fobj(SalpPositions(i,:));
        
        if SalpFitness(1,i)<FoodFitness  %��׷����ʳ��Դ����Ӧ�Ⱥ���ֵ���
            FoodPosition=SalpPositions(i,:);
            FoodFitness=SalpFitness(1,i);           %��׷��ʳ�����ѭ��ֱ����������Ϊֹ
        end
    end
    
    Convergence_curve(l)=FoodFitness;%����ʳ����Ӧ��ֵ����������
    l = l + 1;
end



