%______________________________________________________________________________________________
%  Moth-Flame Optimization Algorithm (MFO)                                                            
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
%  S. Mirjalili, Moth-Flame Optimization Algorithm: A Novel Nature-inspired Heuristic Paradigm, 
%  Knowledge-Based Systems, DOI: http://dx.doi.org/10.1016/j.knosys.2015.07.006
%_______________________________________________________________________________________________
% You can simply define your cost in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of generations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single number numbers

% To run MFO: [Best_score,Best_pos,cg_curve]=MFO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%______________________________________________________________________________________________

function[] = P_Value(Function_name,Test_count)

display(['*************',Function_name , '  is running ***********']);
% the number of comparative algorithms.
pv_count = 8;
pv_name = ["ERHHO", "SMA",  "WOA", "SSA","SCA","HHO","DHHO/M", "HHOCM"];
pv = zeros(pv_count, Test_count);

for i = 1:1:Test_count
SearchAgents_no=30; % Number of search agents搜索代理数目，粒子数

% Function_name='F5'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)

Max_iteration=500; % Maximum numbef of iterations最大迭代

% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Get_Functions_details();

[Best_score_erhho,Best_pos_erhho,cg_curve_erhho]=ERHHO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score_sma,Best_pos_sma,cg_curve_sma]=SMA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score_woa,Best_pos_woa,cg_curve_woa]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score_ssa,Best_pos_ssa,cg_curve_ssa]=SSA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score_sca,Best_pos_sca,cg_curve_sca]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score_hho,Best_pos_hho,cg_curve_hho]=HHO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score_dhho,Best_pos_dhho,cg_curve_dhho]=DHHO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[Best_score_hhocm,Best_pos_hhocm,cg_curve_hhocm]=HHOCM(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);



pv(1,i) = Best_score_erhho;
pv(2,i) = Best_score_sma;
pv(3,i) = Best_score_woa;
pv(4,i) = Best_score_ssa;
pv(5,i) = Best_score_sca;
pv(6,i) = Best_score_hho;
pv(7,i) = Best_score_dhho;
pv(8,i) = Best_score_hhocm;

end

for  i=2:pv_count
    ansSign = ranksum(pv(1,:),pv(i,:));
    display([pv_name(1),'与',pv_name(i),' ==>',Function_name , ':P-Value = ', num2str(ansSign)]);
end

