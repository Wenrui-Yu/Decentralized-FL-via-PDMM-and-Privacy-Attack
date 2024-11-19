%% setting
clear all

%%
epoch=20001;
c=0.4;
alpha=0.1;
z_std=0;
iter=1;

%% privacy
MAX_COUNT=100;
it=1;

%
numSamples=30;
dimension=2;
ni=1;
num_clients=numSamples*2/ni;

% feature
mean1 = ones(1,dimension);
mean2 = -ones(1,dimension);
range=[-2 2 -2 2];
mo=0;

cov1 = eye(dimension)*0.25;
cov2 = eye(dimension)*0.25;

class1 = mvnrnd(mean1, cov1, numSamples);
class2 = mvnrnd(mean2, cov2, numSamples);

features = [class1; class2];
labels = [zeros(numSamples, 1); ones(numSamples, 1)];

%% Random geometric graph
radius=sqrt(log(num_clients)/num_clients).*100;

rand('seed', 4)
sensors=rand(num_clients,2).*100; %coordinate

% adjacency matrix
A=zeros(num_clients,num_clients);
AA=zeros(num_clients,num_clients);
for i=1:num_clients
   for j=2:num_clients
       if i<j && (sensors(i,1)-sensors(j,1))^2+(sensors(i,2)-sensors(j,2))^2 <= radius*radius
           A(i,j)=1;
           A(j,i)=1;
           AA(i,j)=1;
           AA(j,i)=-1;
       end
   end
end
%degree matrix
D=diag(sum(A));
%Laplacian matrix
L=D-A;
%Accessibility Matrix
P=A;
for i=2:num_clients
    P=P+A^i;
end
if ~isempty(find(~P, 1))
    disp('WARNING: It is not connected!');
%     return;
end

%visualization
figure
scatter(sensors(:,1),sensors(:,2))
gplot(A,sensors,'-*')
title(['Random geometric graph, N=',num2str(num_clients)])

%% distributed dataset
N=length(labels);
datt=[features,ones(N,1),labels];

randIndex = randperm(size(datt,1));
datt=datt(randIndex,:);
dat=reshape(datt',[dimension+2,length(labels)/num_clients,num_clients]);
dat=permute(dat,[2,1,3]);

%% pdmm
loss_m=[];
loss_s=[];
error=[];

%pdmm
x=zeros(dimension+1,num_clients);
z=randn([dimension+1,num_clients,num_clients])*z_std;
x_t=x;
z_t=z;

loss_t=[];
accu_t=[];

% sychronous
for epo=1:epoch
    z_tmp=z;
    los=[];
    acc=[];
    for i=1:num_clients
        [x(:,i),yij]=x_update(x(:,i),squeeze(dat(:,:,i)),AA(i,:),squeeze(z(:,i,:)),alpha,c,iter);
        z_tmp(:,:,i)=z_update(squeeze(z(:,:,i)),yij,1);

        lo=-log(h_x(datt(:,1:end-1),x(:,i)))'*datt(:,end)-log(1-h_x(datt(:,1:end-1),x(:,i)))'*(1-datt(:,end));
        los=[los;lo];
        labels_pred=(datt(:,1:end-1)*x(:,i))./abs(datt(:,1:end-1)*x(:,i));
        ac=1-sum(abs((labels_pred+1)/2-datt(:,end)))/(numSamples*2);
        acc=[acc;ac];
    end
    z=z_tmp;
    x_t(:,:,epo+1)=x;
    z_t(:,:,:,epo+1)=z;

    loss_t=[loss_t los];
    accu_t=[accu_t acc];

end

loss_mean=mean(loss_t);
loss_std=std(loss_t);
loss_m=[loss_m;loss_mean];
loss_s=[loss_s;loss_std];

%% approx x_t
z_diff=diff(z_t,1,4);
z_diff=cat(4,zeros([3,60,60,1]),z_diff);
error=[];
tmaxx=[3:9,10:5:50,60:10:100,200:100:1000,2000:1000:10000,12000:2000:20000];
for tmax=tmaxx
    c_node=1;
%     x_opt=x_t(:,c_node,tmax); %corrupted node estimation
%     x_opt=mean(x_t(:,:,tmax),2);
    x_opt=zeros(size(x_t(:,c_node,tmax)));
    
    x_hat=zeros(size(x_t));
    for i=1:num_clients
        for j=1:num_clients
            if AA(i,j)~=0
                x_hat(:,i,tmax-1)=x_opt-(squeeze(z_diff(:,j,i,tmax))-squeeze(z_diff(:,i,j,tmax-1)))/2/c*AA(i,j);
                break
            end
        end
    end
    for t=tmax-2:-1:1
         for i=1:num_clients
            for j=1:num_clients
                if AA(i,j)~=0
                    x_hat(:,i,t)=x_hat(:,i,t+1)-(squeeze(z_diff(:,j,i,t+1))-squeeze(z_diff(:,i,j,t)))/2/c*AA(i,j);
                    break
                end
            end
        end
    end
    num_para=length(x_hat(:,1,1));
    err=[]; 
    for t=2
        Sol=[];
        for node_i=1:num_clients
%             rhs=c*sum(AA(node_i,:)~=0)*(x_hat(:,node_i,t)+x_hat(:,node_i,t+2))-2*c*x_hat(:,:,t+1)*A(node_i,:)';
            rhs=(c*sum(AA(node_i,:)~=0)-1/alpha)*(x_hat(:,node_i,t+2)-x_hat(:,node_i,t))+1/alpha*x_hat(:,node_i,t+3)+(2*c*sum(AA(node_i,:)~=0)-1/alpha)*x_hat(:,node_i,t+1)-2*c*x_hat(:,:,t+2)*A(node_i,:)';
            Sol=[Sol;rhs(1:end-1)'./rhs(end)];
        end
        err=[err,sum(sqrt(sum((Sol-datt(:,1:2)).^2,2)))/numSamples/2]; %average distance
    end
    error=[error;err];
end

%% averaging
x=zeros(dimension+1,num_clients);
x_a=x;
loss_a=[];
accu_a=[];
Sol_a=[];
for epo=1:epoch
    soll=[];
    for i=1:num_clients
        dx=(h_x(squeeze(dat(:,1:end-1,i)),x(:,i))-squeeze(dat(:,end,i)))'*squeeze(dat(:,1:end-1,i));
        x(:,i)=x(:,i)-alpha*dx';
        soll=[soll,dx(1:end-1)'./dx(end)];
    end
    x_a(:,:,epo+1)=x;
    x=mean(x,2)*ones(1,num_clients);
    
    lo=-log(h_x(datt(:,1:end-1),x(:,i)))'*datt(:,end)-log(1-h_x(datt(:,1:end-1),x(:,i)))'*(1-datt(:,end));
    loss_a=[loss_a;lo];
    labels_pred=(datt(:,1:end-1)*x(:,i))./abs(datt(:,1:end-1)*x(:,i));
    ac=1-sum(abs((labels_pred+1)/2-datt(:,end)))/(numSamples*2);
    accu_a=[accu_a;ac];
    
    Sol_a(:,:,epo)=soll';
end

error_a=[];
for t=tmaxx
    error_a=[error_a, sum(sqrt(sum((squeeze(Sol_a(:,:,t))-datt(:,1:2)).^2,2)))/numSamples/2];
end

%%
figure
subplot(1,2,1)
epo=1:epoch;
color=['b'];
y_upper = loss_m + loss_s;
y_lower = loss_m - loss_s;

% plot(loss_a,'m');
loglog(loss_a,'m');
hold on

p=fill([epo, fliplr(epo)], [y_upper(1,:), fliplr(y_lower(1,:))], color(1));
p.EdgeColor = 'none';
p.FaceAlpha=0.5;

% plot(epo, loss_m(1,:), color(1));
loglog(epo, loss_m(1,:), color(1));

grid on
axis([1 20000 0 50]);
xlabel('Iteration (t)');ylabel('Loss');
title('(a)')
legend('FedAvg','PDMM')
set(gca, 'fontsize' ,12)
xticks([10 100 1000 10000])

subplot(1,2,2)
x=1:44;
loglog(tmaxx(x),error_a(x),'-diamond','Linewidth', 1)
hold on
loglog(tmaxx(x),error(x),'-diamond','Linewidth', 1)
grid on
xlabel('Iteration (t)');
ylabel('Reconstruction Error');
legend('FedAvg','PDMM','Location','best')
axis([0 20000 0 10]);
title('(b)');
set(gca, 'fontsize' ,12)

%%
figure
plot(reshape(dat(:,1,:),1,[]),reshape(dat(:,2,:),1,[]),'o');axis(range);
hold on
plot(Sol(:,1),Sol(:,2),'*');
legend('data','reconstruction')
title(['ni=',num2str(ni),', round(observation)=',num2str(it),', t=',num2str(t), ', feature=',num2str(dimension)])

%%
function [xi,yij]=x_update(xi,data,j,zij,alpha,c,iter)
    for k=1:iter
        dxi=(h_x(data(:,1:end-1),xi)-data(:,end))'*data(:,1:end-1)+j*zij'+c*sum(j~=0)*xi';
        xi=xi-alpha*dxi';
    end
    yij=zij+2*c*xi*j;
end

function zji=z_update(zji,yij,theta)
    zji=(1-theta)*zji+theta*yij;
end

function hx=h_x(datax,xi)
    hx=1./(1+exp(-datax*xi));
end
