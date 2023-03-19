clear all
%% Particle Filter %%
%% 4 Tank Problem
% The prior information
data = readmatrix('Measurements.csv');
step = 100;
A1 = 28;        % cm2
A2 = 32;        % cm2
A3 = 28;        % cm2
A4 = 32;        % cm2

kc = 0.50;      % V/cm
T1 = 62;        % s
T2 = 90;        % s
T3 = 23;        % s
T4 = 30;        % s

a1 = 0.071;     % cm2
a2 = 0.057;     % cm2
a3 = 0.071;     % cm2
a4 = 0.057;     % cm2

g = 981;        % cm/s2
gamma1 = 0.7;   % dimensionless
gamma2 = 0.6 ;  % dimensionless
k1 = 3.33;      % cm3/Vs
k2 = 3.35;      % cm3/Vs
kc = 0.5;       % V/cm

A = [-1/T1 0 A3/(A1*T3) 0; 0 -1/T2 0 A4/(A2*T4); 0 0 -1/T3 0; 0 0 0 -1/T4];
B = [gamma1*k1/A1 0 ; 0 gamma2*k2/A2; 0 (1-gamma2)*k2/A3; (1-gamma1)*k1/A1 0];
C = [1/kc 0 0 0; 0 1/kc 0 0];
D= [0];
v1 = 3;
v2 = 3;
uk = [v1; v2];
P = 10^5*eye(4);

%% Discretization
Ts=0.1;      %Sampling Time
csys= ss(A,B,C,D);
dsys= c2d(csys, Ts);
[A B C D] = ssdata(dsys);

%%

N = 500;            % particle size
n = 4;              % states
L = chol(P);        % cholesky factor of P
x = (data(1,:)'*ones(1,N))'+ randn(N,n)*L;  % Adding covariance to each particle
x1_post = x(:,1);
x2_post = x(:,2);
x3_post = x(:,3);
x4_post = x(:,4);
%% Process Noise
Q = 3*eye(4);     % process noise
w = chol(Q)*randn(n,N);        % roughening of the prior
w1 = w(1,:);
w2 = w(2,:);
w3 = w(3,:);
w4 = w(4,:);
%% Roughening the Prior
x1_post = x1_post + w1';
x2_post = x2_post + w2';
x3_post = x3_post + w3';
x4_post = x4_post + w4';

% Initialize Prediction values
xpri = zeros(N,n);
x1pri = xpri(:,1);
x2pri = xpri(:,2);
x3pri = xpri(:,3);
x4pri = xpri(:,4);

%% Prediction Step
for i = 1:N
     x1pri(i) = -a1/A1*sqrt(2*g*x1_post(i)) + a3/A1*sqrt(2*g*x3_post(i)) + (gamma1*k1*v1)/A1 + w1(i);
     x2pri(i) = -a2/A2*sqrt(2*g*x2_post(i)) + a4/A2*sqrt(2*g*x4_post(i)) + (gamma2*k1*v2)/A2 + w2(i);
     x3pri(i) = -a3/A3*sqrt(2*g*x3_post(i)) + (1 - gamma2)*k2*v2/A3  +  w3(i);
     x4pri(i) = -a4/A4*sqrt(2*g*x4_post(i)) + (1 - gamma1)*k1*v1/A4  +  w4(i);
end
xpri  = abs(xpri);
x1pri = abs(x1pri);
x2pri = abs(x2pri);
x3pri = abs(x3pri);
x4pri = abs(x4pri);

% Importance Weights (Likelihood Function)
z1 = data(1,1);
z2 = data(1,2);
z = [z1; z2];
z_true = z * ones(1,N);
R = 10 * eye(2);
z_est  =  C*xpri';
v = z_true - z_est;
for i = 1:N
    q(i) = exp(-0.5 * (v(:,i)' * inv(R) * v(:,i)));
end

%% Normalizing the weights
 for i = 1:N
    wt(i) = q(i)/sum(q);
end

%% Resampling
M = length(wt);
Q = cumsum(wt);
indx = zeros(1, N);
T = linspace(0,1-1/N,N) + rand/N;
i = 1; j = 1;
while(i<=N && j<=M)
    while Q(j) < T(i)
        j = j + 1;
    end
    indx(i) = j;
     x1_post(i) = x1pri(j);
     x2_post(i) = x2pri(j);
     x3_post(i) = x3pri(j);
     x4_post(i) = x4pri(j);
    i = i + 1;
end

x1_pri_cum = cumsum(x1pri);
x2_pri_cum = cumsum(x2pri);
x3_pri_cum = cumsum(x3pri);
x4_pri_cum = cumsum(x4pri);
  
x1_cum = cumsum(x1_post);
x2_cum = cumsum(x2_post);
x3_cum = cumsum(x3_post);
x4_cum = cumsum(x4_post);
 
x_post = [x1_post x2_post x3_post x4_post];

%% Finding Centroid
x1_est = mean(x1_post);
x2_est = mean(x2_post);
x3_est = mean(x3_post);
x4_est = mean(x4_post);

%% plotting
figure(1);
subplot(2,1,1);
plot(x_post(:,1), 'LineWidth', 1);
xlabel("steps");
ylabel("Post height");

subplot(2,1,2);
plot(x1pri, "LineWidth",1);
plot(x_post(:,2), 'LineWidth', 1);
xlabel("steps");
ylabel("prior height");

sgtitle("Plot of h1 cloud");


figure(2);
subplot(2,1,1);
plot(x_post(:,2), 'LineWidth', 1);
xlabel("steps");
ylabel("Post height");

subplot(2,1,2);
plot(x2pri, "LineWidth",1);
plot(x_post(:,2), 'LineWidth', 1);
xlabel("steps");
ylabel("prior height");

sgtitle("Plot of h2 cloud");


figure(3);
subplot(2,1,1);
plot(x_post(:,3), 'LineWidth', 1);
xlabel("steps");
ylabel("Post height");

subplot(2,1,2);
plot(x3pri, "LineWidth",1);
plot(x_post(:,2), 'LineWidth', 1);
xlabel("steps");
ylabel("prior height");

sgtitle("Plot of h3 cloud");


figure(4);
subplot(2,1,1);
plot(x_post(:,4), 'LineWidth', 1);
xlabel("steps");
ylabel("Post height");

subplot(2,1,2);
plot(x4pri, "LineWidth",1);
plot(x_post(:,2), 'LineWidth', 1);
xlabel("steps");
ylabel("prior height");

sgtitle("Plot of h4 cloud");


%% 

figure(5);
histogram(x_post(:,1))
figure(6);
histogram(x1pri)