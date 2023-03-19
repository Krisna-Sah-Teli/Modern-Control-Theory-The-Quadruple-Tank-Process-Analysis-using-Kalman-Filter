clear all
close all

measurementData=readmatrix('Measurements.xlsx');

% Data from the paper:
A1 = 28;        % cm2
A2 = 32;        % cm2
A3 = 28;        % cm2
A4 = 32;        % cm2
kc = 0.50;      % V/cm
g=981;

% Initial heights and voltage
h10 = 12.4;
h20 = 12.7;
h30 = 1.8;
h40 = 1.4;
v10= 3;
v20= 3;
Xo=[h10; h20; h30; h40];
Uo=[v10; v20];

% syms A1 A2 A3 A4 T1 T2 T3 T4 gamma1 gamma2 k1 k2 kc;
a1= 0.071;
a2= 0.057;
a3= 0.071;
a4= 0.057;

%Minimum Phase Characteristics Values: 
gamma1 = 0.7;   % dimensionless
gamma2 = 0.6 ;  % dimensionless
k1 = 3.33;      % cm3/Vs
k2 = 3.35;      % cm3/Vs
T1 = 62;        % s
T2 = 90;        % s
T3 = 23;        % s
T4 = 30;

Ac = [-1/T1 0 A3/(A1*T3) 0; 0 -1/T2 0 A4/(A2*T4); 0 0 -1/T3 0; 0 0 0 -1/T4];
Bc = [(gamma1*k1)/A1 0 ; 0 (gamma2*k2)/A2; 0 ((1-gamma2)*k2)/A3; ((1-gamma1)*k1)/A4 0];
Cc = [kc 0 0 0; 0 kc 0 0];
Dc=[0];

%%
X_post=[1; 1; 1; 1];
P_post=100 * eye(4);
Q=3*eye(4);
R=10*eye(2);

%% Storage definition and initialization
X_post_store= [];
X_prior_store= [];
P_post_store= [];
P_prior_store= [];
KG_store= [];
Innov_store= [];
Residual_store= [];
Norm_store= [];
v_store=[];
Covariance_post_store= [];
Covariance_prior_store= [];

%% Discretization
Ts=0.1;      %Sampling Time
csys= ss(Ac,Bc,Cc,Dc);
dsys= c2d(csys, Ts);
[A B H D] = ssdata(dsys);

%% 
for i=1:10000    
    % Computation of input u using forward difference equation
    h1=measurementData(i,1);
    h2=measurementData(i,2);
    h3=measurementData(i,3);
    h4=measurementData(i,4);
    
    devH1= ( measurementData(i+1,1)- measurementData(i,1) )/ Ts;
    devH2= ( measurementData(i+1,2)- measurementData(i,2) )/ Ts;
    
    v1=(devH1 + (a1*sqrt(2*g*h1))/A1 - (a3*sqrt(2*g*h3))/A1)*(A1/(gamma1*k1));
    v2=(devH2 + (a2*sqrt(2*g*h2))/A2 - (a4*sqrt(2*g*h4))/A4)*(A2/(gamma2*k2));   
    
    U=[v1; v2];

    %Prediction
    X_prior=A*(X_post - Xo) + B*(U-Uo) + Xo;
    P_prior=A*P_post*transpose(A) + Q;
    
    %Calculate Kalman Gain
    KG=P_prior*transpose(H)*(inv(H*P_prior*transpose(H) + R));    

    %Measurement Data (True data)
    Z_true = H*transpose(measurementData(i,1:4));

    %Estimated Measurement
    Z_est=H*X_prior;

    %Error between estimated measurement and true measurement
    Z_error=Z_true - Z_est;

    %Update with kalman gain    
    X_post= X_prior + KG*Z_error;
    P_post=P_prior-KG*H*P_prior;
    
    norm1= norm(P_prior);
    norm2= norm(P_post);
    if abs(norm2-norm1)<= 5e-3
        disp("This is converged")
    end


    %Storage
    X_prior_store(i, 1:4, 1)= X_prior;
    X_post_store(i, 1:4, 1)= X_post;

    P_prior_store(i, 1:4, 1:4)= P_prior;
    P_post_store(i, 1:4, 1:4)= P_post;

    KG_store(i, 1:4, 1:2)= KG;

    Innov_store(i, 1:2, 1)= Z_error;
    
    residual= Z_true- H* X_post;
    Residual_store(i, 1:2, 1)= residual;

    Covariance_post_store(i)= trace(P_post);
    Covariance_prior_store(i)= trace(P_prior);
    v_store(i, 1:2)= [v1, v2];
    
    
end

%% plot for x prior
t=10000;
figure(1);
subplot(2,1,1)
plot(X_prior_store(1:1000,1), 'b-', 'LineWidth', 2);
hold on; 
plot(X_prior_store(1:1000,2), 'r-', 'LineWidth', 1);
legend("h1", "h2")
xlabel("steps");
ylabel("height");


subplot(2,1,2)
plot(X_prior_store(1:1000,3), 'b.', 'LineWidth', 2);
hold on; 
plot(X_prior_store(1:1000,4), 'r.', 'LineWidth', 2);
legend("h3", "h4")
xlabel("steps");
ylabel("height");
sgtitle("The X prior Plot")


%% plot for x post

figure(2);
subplot(2,1,1)
plot(X_post_store(1:1000,1), 'b-', 'LineWidth', 2);
hold on; 
plot(X_post_store(1:1000,2), 'r:', 'LineWidth', 1);
legend("h1", "h2")
xlabel("steps");
ylabel("height");

subplot(2,1,2);
plot(X_post_store(1:1000, 3), 'b-', 'LineWidth', 2);
hold on; 
plot(X_post_store(1:1000, 4), 'r:', 'LineWidth', 2);
legend("h3", "h4")
xlabel("steps");
ylabel("height");

sgtitle("The X post Plot")

%% common plot for X proir and x post

figure(3)
subplot(2,1,1);
plot(X_post_store(1:1000,1:2), '*', 'LineWidth', 2);
hold on;
plot(X_prior_store(1:1000,1:2), '.', 'LineWidth', 2);
legend("X post h1", "X post h2", "X prior h1", "X prior h2");
xlabel("steps");
ylabel("height");
title("Plot of X post and X prior");

subplot(2,1,2);
plot(X_post_store(1:1000,3:4), '*', 'LineWidth', 2);
hold on;
plot(X_prior_store(1:1000,3:4), '.', 'LineWidth', 2);
legend("X post h3", "X post h4", "X prior h3", "X prior h4");
xlabel("steps");
ylabel("height");
title("Plot of X post and X prior");




%% plot of P post

figure(4);
subplot(3,1,1);
plot(Covariance_prior_store(1:1000), 'r.', 'LineWidth', 2);
xlabel("steps");
ylabel("trace");
title("Prior Covariance plot");

subplot(3,1,2); 
plot(Covariance_post_store(1:1000), 'b.', 'LineWidth', 1);
xlabel("steps");
ylabel("trace"); 
title("Posterior Covariance plot");

subplot(3,1,3);
plot(Covariance_prior_store(1:1000), 'r*', 'LineWidth', 2);
hold on; 
plot(Covariance_post_store(1:1000), 'b.', 'LineWidth', 1);
xlabel("steps");
ylabel("trace");
legend("Pprior", "Ppost");
title("The Covariance Plot");


%% Kalman gain plot

figure(5);
plot(KG_store(1:100,1,1), 'b-', 'LineWidth', 2);
xlabel("steps");
ylabel("Kalman gain");
title("The plot of 1st row 1st column element of Kalman Gain matrix of each step");


%% Innovative and residual plot
figure(6);
subplot(2,1,1);
plot(Innov_store(1:100,1), 'b-', 'LineWidth', 2);
hold on; 
plot(Residual_store(1:100,1), 'r:', 'LineWidth', 2);
legend("h1 Innov", "h2 Resid")
xlabel("steps");
ylabel("height diff");


subplot(2,1,2);
plot(Innov_store(1:100,2), 'b-', 'LineWidth', 2);
hold on; 
plot(Residual_store(1:100,2), 'r:', 'LineWidth', 2);
legend("h1 Innov", "h2 Resid")
xlabel("steps");
ylabel("height diff");

sgtitle("The Innovative and Residual Plot");


%%  The End


