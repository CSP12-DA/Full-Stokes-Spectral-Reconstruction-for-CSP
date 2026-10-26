
%% ============================================================
% CCSP Stokes Spectrum Reconstruction (Legendre + DCT + CVX)
%
% 1. Import the measured modulation spectrum
% 2. Construct the PSIM modulation matrix
% 3. Reconstruct the Stokes parameters using a sparse
%    dictionary composed of Legendre polynomials and DCT bases
% 4. Display S0, s1, s2, and s3 in a 2-by-2 subplot layout
%
% Requires the MATLAB CVX toolbox
%% ============================================================

clear; clc; close all;

%% ==================== 1. Parameter Settings ====================

% Path to the measured modulation spectrum CSV file
% Column 1: Wavelength; Column 2: Modulation intensity
MODULATION_CSV_PATH = 'D:\modulation_spectrum.csv';

% Wavelength range and sampling interval (nm)
LAMBDA_MIN = 400;
LAMBDA_MAX = 700;
LAMBDA_STEP = 1;

% CCSP dictionary parameters
NUM_LEGENDRE = 5;        % Number of Legendre polynomial bases (P1-P5)
NUM_DCT = 240;           % Number of DCT basis functions
BETA = 5e-3;             % L1 sparsity regularization weight

% PSIM physical parameters
DELTA_N1 = 0.0090;       % Birefringence of the first retarder
DELTA_N2 = 0.0090;       % Birefringence of the second retarder
D1_NM = 4000000;         % Thickness of the first retarder (nm)
D2_NM = 8000000;         % Thickness of the second retarder (nm)
THETA1_DEG = 0;          % Fast-axis angle of the first retarder (deg)
THETA2_DEG = 45;         % Fast-axis angle of the second retarder (deg)
ANALYZER_DEG = 0;        % Linear polarizer angle (deg)

EPSILON = 1e-8;          % Small constant to prevent division by zero


%% ==================== 2. Import the Measured Modulation Spectrum ====================

% Read the CSV file
data = readmatrix(MODULATION_CSV_PATH);

% Remove rows containing only NaN values caused by headers or blank lines
data = data(~all(isnan(data),2),:);

% Check the number of columns
if size(data,2) < 2
    error('The CSV file must contain at least two columns: wavelength and modulation intensity.');
end

% Extract wavelength and modulation intensity
wavelength = data(:,1);
measured_spectrum = data(:,2);

% Convert to column vectors
wavelength = wavelength(:);
measured_spectrum = measured_spectrum(:);

% Check data validity
if any(~isfinite(wavelength)) || ...
        any(~isfinite(measured_spectrum))
    error('The modulation spectrum contains NaN or Inf values.');
end

% Check whether the wavelengths are strictly increasing
if any(diff(wavelength) <= 0)
    error('The input wavelengths must be strictly increasing.');
end

% Check the wavelength range and sampling interval
expected_wavelength = (LAMBDA_MIN:LAMBDA_STEP:LAMBDA_MAX)';

if length(wavelength) ~= length(expected_wavelength) || ...
        any(abs(wavelength-expected_wavelength) > 1e-6)
    error('The input wavelengths must range from %.1f to %.1f nm with a sampling interval of %.1f nm.', ...
        LAMBDA_MIN,LAMBDA_MAX,LAMBDA_STEP);
end

% Use the original sampling points directly
N = length(wavelength);

fprintf('Measured modulation spectrum imported successfully. Number of sampling points: %d\n',N);


%% ==================== 3. Construct the PSIM Modulation Coefficients ====================

% Calculate the four modulation coefficients using the Mueller model
% I(lambda) = a0*S0 + a1*S1 + a2*S2 + a3*S3

psim_coeffs = build_psim_coefficients( ...
    wavelength,DELTA_N1,DELTA_N2, ...
    D1_NM,D2_NM,THETA1_DEG,THETA2_DEG,ANALYZER_DEG);


%% ==================== 4. Construct the Sparse Dictionary ====================

reconstruction_timer = tic;

% Map the wavelength sampling positions to [-1, 1]
x = linspace(-1,1,N)';

% ---------- 4.1 Legendre Polynomial Bases ----------

B_legendre = zeros(N,NUM_LEGENDRE);

P_prev = ones(N,1);      % P0
P_curr = x;              % P1

for k = 1:NUM_LEGENDRE

    if k == 1
        v = P_curr;
    else
        % Legendre polynomial recurrence relation
        P_next = ((2*k-1)*x.*P_curr-(k-1)*P_prev)/k;
        P_prev = P_curr;
        P_curr = P_next;
        v = P_curr;
    end

    % L2-normalize each basis function
    B_legendre(:,k) = v/max(norm(v),EPSILON);
end

% ---------- 4.2 DCT Orthogonal Bases ----------

% Analytically construct the orthogonal IDCT-II basis
% without requiring an additional toolbox
n = (0:N-1)';
k = 0:NUM_DCT-1;

B_dct = sqrt(2/N)*cos(pi*(n+0.5)*k/N);

% Normalization factor for the constant DCT basis
B_dct(:,1) = B_dct(:,1)/sqrt(2);

% ---------- 4.3 Combine the Dictionaries ----------

B = [B_legendre,B_dct];
NUM_BASIS = size(B,2);


%% ==================== 5. Construct the CCSP Measurement Matrix ====================

% A = [diag(a0)*B, diag(a1)*B, diag(a2)*B, diag(a3)*B]
% Use element-wise multiplication instead of diagonal matrix operations

A = zeros(N,4*NUM_BASIS);

for j = 1:4
    cols = (j-1)*NUM_BASIS+(1:NUM_BASIS);
    A(:,cols) = bsxfun(@times,psim_coeffs(:,j),B);
end


%% ==================== 6. CVX Sparse Optimization ====================

% Optimization objective:
% min 0.5*||A*c-I||_2^2 + BETA*||c||_1
%
% First term: Modulation spectrum data-fitting loss
% Second term: Sparsity regularization
% c: Combined sparse coefficients of the four Stokes components

cvx_begin quiet

    variable coefficients(4*NUM_BASIS,1)

    minimize( ...
        0.5*sum_square(A*coefficients-measured_spectrum) ...
        + BETA*norm(coefficients,1) ...
    )

cvx_end

% Check the solver status
if ~any(strcmpi(cvx_status,{'Solved','Inaccurate/Solved'}))
    error('CVX optimization failed: %s',cvx_status);
end


%% ==================== 7. Reconstruct the Stokes Spectra ====================

% Recover the sparse coefficients of the four Stokes components
C = reshape(coefficients,NUM_BASIS,4);

% Reconstruct S0-S3 using the sparse dictionary
S = B*C;

% Extract the reconstructed S0
S0 = S(:,1);

% Calculate the normalized Stokes parameters
% s1 = S1/S0, s2 = S2/S0, s3 = S3/S0
% Use |S0| with a lower bound to prevent division by zero
s = bsxfun(@rdivide,S(:,2:4),max(abs(S0),EPSILON));

% Record the reconstruction time
reconstruction_time = toc(reconstruction_timer);

fprintf('CVX status: %s\n',cvx_status);
fprintf('Reconstruction time: %.4f s\n',reconstruction_time);


%% ==================== 8. Visualize the Reconstruction Results ====================

% Display the results in a 2-by-2 subplot layout
figure('Color','w','Position',[100,100,1100,750]);

% Reconstructed data for the four subplots
plot_data = [S0,s];

% Subplot titles
names = {'S_0','s_1','s_2','s_3'};

for j = 1:4

    subplot(2,2,j);

    plot(wavelength,plot_data(:,j),'-','LineWidth',1.5);

    xlabel('Wavelength (nm)');
    ylabel(names{j});
    title(['Reconstructed ',names{j}]);

    xlim([LAMBDA_MIN,LAMBDA_MAX]);
    grid on;

end


%% ============================================================
% Local Function: Calculate PSIM Mueller Modulation Coefficients
%% ============================================================

function coefficients = build_psim_coefficients( ...
    wavelength,dn1,dn2,d1,d2,theta1,theta2,alpha)

    % Convert angles from degrees to radians
    theta1 = deg2rad(theta1);
    theta2 = deg2rad(theta2);
    alpha = deg2rad(alpha);

    % Mueller matrix of the linear polarizer
    c = cos(2*alpha);
    s = sin(2*alpha);

    P = 0.5*[ ...
        1,c,s,0;
        c,c*c,c*s,0;
        s,c*s,s*s,0;
        0,0,0,0];

    % Initialize the modulation coefficients
    N = length(wavelength);
    coefficients = zeros(N,4);

    for i = 1:N

        % Phase retardations of the two retarders
        delta1 = 2*pi*dn1*d1/wavelength(i);
        delta2 = 2*pi*dn2*d2/wavelength(i);

        % Mueller matrices of the two retarders
        R1 = retarder(delta1,theta1);
        R2 = retarder(delta2,theta2);

        % Optical path sequence: R1 -> R2 -> Polarizer
        M = P*R2*R1;

        % The first row contains the four Stokes modulation coefficients
        coefficients(i,:) = M(1,:);

    end
end


%% ============================================================
% Local Function: Mueller Matrix of a Linear Retarder
%% ============================================================

function M = retarder(delta,theta)

    c = cos(2*theta);
    s = sin(2*theta);
    cd = cos(delta);
    sd = sin(delta);

    M = [ ...
        1,0,0,0;
        0,c*c+s*s*cd,c*s*(1-cd),-s*sd;
        0,c*s*(1-cd),s*s+c*c*cd,c*sd;
        0,s*sd,-c*sd,cd];

end
