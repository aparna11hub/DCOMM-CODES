# DCOMM-CODES

EXP:1 
AIM:Digital sampling of an analog sine wave.

clc;
clf;
scf();

// Common parameters
duration = 2;
frequency = 2;
amplitude = 1;

// Sampling rates to test
Fs = [100, 10, 4, 3, 1];

for i = 1:length(Fs)
    t = 0:1/Fs(i):duration;
    sine_wave = amplitude * sin(2 * %pi * frequency * t);
    
    subplot(3, 2, i);
    plot(t, sine_wave, '.-');
    xlabel("Time (s)");
    ylabel("Amplitude");
    title(msprintf("Sine Wave %d (2 Hz, Fs = %d Hz)", i, Fs(i)));
    xgrid();
end

EXP:2 
AIM:Calculate and understand the concept of entropy .

clc();
p = input("Enter the probability array in square brackets : ");
n = length(p);
H = 0;

for i = 1:n
    H = H + p(i) * log2(1 / p(i));
end

disp("Entropy =");
disp(H);

EXP:4 
AIM:Vertical redundancy Check (VRC) code generation and error detection

clc();

// --- Transmitter ---
a = input("Enter your 7-bit message signal : ");
parity_bit = modulo(sum(a), 2);     // Count 1s and take mod 2
a(8) = parity_bit;                  // Add even parity bit
disp("Data of even parity to be transmitted:");
disp(a);

// --- Receiver ---
r = input("Enter 8-bit received data : ");
if modulo(sum(r), 2) == 0 then
    disp("No Error");
    disp("Data of even parity received :");
    disp(r(1:7));
else
    disp("Error Detected");
end

EXP:8
AIM:Implement Amplitude Shift Keying (ASK) modulation.

clc;
clear;
close;

f = input("Enter the analog carrier frequency in Hz = ");
t = 0:0.00001:0.5;
x = cos(2 * %pi * f * t);   // Carrier signal
l = input("Enter the digital binary data = ");

// Generate message and ASK signal
message = [];
ASK = [];

for i = 1:length(l)
    bit = l(i);
    m_s = bit * ones(1, length(t));  // 1 → ones, 0 → zeros
    message = [message, m_s];
    ASK = [ASK, bit * x];            // Multiply bit with carrier
end

// Plot message, carrier, and ASK waveform
subplot(3, 1, 1);
plot(message);
xlabel("Time");
ylabel("Amplitude");
title("Binary Message Signal");

subplot(3, 1, 2);
plot(repmat(x, 1, length(l)));
xlabel("Time");
ylabel("Amplitude");
title("Carrier Signal");

subplot(3, 1, 3);
plot(ASK);
xlabel("Time");
ylabel("Amplitude");
title("ASK Waveform");

EXP:9
AIM:To implement various line codes.

clc;
clear;
close;

data = input("Enter bit sequence: ");
N = length(data);
rz = [];

for i = 1:N
    rz = [rz, data(i) * [ones(1,50) zeros(1,50)]];
end

t = 0:0.01:N;

subplot(2,1,1);
plot2d3(0:N-1, data);
title("Input Data");

subplot(2,1,2);
plot2d(t(1:length(rz)), rz);
title("Unipolar RZ");


