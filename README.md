# DCOMM-CODES

EXP:1 AIM:

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

EXP
