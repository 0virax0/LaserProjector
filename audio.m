amp=10; 
fs=20500;  % sampling frequency
duration=1;
freqL=100;
freqR=100;
values=[(0:1/fs:duration)' (0:1/fs:duration)'];

signal=[amp.*sin(2.*pi.* freqL.*values(:,1))  amp.*sin(2.*pi.*freqR.*values(:,2) + pi/2)]; %sine waves
%draw a square
% /
frame = 0.0025;
x = 0:1/fs:frame;   x = x';
rise = (amp .* x ./ frame) - ones(length(x),1) .* (amp ./2);
fall = -rise;
low = ones(length(x),1) .* (-amp/2);
high = -low;
squareWaveL = [rise; high; fall; low]; squareWaveL = squareWaveL(1:end-2);
squareWaveR = [low; rise; high; fall]; squareWaveR = squareWaveR(1:end-2);
x = 0:1/fs:(frame*4) ;   x = x';
%signal = [repmat(squareWaveL, duration/frame,1) repmat(squareWaveR, duration/frame,1)];
figure(1);
plot(x, squareWaveL, x, squareWaveR)

simTres = 0.005;  %simulation time resolution
persistenceFrames = 2;
%for c = 1: duration/simTres - persistenceFrames  
 %   figure(2);
  %  plot(signal(fs*simTres*c:fs*simTres*(c+persistenceFrames),1), signal(fs*simTres*c:fs*simTres*(c+persistenceFrames),2));
   % pause(0.001)
%end
sound(signal)