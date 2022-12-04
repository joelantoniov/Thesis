function x = phi(y, c, glob, idy)
fir = (y-glob(idy));
sec = sqrt((y-glob(idy))^2 + c^2);
x = fir/sec;