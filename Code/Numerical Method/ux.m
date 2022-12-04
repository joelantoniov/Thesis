function x = ux(h, a, glob, m, c)
start = 1;
sum = 0;

while start <= m
    val1 = phi(a, c, glob, start) - phi(a+h, c, glob, start);
    val2 = glob(start+1) - glob(start);
    sum = sum + (val1/h)*(val2);
    start = start +1;
end    
x = (sum/2);