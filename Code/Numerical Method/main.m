function x = maim(delta, mu, tao, a, b, h, c,ti)
m = (b-a)/h;
start = 1;
glob1(1) = a;
tmp_a = a;

while start < m
    glob1(start+1) = funcion(a + (h*start));
    start = start+1;
end   

start = 1;

glob1(m+1) = 0;

while start < m;
    tmp_a = a + (start*h);
    start = start+1;
    val1 = glob1(start)*delta*ux(h, tmp_a, glob1, m, c);
    val2 = mu*(uxxx(h, tmp_a, glob1, m, c, val1));
    glob1(start) = (glob1(start) - tao*(val1 + val2));
    %if tmp_a == -3 || tmp_a == -2 || tmp_a == -1 || tmp_a == 0 || tmp_a == 1 || tmp_a == 2 || tmp_a == 3 || tmp_a == 4 || tmp_a == 5
    %       fprintf('For x = %d, then Exact Solution: %f,   Numerical Solution: %f\n',  tmp_a, exact_solution(tmp_a, ti), glob1(tmp_a));
    %end
end
x = glob1;
