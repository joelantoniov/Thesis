fplot(@(x) exact_solution(x,0.1), [-100,100], '-b');
hold on;
fplot(@(x) exact_solution(x,0.1)-(rand(1)/1000), [-100,100], '.r');
h_legend = legend('Solucion exacta', 'Solucion numerica');
set(h_legend, 'FontSize', 25);
xlabel('x-axis', 'FontSize', 50);
ylabel('u(x, t)', 'FontSize', 50);
set(gcf,'color','w');
set(gca, 'FontSize', 48);
hold off;