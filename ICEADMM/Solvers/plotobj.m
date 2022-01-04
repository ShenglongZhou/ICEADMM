function  plotobj(noiter,obj1,obj2,txt)
    figure('Renderer', 'painters', 'Position',[1100 400 400 320]);
    axes('Position', [0.13 0.14 0.85 0.8] );
    colors = {'#173f5f','#20639b','#3caea3','#f6d55c','#ed553b'}; 
    if  noiter >100
        iter = log2(1:noiter);
    else
        iter = 1:noiter;    
    end
    h = plot(iter,obj1(1:end),iter,obj2(1:end));  grid on
    h(1).LineWidth  = 1.5;   h(2).LineWidth  = 1.5;       
    h(1).LineStyle  = '-';   h(2).LineStyle  = ':'; 
    h(1).Color = colors{3};  h(2).Color = colors{5};
    legend('$f(y^k)$','$F(X^k)$',...
           'Interpreter','latex','location','SouthWest')
    xlabel('Iterations'); ylabel('Objective'); title(txt);
    if  noiter >100
        a = round(iter(end));
        set(gca, 'XTick', 0:a);
        for j = 1:a+1; xtl{j} = ['2^{' num2str(j-1),'}']; end
        set(gca, 'XTickLabel', xtl);
        axis([0 a+1 min([obj1 obj2])/1.01 max([obj1 obj2])*1.01]);
    end
    
end

