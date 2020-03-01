for i=0:0.5:40
    for j=[0.1 0.7]
        try
        phase_transition(i,j)
        catch
        end
        end
end
    