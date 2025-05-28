function [newX, newY, newZ] = densifyTrajectory(x, y, z, N)
    % Initialize new arrays to hold the dense points
    newX = [];
    newY = [];
    newZ = [];
    
    % Loop through each pair of points
    for i = 1:length(x)-1
        % Get the current and next point
        x0 = x(i);
        y0 = y(i);
        z0 = z(i);
        x1 = x(i+1);
        y1 = y(i+1);
        z1 = z(i+1);
        
        % Create N points between (x0, y0, z0) and (x1, y1, z1)
        for j = 0:N
            % Interpolate the parameters
            t = j / N; % Normalized parameter
            newX(end+1) = (1-t) * x0 + t * x1;
            newY(end+1) = (1-t) * y0 + t * y1;
            newZ(end+1) = (1-t) * z0 + t * z1; % Interpolating z values
        end
    end
    
    % Optionally, add the last point to the new arrays
    newX(end+1) = x(end);
    newY(end+1) = y(end);
    newZ(end+1) = z(end);
end
