close all; clear all;
%%
% Initialize plot
ax = axes;
hold(ax, 'on');
grid on;
xlabel(ax, 'Longitude');
ylabel(ax, 'Latitude');
zlabel(ax, 'Altitude (m)');

% Load the Geofence from KML File
geofence = kml2struct('AERPAW_UAV_Geofence_Phase_1.kml');

% Extract the geofence boundaries
geofence_lat = geofence.Lat;
geofence_lon = geofence.Lon;

% Define eNodeB locations
lat_eNBs = [35.7275, 35.728056, 35.725, 35.733056];
lon_eNBs = [-78.695833, -78.700833, -78.691667, -78.698333];
alt_eNBs = [10, 10, 10, 10]; % altitudes for eNodeBs
r_eNBs = [0.0006, 0.0006, 0.0008, 0.0035];

% Plot the eNodeBs
colors = ["m", "g", "b", "r"];
%colors = ["k", "k", "k", "k"];
for i = 1:4
        if i==1
        sym = "s"; %square
    elseif i==2
        sym = "^"; %triangle
    elseif i==3
        sym = "d"; %diamond
    elseif i==4
        sym = "p"; %cross
        end
    %plot3(ax, lon_eNBs(i), lat_eNBs(i), alt_eNBs(i), colors(i) + sym, 'MarkerSize', 10, 'MarkerFaceColor', colors(i));
    plot3(ax, lon_eNBs(i), lat_eNBs(i), alt_eNBs(i), colors(i) + sym, 'MarkerSize', 10, 'MarkerFaceColor', colors(i));
    % Plot "neighborhoods"
    plot_circle(lon_eNBs(i),lat_eNBs(i), r_eNBs(i),ax,colors(i));
end

%Drone starting position
global lat_drone lon_drone alt_drone
lat_drone = 35.7274823;
lon_drone = -78.6962747;
alt_drone = 0; % Start from ground (altitude = 0)
dronePlot = plot3(lon_drone, lat_drone, alt_drone, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

% Plot the geofence
plot(ax, geofence_lon, geofence_lat, 'r-', 'LineWidth', 2);
%% Sampling function
function [latSamples, lonSamples] = sampleNeighborhood(lat_center, lon_center, radius_deg, N, geofence_lat, geofence_lon)
    % Convert radius in degrees to meters
    radius_m = radius_deg * 111320; % approx conversion (valid near the equator)
    latSamples = zeros(N, 1);
    lonSamples = latSamples;

    % Look at 100 evenly spaced points on neighborhood edge
    thetas = linspace(0, 2*pi, 100); % +1 to close the circle
    thetas(end) = []; % remove duplicate point
    delta_lat = (radius_m / 6371000) * (180 / pi) * cos(thetas);
    delta_lon = (radius_m / 6371000) * (180 / pi) * sin(thetas);
    tmp_lat = lat_center + delta_lat;
    tmp_lon = lon_center + delta_lon;
    %Check which points are in geofence
    valid_idcs = inpolygon(tmp_lon, tmp_lat, geofence_lon, geofence_lat);
    % disp(valid_idcs)
    %Extract longest string of 1s
    %Assign groups
    group = (cumsum(~valid_idcs) + 1) .* valid_idcs;
    
    %Count lengths
    [run_lengths,~,ic] = unique(nonzeros(group)); 
    counts = accumarray(ic, 1); 
    
    %Find max
    [maxlen, maxidx] = max(counts);
    run_id = run_lengths(maxidx);
    
    %Find start index
    start_index = find(group == run_id, 1, 'first');
    stop_index = start_index + maxlen - 1;
    
    %Find 10 points within arc in geofence
    thetas = linspace(thetas(start_index),thetas(stop_index), 10);
    delta_lat = (radius_m / 6371000) * (180 / pi) * cos(thetas);
    delta_lon = (radius_m / 6371000) * (180 / pi) * sin(thetas);
    latSamples = lat_center + delta_lat;
    lonSamples = lon_center + delta_lon;
end

%% geofence 
% Function to check if a point is inside the geofence
is_in_geofence = @(lat, lon) inpolygon(lon, lat, geofence_lon, geofence_lat);

%% Haversine formula
%Distance between 2 points on a sphere given lat and lon
haversine = @(lat1, lon1, lat2, lon2) 2 * 6371000 * ...
    asin(sqrt(sin(deg2rad(lat2 - lat1) ./ 2).^2 + cos(deg2rad(lat1)) .* cos(deg2rad(lat2)) .* sin(deg2rad(lon2 - lon1) ./ 2).^2));
%% Solve TSPN with intlinprog
nStops = 5; % Number of stops to visit (num BS plus starting point)
stopsLon = [-78.6962747 -78.695833, -78.700833, -78.691667, -78.698333];
stopsLat = [35.7274823 35.7275, 35.728056, 35.725, 35.733056];

%Calculate total nodes and samples
numSamples=10;
total_samples = (nStops-1)*numSamples;
total_nodes = total_samples + 1;

%Generate all edges
idxs = nchoosek(1:total_nodes,2);

%Generate all samples for the neighborhoods
lat_samples = zeros(nStops-1, numSamples);
lon_samples = lat_samples;

for ii = 1:nStops-1
    [lat_samples(ii,:), lon_samples(ii,:)] = sampleNeighborhood(stopsLat(ii+1), stopsLon(ii+1), r_eNBs(ii), numSamples, geofence_lat, geofence_lon);
    scatter(ax, lon_samples(ii,:), lat_samples(ii,:), '*', 'MarkerFaceColor', colors(ii))
end
%Compile all samples (Add the takeoff/landing point)
lat_samples = reshape(lat_samples', 1, total_samples);
lon_samples = reshape(lon_samples', 1, total_samples);

lat_samples = [stopsLat(1) lat_samples];
lon_samples = [stopsLon(1) lon_samples];

%Indices of nodes in each neighborhood
N1_idcs = (1:10)+1;
%scatter(ax, lon_samples(N1_idcs), lat_samples(N1_idcs), '*', 'MarkerFaceColor', colors(1))
N2_idcs = N1_idcs + 10;
%scatter(ax, lon_samples(N2_idcs), lat_samples(N2_idcs), '*', 'MarkerFaceColor', colors(2))
N3_idcs = N2_idcs + 10;
%scatter(ax, lon_samples(N3_idcs), lat_samples(N3_idcs), '*', 'MarkerFaceColor', colors(3))
N4_idcs = N3_idcs + 10;
%scatter(ax, lon_samples(N4_idcs), lat_samples(N4_idcs), '*', 'MarkerFaceColor', colors(4))

%Remove edges within a neighborhood
remove_idcs=[];
for i = 1:length(idxs)
    if(any(N1_idcs(:) == idxs(i,1)) && any(N1_idcs(:) == idxs(i,2)))
        %Both nodes of edge are in N1. Remove it.
        remove_idcs = [remove_idcs i];
    elseif(any(N2_idcs(:) == idxs(i,1)) && any(N2_idcs(:) == idxs(i,2)))
        remove_idcs = [remove_idcs i];
    elseif(any(N3_idcs(:) == idxs(i,1)) && any(N3_idcs(:) == idxs(i,2)))
        remove_idcs = [remove_idcs i];
    elseif(any(N4_idcs(:) == idxs(i,1)) && any(N4_idcs(:) == idxs(i,2)))
        remove_idcs = [remove_idcs i];
    end
end
idxs(remove_idcs,:) = [];

%Calculate all distances
dist = haversine(lat_samples(idxs(:,1)), lon_samples(idxs(:,1)), lat_samples(idxs(:,2)), lon_samples(idxs(:,2)));
lendist = length(dist);
n = lendist;

%To enforce condition on both edges and nodes, x is now of length
%n+total_nodes with shape [edges y]
intcon = 1:n+total_nodes;
A_deg = zeros(total_nodes, n+total_nodes);
for i = 1:total_nodes
    edgeMask = (idxs(:,1) == i) | (idxs(:,2) == i);
    A_deg(i, 1:n) = edgeMask';
    A_deg(i,n+i) = -2;
end
b_deg = zeros(total_nodes, 1);

% A_deg_neg = -1*A_deg;
% b_deg_neg = zeros(total_nodes, 1);

A_group = zeros(nStops - 1, n + total_nodes);
A_group(1, n + N1_idcs) = 1;
A_group(2, n + N2_idcs) = 1;
A_group(3, n + N3_idcs) = 1;
A_group(4, n + N4_idcs) = 1;

b_group = ones(nStops - 1, 1);

A_start = zeros(1, n + total_nodes);
A_start(n + 1) = 1; % y_1
b_start = 1;

% Combine constraints
A = [];
b = [];

% Equality constraints
Aeq = [A_deg; A_group; A_start];
beq = [b_deg; b_group; b_start];

%Objective
f = [dist'; zeros(total_nodes, 1)];
lb = zeros(1,n+total_nodes);
ub1 = ones(1,n);
ub2 = 2*ones(1,total_nodes);
ub=[ub1 ub2];
%Solve TSPN
opts = optimoptions('intlinprog','Display','iter','Heuristics','advanced');
[xopt, fval, exitflag, output] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, opts);

% Extract solution
x_edges = xopt(1:n);
y_nodes = xopt(n + 1:end);

% Plot selected edges
selected_edges = idxs(find(round(x_edges)), :);
for i = 1:size(selected_edges, 1)
    plot(ax, [lon_samples(selected_edges(i,1)), lon_samples(selected_edges(i,2))], ...
              [lat_samples(selected_edges(i,1)), lat_samples(selected_edges(i,2))], ...
              'k--', 'LineWidth', 1.5);
end

% Mark selected nodes
selected_nodes = find(round(y_nodes));
plot(ax, lon_samples(selected_nodes), lat_samples(selected_nodes), 'ko', 'MarkerFaceColor', 'k');
ordered_nodes = zeros(length(selected_nodes),1);
ordered_nodes(1) = selected_edges(1,1);
ordered_nodes(2) = selected_edges(1,2);
selected_edges(1,:) = [];
for i = 3:5
    idx = find(selected_edges(:,1) == ordered_nodes(i-1));
    if any(idx)
        ordered_nodes(i) = selected_edges(idx,2);
        selected_edges(idx,:) = [];
    else
        idx = find(selected_edges(:,2) == ordered_nodes(i-1));
        ordered_nodes(i) = selected_edges(idx,1);
        selected_edges(idx,:) = [];
    end
end
ordered_nodes(1) = [];
disp(ordered_nodes);

%Waypoint for traveling to and from LW3
waypoint_lon = -78.6943;
waypoint_lat = 35.7249;
%Insert waypoint before and after LW3 node in ordered nodes
%Find which node is selected for N3 and which order it comes
[ord_idx, N3_idx] = find(ordered_nodes == N3_idcs);
lat_stops = zeros(1,nStops-1);
lon_stops = zeros(1,nStops-1);
ii = 1;
ord_i = 1;
while ii <= length(lat_stops)
    if ii == ord_idx
        lat_stops(ii) = waypoint_lat;
        lat_stops(ii+1) = lat_samples(ordered_nodes(ord_i));
        lat_stops(ii+2) = waypoint_lat;
        lon_stops(ii) = waypoint_lon;
        lon_stops(ii+1) = lon_samples(ordered_nodes(ord_i));
        lon_stops(ii+2) = waypoint_lon;
        ii = ii + 3;
    else
        lat_stops(ii) = lat_samples(ordered_nodes(ord_i));
        lon_stops(ii) = lon_samples(ordered_nodes(ord_i));
        ii = ii + 1;
    end
    ord_i = ord_i + 1;
end
plot(ax, lon_stops, lat_stops, 'o', 'MarkerFaceColor', 'k');

%% Functions
function p = plot_circle(x,y,r,ax,i)
    theta = 0:pi/50:2*pi;
    xs = r * cos(theta) + x;
    ys = r * sin(theta) + y;
    p = plot(ax,xs,ys, '--', 'color', i);
end