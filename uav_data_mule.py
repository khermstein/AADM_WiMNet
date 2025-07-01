import asyncio
import math
import datetime
import csv
import numpy as np
import time 
import ast
import random
import os

from typing import List, TextIO
from struct import unpack
from argparse import ArgumentParser

import scipy.optimize as opt
import matplotlib.pyplot as plt
from pykml import parser
from matplotlib import path
from itertools import groupby, combinations

from aerpawlib.runner import StateMachine
from aerpawlib.aerpaw import AERPAW_Platform
from aerpawlib.vehicle import Vehicle, Drone
from aerpawlib.runner import state, timed_state, background #, in_background
from aerpawlib.util import Coordinate, VectorNED
from aerpawlib.safetyChecker import SafetyCheckerClient

from get_data_volumes import getDataVolumes
from get_signal_strength import getSignalStrengths
from get_download_data import getDownloadData
from read_plan_file import extract_waypoints


STEP_SIZE = 10  # when going forward - how far, in meters

SEARCH_ALTITUDE = 22 # in meters


class DataMule(StateMachine):
    start_time = None
    flight_time = 600

    waypoints = []
    update_bs_id =""
    nextWaypointIndex = 0
    lastWaypointIndex = 0  
    nextBS = [0]
    waitTime = 0
    uav_altitude = 25
    angles = []
    cnt = 0
    updateBS = True
    updateWaypoint = True
    t1 = None
    max_speed = 10 #mps
    target_speed = 10
    BS_time = {}
    volumes = {}
    curr_BS = -1
    time_flag = False
    download_flag = False
    cancel_flight_flag = False

    lat_eNBs = [35.7275, 35.728056, 35.725, 35.733056]
    lon_eNBs = [-78.695833, -78.700833, -78.691667, -78.698333]
    alt_eNBs = [10, 10, 10, 10] # altitudes for eNodeBs
    r_eNBs = [0.00028, 0.0005, 0.0008, 0.003] #Radius for eNobeB neighborhoods
    dummy_waypoint_lon = -78.6943
    dummy_waypoint_lat = 35.7249

    _next_sample: float = 0
    _sampling_delay: float
    _cur_line: int
    _csv_writer: object
    _log_file: TextIO
    #vehicle.set_groundspeed(target_speed)

    def initialize_args(self, extra_args: List[str]):
        """Parse arguments passed to vehicle script"""
        # Default output CSV file for search data
        
        directory = "/root/Results"
        os.makedirs(directory, exist_ok=True)
        
        # default_file = (
            # f"GPS_DATA_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv"
        # )

        default_file = os.path.join(
            directory, 
            #f"GPS_DATA_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_vehicleOut.txt"
        )

        #print(f"File will be saved as: {default_file}")
        
        parser = ArgumentParser()
        parser.add_argument("--safety_checker_ip", help="ip of the safety checker server")
        parser.add_argument("--safety_checker_port", help="port of the safety checker server")
        parser.add_argument(
            "--skipoutput", help="don't dump gps data to a file", action="store_false"
        )
        parser.add_argument(
            "--output", help="log output file", required=False, default=default_file
        )
        parser.add_argument(
            "--samplerate",
            help="log sampling rate (Hz)",
            required=False,
            type=float,
            default=1,
        )        

        args = parser.parse_args(args=extra_args)

        #self.fake_radio = args.fake_radio
        self.safety_checker = SafetyCheckerClient(args.safety_checker_ip, args.safety_checker_port)
        #self.flight_time = datetime.timedelta(seconds=600)  # Default search time (10 minutes)
        
        self._sampling = args.skipoutput
        self._sampling_delay = 1 / args.samplerate
        
        if self._sampling:
            self._log_file = open(args.output, "w+")
            self._cur_line = sum(1 for _ in self._log_file) + 1
            self._csv_writer = csv.writer(self._log_file)
        

    def _dump_to_csv(self, vehicle: Vehicle, line_num: int, writer):
        """
        This function will continually log stats about the vehicle to a file specified by command line args
        """
        #print('dump_to_csv')
        pos = vehicle.position
        lat, lon, alt = pos.lat, pos.lon, pos.alt
        volt = vehicle.battery.voltage
        blevel = vehicle.battery.level
        timestamp = datetime.datetime.now()
        gps = vehicle.gps
        fix, num_sat = gps.fix_type, gps.satellites_visible
        if fix < 2:
            lat, lon, alt = -999, -999, -999
        vel = vehicle.velocity
        attitude = vehicle.attitude
        attitude_str = (
            "("
            + ",".join(map(str, [attitude.pitch, attitude.yaw, attitude.roll]))
            + ")"
        )

        # If you ever update this list of parameters logged please also change
        #  ../../../PostProcessing/log2csv.py    and
        #  ../../GPSLogger/gps_logger.py
        # to keep them in sync

        writer.writerow(
            [line_num, lon, lat, alt, attitude_str, vel, volt, timestamp, fix, num_sat]
        )

    @background
    async def periodic_dump(self, vehicle: Vehicle):        
        #await asyncio.sleep(1)
        await asyncio.sleep(self._sampling_delay)
        #await sleep(self._sampling_delay)
        if not self._sampling:
            return
        self._dump_to_csv(vehicle, self._cur_line, self._csv_writer)        
        #self._log_file.flush()
        #os.fsync(self._log_file)
        self._cur_line += 1

    def cleanup(self):
        if self._sampling:
            self._log_file.close()
                
    def getToken(self):
        try:
            with open('tn.txt', 'r') as f:
                return f.read().strip()  # Read and return the token
        except FileNotFoundError:
            print("Error: The token file 'token.txt' was not found.")
            return None  # Return None if the file does not exist
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None  # Handle any other exceptions
    
    def setBSForBroadcast(self, bs_id): 
        with open('bc.txt', 'w') as f:
            f.write(bs_id)          

    def checkSNR(self):    
        return getSignalStrengths()
        
    def checkDownload(self):
        return getDownloadData()            
        
    #This function will provide the individual data volume with corresponding Base station or LakeWheelers
    def checkDataVolumes(self):
        return getDataVolumes()
        
    def extractWaypointsFromPlanFile(self, plan_file):
        return extract_waypoints(plan_file)
    
    def inpolygon(self, xq, yq, xv, yv):
        shape = xq.shape
        #xq = xq.reshape(-1)
        #yq = yq.reshape(-1)
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)
        q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
        p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
        return p.contains_points(q).reshape(shape)
    
    def sampleNeighborhood(self, lat_center, lon_center, radius_deg, N, geofence_lat, geofence_lon):
        radius_m = radius_deg * 111320
        latSamples = np.zeros((N,1))
        lonSamples = np.copy(latSamples)

        #Look at 100 evenly spaced points on neighborhood edge
        thetas=np.linspace(0,2*np.pi, 100)
        delta_lat = (radius_m / 6371000) * (180/np.pi) * np.cos(thetas)
        delta_lon = (radius_m / 6371000) * (180/np.pi) * np.sin(thetas)
        tmp_lat = np.array(lat_center+delta_lat)
        tmp_lon = np.array(lon_center+delta_lon)
        
        valid_idcs = self.inpolygon(tmp_lon, tmp_lat, geofence_lon, geofence_lat)
        x = [1 if valid else 0 for valid in valid_idcs]
        # print([k for k,g in groupby(x)])
        # print([list(g) for k,g in groupby(x)])
        max_group = 0
        max_idx = 0
        curr_idx = 0
        for k,g in groupby(x):
            length = len(list(g))
            if k == 1:
                if length > max_group:
                    max_group = length
                    max_idx = curr_idx
            curr_idx += length
        
        thetas = np.linspace(thetas[max_idx], thetas[max_idx + max_group -1], N)
        delta_lat = (radius_m / 6371000) * (180/np.pi) * np.cos(thetas)
        delta_lon = (radius_m / 6371000) * (180/np.pi) * np.sin(thetas)
        latSamples = lat_center+delta_lat
        lonSamples = lon_center+delta_lon
        return lonSamples, latSamples
    
    def haversine(self, lat1, lon1, lat2, lon2):
        return 2 * 6371000 * np.arcsin(np.sqrt(np.sin(np.deg2rad(lat2 - lat1) / 2)**2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2))

    def readGeofence(self, filepath):
        with open(filepath, 'r', encoding="utf-8") as f:
            root = parser.parse(f).getroot()
        coords = root.Document.Placemark.Polygon.outerBoundaryIs.LinearRing.coordinates
        lonlatel = coords.text.strip().split(' ')
        #print(lonlatel)
        geofence_lon = []
        geofence_lat = []
        for xyz in lonlatel:
            lon, lat, _ = xyz.split(',')
            geofence_lon.append(float(lon))
            geofence_lat.append(float(lat))
        geofence_lon  = np.reshape(np.array(geofence_lon), (-1,1))
        geofence_lat  = np.reshape(np.array(geofence_lat), (-1,1))
        return geofence_lon, geofence_lat

    def compute_initial_waypoints(self):       
        # Write your code here
        N = 10
        total_nodes = N*len(self.lat_eNBs)
        all_lons = np.zeros((len(self.lat_eNBs), N))
        all_lats = np.zeros((len(self.lat_eNBs), N))
        lz_lat = 35.7271223
        lz_lon = -78.6962747
        geofence_lon, geofence_lat = self.readGeofence('./AERPAW_UAV_Geofence_Phase_1.kml')

        for i in range(len(self.lat_eNBs)):
            lonSamples, latSamples = self.sampleNeighborhood(self.lat_eNBs[i], self.lon_eNBs[i], self.r_eNBs[i], N, geofence_lat, geofence_lon)
            plt.scatter(lonSamples,latSamples, marker='x')
            all_lons[i,:] = lonSamples
            all_lats[i,:] = latSamples
        all_lons = np.squeeze(np.reshape(all_lons, (1,-1)))
        all_lats = np.squeeze(np.reshape(all_lats, (1,-1)))
        # all_lats = np.insert(all_lats, 0, lz_lat)
        # all_lons = np.insert(all_lons, 0, lz_lon)

        idxs  = list(combinations(np.arange(0,total_nodes), 2))

        self.N1_idcs = np.arange(start=0, stop = N)
        self.N2_idcs = self.N1_idcs + N
        self.N3_idcs = self.N2_idcs + N
        self.N4_idcs = self.N3_idcs + N

        #Remove edges within a neighborhood
        remove_idcs = []
        for i in range(len(idxs)):
            if np.any(self.N1_idcs == idxs[i][0]) and np.any(self.N1_idcs == idxs[i][1]):
                remove_idcs.append(i)
            elif np.any(self.N2_idcs == idxs[i][0]) and np.any(self.N2_idcs == idxs[i][1]):
                remove_idcs.append(i)
            elif np.any(self.N3_idcs == idxs[i][0]) and np.any(self.N3_idcs == idxs[i][1]):
                remove_idcs.append(i)
            elif np.any(self.N4_idcs == idxs[i][0]) and np.any(self.N4_idcs == idxs[i][1]):
                remove_idcs.append(i)
        idxs = np.delete(idxs, remove_idcs, 0)
        #Calculate all distances
        dist = self.haversine(all_lats[idxs[:,0]], all_lons[idxs[:,0]], all_lats[idxs[:,1]], all_lons[idxs[:,1]])
        n = len(dist)

        #Degree constraints: any selected nodes must have an incoming and outoing edge (degree 2)
        A_deg = np.zeros((total_nodes, n+total_nodes))
        for i in range(total_nodes):
            edgeMask = np.logical_or(idxs[:,0] == i, idxs[:,1] == i).astype(int)
            A_deg[i,0:n] = edgeMask
            A_deg[i,n+i] = -2
        b_deg = np.zeros((total_nodes, 1))

        #Group constraints: Only select on node out of any group
        A_group = np.zeros((len(self.lat_eNBs), n + total_nodes))
        A_group[0, n+self.N1_idcs] = 1
        A_group[1, n+self.N2_idcs] = 1
        A_group[2, n+self.N3_idcs] = 1
        A_group[3, n+self.N4_idcs] = 1
        b_group = np.ones((len(self.lat_eNBs), 1))

        #Enforce the starting and stopping point
        A_start = np.zeros((1,n + total_nodes))
        A_start[0,n] = 1
        b_start = np.ones((1,1))

        c = np.squeeze(np.concatenate((np.reshape(dist, (1,-1)), np.zeros((1,total_nodes))), axis=1))

        A_eq = np.concatenate((A_deg, A_group, A_start), axis=0)
        b_eq = np.concatenate((b_deg, b_group, b_start), axis=0)

        # print(c.shape)
        # print(A_eq.shape)
        # print(b_eq.shape)

        res = opt.linprog(c,A_eq=A_eq, b_eq=b_eq, bounds=(0,1), integrality=1)
        # tot_dist = res.x @ c
        # self.flying_time_at_max_v = tot_dist / 10

        x_edges = np.round(res.x[0:n])
        y_nodes = np.round(res.x[n::])
        selected_nodes = np.squeeze(np.where(y_nodes == 1))
        selected_edges = np.squeeze(idxs[np.where(x_edges == 1),:])

        edges_copy = np.copy(selected_edges)
        #Extract node ordering
        ordered_nodes = []
        ordered_nodes.append(selected_edges[0,0])
        ordered_nodes.append(selected_edges[0,1])
        edges_copy = np.delete(edges_copy, 0, axis=0)

        for i in range(2,5):
            idx = np.squeeze(np.where(edges_copy[:,0] == ordered_nodes[i-1]))
            if np.any(idx):
                ordered_nodes.append(edges_copy[idx,1])
                edges_copy = np.delete(edges_copy, idx, axis=0)
            else:
                idx = np.squeeze(np.where(edges_copy[:,1] == ordered_nodes[i-1]))
                ordered_nodes.append(edges_copy[idx,0])
                edges_copy = np.delete(edges_copy, idx, axis=0)

        idx = np.squeeze(np.where(np.logical_and(ordered_nodes >= self.N1_idcs[0], ordered_nodes <= self.N1_idcs[-1])))
        if len(idx) > 1:
            if ordered_nodes[1] >= self.N4_idcs[0] and ordered_nodes[1] <= self.N4_idcs[-1]:
                ordered_nodes = np.delete(ordered_nodes, -1)
            else:
                ordered_nodes = np.flip(ordered_nodes)
                ordered_nodes = np.delete(ordered_nodes, -1)
        else:
            if ordered_nodes[idx-1] >= self.N4_idcs[0] and ordered_nodes[idx-1] <= self.N4_idcs[-1]:
                ordered_nodes = np.flip(ordered_nodes)
            ordered_nodes = np.delete(ordered_nodes, -1)
            ordered_nodes = np.roll(ordered_nodes, -idx)

        lat_stops = all_lats[ordered_nodes]
        lon_stops = all_lons[ordered_nodes]

        for node in ordered_nodes:
            if np.isin(node, self.N1_idcs):
                self.nextBS.append(1)
            elif np.isin(node, self.N2_idcs):
                self.nextBS.append(2)
            elif np.isin(node, self.N3_idcs):
                self.nextBS.append(3)
                BS3_idx = len(self.nextBS) - 1
            elif np.isin(node, self.N4_idcs):
                self.nextBS.append(4)
            else:
                self.nextBS.append(0)

        self.nextBS = np.insert(self.nextBS, [BS3_idx, BS3_idx+1], -1)
        self.nextBS = np.append(self.nextBS, 0)

        idx = np.squeeze(np.where(np.logical_and(ordered_nodes >= self.N3_idcs[0], ordered_nodes <= self.N3_idcs[-1])))

        lat_stops = np.insert(lat_stops, [idx, idx+1], self.dummy_waypoint_lat)
        lon_stops = np.insert(lon_stops, [idx, idx+1], self.dummy_waypoint_lon)
        lat_stops = np.insert(lat_stops, 0, lz_lat)
        lon_stops = np.insert(lon_stops, 0, lz_lon)
        lat_stops = np.append(lat_stops, lz_lat)
        lon_stops = np.append(lon_stops, lz_lon)

        self.tot_dist = res.x @ c
        # for i in range(len(lat_stops)-1):
        #     self.tot_dist += self.haversine(lat_stops[i], lon_stops[i],lat_stops[i+1], lon_stops[i+1])
        self.flying_time_at_max_v = self.tot_dist / 10
        # If you do not use a plan file, there are a set of defualt waypoints. Please check at the end of this function to know how to use.
        # If you use a plan file from the QGroundControl and want to extract the waypoints. Use: extractWaypointsFromPlanFile function
        # This will return you the same format waypoints from your plan file used at the end of this function.
        # Example: How to extract waypoints from plan file
        #myWaypoints = self.extractWaypointsFromPlanFile('aadm.plan')
        # print(myWaypoints)
                      
        # Example. How to use waypoints in the experiment
        #self.waypoints = myWaypoints
        for i in range(len(lat_stops)):
            self.waypoints.append({"latitude": lat_stops[i], "longitude": lon_stops[i]})
            AERPAW_Platform.log_to_oeo(f"Waypoint: Latitude: {lat_stops[i]}, Longitude: {lon_stops[i]}, BS: {self.nextBS[i]}")
        AERPAW_Platform.log_to_oeo(f"Total distance: {self.tot_dist}")
                
        ########################################### Please don't modify this code ############################
        # Default waypoints if no waypoints are generated by the experimenter.                    
        if not self.waypoints:
            default_waypoints = [
                {"latitude": 35.72752574530495, "longitude": -78.69616739508318},
                {"latitude": 35.730250742963975, "longitude": -78.6981208320202},
                {"latitude": 35.72735286298385, "longitude": -78.70022275170658},
                {"latitude": 35.723510306455935, "longitude": -78.69314232782715},
                {"latitude": 35.72719710997374, "longitude": -78.69645491282358}
            ]
            self.waypoints = default_waypoints                    

        self.lastWaypointIndex = len(self.waypoints) - 1   
    
    def remove_waypoint(self, remove_BS):
        lz_lat = 35.7271223
        lz_lon = -78.6962747

        idcs = []
        BS_idx = np.squeeze(np.where(self.nextBS == remove_BS))
        idcs.append(BS_idx)
        if remove_BS == 2 and self.nextBS[BS_idx + 1] == -1:
            self.nextBS = np.insert(self.nextBS, BS_idx+1, 0)
            self.waypoints = np.insert(self.waypoints, BS_idx+1, {"latitude": lz_lat, "longitude": lz_lon})
            self.lastWaypointIndex += 1
        #Remove waypoints for BS3
        if remove_BS == 3:
            idcs.append(BS_idx-1)
            idcs.append(BS_idx+1)
            self.lastWaypointIndex -= 2
        
        self.nextBS = np.delete(self.nextBS, idcs)
        self.waypoints = np.delete(self.waypoints, idcs)

        dist=[]
        for i in range(len(self.waypoints)-1):
            dist.append(self.haversine(self.waypoints[i]["latitude"], self.waypoints[i]["longitude"], self.waypoints[i+1]["latitude"], self.waypoints[i+1]["longitude"]))
        self.tot_dist = np.sum(dist)
        self.lastWaypointIndex -= 1
        #AERPAW_Platform.log_to_oeo(f"Last index: {self.lastWaypointIndex}")

    
    def update_target_bs(self):
        
        # Example. How to extract data volume with corresponding BSs/LWs. 
        if self.cnt == 0:
            self.cnt = 1         # Print only once
            data_volume = self.checkDataVolumes()
            total_volume = 0
            for bs_id, volume in data_volume.items():
                #print(f"BS{bs_id}>Data Volume: {volume}") #please check _vehicle_log.txt in /root/Results directory
                #AERPAW_Platform.log_to_oeo(f"BS{bs_id}>Data Volume: {volume}")
                self.volumes[bs_id] = volume
                total_volume += volume
            if total_volume == 0:
                self.cnt = 0
            else:
                visit_BS = []
                for bs_id, volume in self.volumes.items():
                    if volume == 0:
                        self.remove_waypoint(int(bs_id))
                        self.BS_time[bs_id] = 0
                        AERPAW_Platform.log_to_oeo(f"BS{bs_id}> Removed")
                        continue
                    if volume < 70:
                        self.BS_time[bs_id] = 0
                        total_volume -= volume
                        continue
                    elif bs_id == "4" and volume / total_volume < 0.15:
                        self.remove_waypoint(4)
                        self.BS_time["4"] = 0
                        total_volume -= volume
                        AERPAW_Platform.log_to_oeo(f"BS{4}> Removed")
                        continue
                    visit_BS.append(bs_id)
                num_BS_visits = len(visit_BS)
                self.flying_time_at_max_v = self.tot_dist / 10
                AERPAW_Platform.log_to_oeo(f"Final Waypoints: {self.waypoints}")

                time_diff = 0
                for bs in visit_BS:
                    self.BS_time[bs] = (self.volumes[bs] / total_volume) * (500 - self.flying_time_at_max_v - 30) #subtract 30 for landing
                    if self.volumes[bs] < 100 and self.BS_time[bs] > 20:
                        time_diff += self.BS_time[bs] - 20
                        self.BS_time[bs] = 20

                    # Redistribute some of the time for 1
                    if np.isin('1', visit_BS):
                        if bs == "1":
                            self.BS_time[bs] -= 10
                        else:
                            self.BS_time[bs] += 10 / num_BS_visits

                # Redistribute time diff to BS above threshold
                BS_above_thresh_counter = 0
                for bs_id, volume in self.volumes.items():
                    if volume > 100:
                        BS_above_thresh_counter += 1
                
                for bs_id, volume in self.volumes.items():
                    if volume > 100:
                        self.BS_time[bs_id] += time_diff / BS_above_thresh_counter

                for bs_id, time in self.BS_time.items():
                    AERPAW_Platform.log_to_oeo(f"BS{bs_id}>Alotted Time: {time}")

        #else:
        #    pass
        
        # Download
        download_data = self.checkDownload()
        for bs_id, total_received_data in download_data.items():
            print(f"BS{bs_id}>Total received data: {total_received_data}")
        
        if self.curr_BS == 0 or self.curr_BS == -1:
            # Example. How to extract SNRs with corresponding LWs. 
            SNRs = self.checkSNR()
            for bs_id, snr in SNRs.items():
                print(f"BS{bs_id}>SNR: {snr}") #please check _vehicle_log.txt in /root/Results directory
            
            #SNRs = self.checkSignalStrengths()
            sorted_snr = sorted(SNRs.items(), key=lambda item: item[1], reverse=True)        
            
            i=0
            while i < 4:
                max_bs_id, max_snr = sorted_snr[i]
                #Example. How to set the target base station to start downloading data
                if download_data[str(max_bs_id)] >= self.volumes[str(max_bs_id)]:
                    i+=1
                else:
                    break
            self.update_bs_id = max_bs_id
        else:
            self.update_bs_id = str(self.curr_BS)
        
        print(f'BS{self.update_bs_id} has been applied.')
    
    def update_uav_waypoints(self):
        # For autonmous trajectory, you should create a waypoint and update the waypoint in self.waypoints
        # You might change or remove the condition: self.nextWaypointIndex == self.lastWaypointIndex in go_forward state if you need.

        # Example: How to update the next waypoint. 
        self.nextWaypointIndex += 1                
        
        #Example. How to change the UAV orientation in a waypoint
        self.angles = np.arange(0, 360, 45) 
        
        # set wait time
        # Example: How to set a waitTime at a waypoint. 
        try:
            #AERPAW_Platform.log_to_oeo(f"times2: {help(list(self.BS_time.keys())[0])}")
            #AERPAW_Platform.log_to_oeo(f"currBS: {self.curr_BS}")
            #AERPAW_Platform.log_to_oeo(f"Waypoint {self.nextWaypointIndex} Time {self.BS_time[str(self.curr_BS)]}")
            self.waitTime = self.BS_time[str(self.curr_BS)]
        except Exception as e:
            #AERPAW_Platform.log_to_oeo(f"err: {e}")
            AERPAW_Platform.log_to_oeo(f"Default wait time")
            #AERPAW_Platform.log_to_oeo(f"{self.BS_time}")
            self.waitTime = 5

        
        
    async def run_update_target_bs(self):
        """Run update_target_bs() every second periodically."""
        while True:        
            if self.updateBS == True:
                # Update base station
                self.update_target_bs()  
                
                # Send Selected BS to all BSs and 
                # Download Data from Target BS-i
                if not self.update_bs_id:
                    self.setBSForBroadcast("1") # sends only to LW1 if no base station is selected
                else:
                    self.setBSForBroadcast(self.update_bs_id) # experimenter changes bs to download data from 

                # Update UAV Waypoints
                if self.updateWaypoint == True:
                    self.update_uav_waypoints()
                    AERPAW_Platform.log_to_oeo(f"Next Waypoint {self.nextWaypointIndex}")
                    self.updateWaypoint = False

                await asyncio.sleep(1)

            else:
                break

    async def timer(self, duration):
        AERPAW_Platform.log_to_oeo(f"Timer started for: {self.curr_BS}, {duration} sec")
        await asyncio.sleep(duration)
        self.time_flag = True

    async def download_checker(self):
        AERPAW_Platform.log_to_oeo(f"Download Tracker started for: {self.curr_BS}")
        download_data = self.checkDownload()
        curr_download = download_data[str(self.curr_BS)]
        while curr_download < self.volumes[str(self.curr_BS)]:
            AERPAW_Platform.log_to_oeo(f"Download: {curr_download} Volume: {self.volumes[str(self.curr_BS)]}")
            await asyncio.sleep(1)
            download_data = self.checkDownload()
            curr_download = download_data[str(self.curr_BS)]
        AERPAW_Platform.log_to_oeo(f"Setting Download Flag: {curr_download}, {self.volumes[str(self.curr_BS)]}")
        self.download_flag = True
    
    @state(name="start", first=True)
    async def start(self, vehicle: Drone):
        #self.init_args()
        #reset
        self.updateWaypoint = False
        self.setBSForBroadcast("")
        #self.start_download('0')
        
        # record the start time of the search
        self.start_time = datetime.datetime.now()

        #print("Taking off")
        AERPAW_Platform.log_to_oeo(f"Calculating trajectory")
        self.compute_initial_waypoints()                
        AERPAW_Platform.log_to_oeo(f"Finished calculating trajectory")
        ## Start the independent task for updating base stations every second
        asyncio.ensure_future(self.run_update_target_bs())
        
        await vehicle.takeoff(25) # fixed 25m
        #print("Took off")    
        AERPAW_Platform.log_to_oeo(f"Taking off to {25}m")
        

        #print("Start downloading data at altitude 25 m")    
        #self.start_download('1') # Starts downloading data from a base station
        #await asyncio.sleep(1)
        #asyncio.ensure_future(self.run_update_target_bs())
        
        return "go_forward"

    @state(name="go_forward")
    async def go_forward(self, vehicle: Vehicle):
        # home_coords = Coordinate(
            # vehicle.home_coords.lat, vehicle.home_coords.lon, vehicle.position.alt
        # )
        # print(home_coords)
    
        await vehicle.set_groundspeed(self.target_speed)
        
        cur_pos = vehicle.position
        
        #print("drone",cur_pos)
        #print(type(cur_pos))
        #waypoint = self.waypoints[1]
        waypoint = self.waypoints[self.nextWaypointIndex]
        self.curr_BS = self.nextBS[self.nextWaypointIndex]
        
        
        latitude = waypoint['latitude']
        longitude = waypoint['longitude']
        AERPAW_Platform.log_to_oeo(f"En route to Waypoint: {self.nextWaypointIndex} Latitude: {latitude}, Longitude: {longitude}, BS: {self.curr_BS}")
        #altitude = waypoint['altitude']
        altitude = self.uav_altitude
        #print(f"Waypoint: Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude} m")
        next_pos = Coordinate(lat=latitude, lon=longitude, alt=altitude)
        
        #print(cur_pos.bearing(next_pos))
        #vec = next_pos - cur_pos
        #print(vec)
        
        # take a turn towards to next waypoint
        new_heading = cur_pos.bearing(next_pos)
        turning = asyncio.ensure_future(vehicle.set_heading(new_heading))

        # wait for vehicle to finish turning
        while not turning.done():
            await asyncio.sleep(0.2)
        await turning        
        
        
        (valid_waypoint, msg) = self.safety_checker.validateWaypointCommand(
            cur_pos, next_pos
        )
        
        # if the next location violates the geofence, return home
        if not valid_waypoint:        
            print("Can't go there:")
            return "return_to_launch_and_land"
        self.cancel_flight_flag = False
        if self.curr_BS > 0:
            download_data = self.checkDownload()
            if download_data[str(self.curr_BS)] >= self.volumes[str(self.curr_BS)]:
                self.cancel_flight_flag = True
        elif self.curr_BS == -1 and self.nextBS[self.nextWaypointIndex+1] == 3:
            download_data = self.checkDownload()
            if download_data["3"] >= self.volumes["3"]:
                self.cancel_flight_flag = True
                self.nextWaypointIndex += 2

        if not self.cancel_flight_flag:
            curr_coord = Coordinate(vehicle.position.lat, vehicle.position.lon, vehicle.position.alt)
            full_vector = next_pos - curr_coord
            move_vector = 0.5 * full_vector

            # otherwise move forward to the next location
            #print("UAV goes towards the target base station")
            moving = asyncio.ensure_future(
                vehicle.goto_coordinates(vehicle.position + move_vector) #, target_heading=self._default_heading)
            )

            while not moving.done(): # wait until the vehicle is done moving
                await asyncio.sleep(0.2)

            await moving

            # We have traveled halfway, check if we finished downloading while en route
            if self.curr_BS > 0:
                download_data = self.checkDownload()
                if download_data[str(self.curr_BS)] >= self.volumes[str(self.curr_BS)]:
                    self.cancel_flight_flag = True
            elif self.curr_BS == -1 and self.nextBS[self.nextWaypointIndex+1] == 3:
                download_data = self.checkDownload()
                if download_data["3"] >= self.volumes["3"]:
                    self.cancel_flight_flag = True
                    self.nextWaypointIndex += 2

            if self.cancel_flight_flag:
                AERPAW_Platform.log_to_oeo(f"Finished download en route to: {self.nextWaypointIndex}, BS: {self.curr_BS}")
            else:
                moving = asyncio.ensure_future(
                    vehicle.goto_coordinates(next_pos) #, target_heading=self._default_heading)
                )
                while not moving.done(): # wait until the vehicle is done moving
                    await asyncio.sleep(0.2)

                await moving
                AERPAW_Platform.log_to_oeo(f"Arrived at Waypoint: {self.nextWaypointIndex}, BS: {self.curr_BS}")
        else:
            AERPAW_Platform.log_to_oeo(f"Already finished download for: {self.nextWaypointIndex}, BS: {self.curr_BS}")
        # Try moving in 1 meter increments along the way to waypoint
        # self.cancel_flight_flag = False
        # curr_coord = Coordinate(vehicle.position.lat, vehicle.position.lon, vehicle.position.alt)
        # distance = curr_coord.distance(next_pos)
        # full_vector = next_pos - curr_coord
        # norm_vector = full_vector.norm()
        # arrived = False
        # while not arrived:
        #     curr_coord = Coordinate(vehicle.position.lat, vehicle.position.lon, vehicle.position.alt)
        #     distance = curr_coord.distance(next_pos)
        #     if distance < 0.1:
        #         arrived = True
        #         vector = 0*norm_vector
        #     elif distance > 10:
        #         vector = 10*norm_vector
        #     elif distance > 5:
        #         vector = 5*norm_vector
        #     else:
        #         vector = 3*norm_vector

        #     await vehicle.set_velocity(vector, duration=1)

        #     if self.curr_BS > 0:
        #         download_data = self.checkDownload()
        #         if download_data[str(self.curr_BS)] >= self.volumes[str(self.curr_BS)]:
        #             self.cancel_flight_flag = True
        #             AERPAW_Platform.log_to_oeo(f"Finished download en route to: {self.nextWaypointIndex}, BS: {self.curr_BS}")
        #             break
            
        # if arrived:
        #     AERPAW_Platform.log_to_oeo(f"Arrived at Waypoint: {self.nextWaypointIndex}, BS: {self.curr_BS}")
        
        self.updateWaypoint = True
        
        #print('self.nextWaypointIndex: ',self.nextWaypointIndex, 'self.lastWaypointIndex: ',self.lastWaypointIndex)
        
        #is_download_complete = self.getToken()  
                
        self.t1 = datetime.datetime.now() - self.start_time
        self.t1 = self.t1.total_seconds()
        #print(self.t1)
        #print(self.flight_time)
        
        download_complete = str(self.getToken())
        #print('download_complete', download_complete)
        if self.nextWaypointIndex == self.lastWaypointIndex or download_complete == '-1' or self.t1 >= self.flight_time:
            AERPAW_Platform.log_to_oeo(f"Token Detected")              
            return "return_to_launch_and_land"  
        #Update waypoints     
        #self.update_uav_waypoints()
        
        # print("Wait 10 seconds for data download at the current waypoint before going to the next waypoint.")
        
        #self.wait_for_download()

        return "wait_for_download"        
                

    @state(name="wait_for_download")
    async def wait_for_download(self, vehicle: Drone):
        #cur_pos = vehicle.position
        AERPAW_Platform.log_to_oeo(f"Awaiting BS: {self.curr_BS}")
        await asyncio.sleep(2) #wait four seconds
        if self.curr_BS == -1 or self.curr_BS == 0 or self.cancel_flight_flag:
            return "go_forward"
        
        #await asyncio.sleep(self.waitTime)  # Non-blocking wait for self.waitTime
        #Start timer and download checker to exit out of alignment early if needed
        timer = asyncio.ensure_future(self.timer(self.BS_time[str(self.curr_BS)]))
        download_detector = asyncio.ensure_future(self.download_checker())
        altitude = self.uav_altitude
        cur_pos = vehicle.position
        latitude = self.lat_eNBs[self.curr_BS - 1]
        longitude = self.lon_eNBs[self.curr_BS - 1]
        BS_pos = Coordinate(lat=latitude, lon=longitude, alt=altitude)
        
        BS_head = cur_pos.bearing(BS_pos)
        turning = asyncio.ensure_future(vehicle.set_heading(BS_head))

        # wait for vehicle to finish turning
        while not turning.done():
            await asyncio.sleep(0.2)
        await turning

        if self.time_flag:
            AERPAW_Platform.log_to_oeo(f"Time limit reached for BS: {self.curr_BS}")
            self.time_flag = False
            self.download_flag = False
            download_detector.cancel()
            return "go_forward"
        elif self.download_flag:
            AERPAW_Platform.log_to_oeo(f"Download complete for BS: {self.curr_BS}")
            self.download_flag = False
            self.time_flag = False # Reset both flag just in case
            timer.cancel()
            return "go_forward"

        curr_snrs = []
        for angle in self.angles:
            # Initialize new_heading with the vehicle's current heading
            new_heading = BS_head + angle  # Use 'angle' from the loop
            
            turning = asyncio.ensure_future(vehicle.set_heading(new_heading))  # Correct vehicle reference

            # Wait for the vehicle to finish turning
            while not turning.done():
                await asyncio.sleep(0.1) #4 seconds at each rotation
            
            if self.time_flag:
                AERPAW_Platform.log_to_oeo(f"Time limit reached for BS: {self.curr_BS}")
                self.time_flag = False
                self.download_flag = False
                download_detector.cancel()
                return "go_forward"
            elif self.download_flag:
                AERPAW_Platform.log_to_oeo(f"Download complete for BS: {self.curr_BS}")
                self.time_flag = False
                self.download_flag = False
                timer.cancel()
                return "go_forward"
            
            SNRs = self.checkSNR()
            curr_snrs.append(SNRs[str(self.curr_BS)])
            AERPAW_Platform.log_to_oeo(f"SNR: {SNRs[str(self.curr_BS)]} Heading {new_heading}")
        peak_head_idx = np.argmax(curr_snrs)
        peak_head = BS_head + self.angles[peak_head_idx]

        AERPAW_Platform.log_to_oeo(f"Peak SNR: {curr_snrs[peak_head_idx]} Heading {peak_head}")
        turning = asyncio.ensure_future(vehicle.set_heading(peak_head))  # Correct vehicle reference
        # Wait for the vehicle to finish turning
        while not turning.done():
            await asyncio.sleep(0.1)

        while not self.download_flag:
            await asyncio.sleep(1)
            if self.time_flag:
                AERPAW_Platform.log_to_oeo(f"Time limit reached for BS: {self.curr_BS}")
                self.time_flag = False
                self.download_flag = False
                download_detector.cancel()
                break
        
        if not self.download_flag:
            download_detector.cancel()
        else:
            AERPAW_Platform.log_to_oeo(f"Download complete for BS: {self.curr_BS}")
            self.time_flag = False
            self.download_flag = False
            timer.cancel()

        return "go_forward"  # Assuming you want to return this after completion
          
          
    @state(name="return_to_launch_and_land")
    async def return_to_launch_and_land(self, vehicle: Drone):
        lz_lat = 35.7271223
        lz_lon = -78.6962747
        cur_pos = vehicle.position
        AERPAW_Platform.log_to_oeo('Return to launch and landing.')

        if cur_pos.lon > self.dummy_waypoint_lon:
            AERPAW_Platform.log_to_oeo('Using dummy waypoint')
            dummy_waypoint = Coordinate(self.dummy_waypoint_lat, self.dummy_waypoint_lon, vehicle.position.alt)
            new_heading = cur_pos.bearing(dummy_waypoint)
            turning = asyncio.ensure_future(vehicle.set_heading(new_heading))
            # wait for vehicle to finish turning
            while not turning.done():
                await asyncio.sleep(0.2)

            await turning        
                    
            await vehicle.goto_coordinates(
                dummy_waypoint #, target_heading=self._default_heading
            )

        if cur_pos.lon > lz_lon:
            AERPAW_Platform.log_to_oeo('Using dummy lz')
            dummy_waypoint = Coordinate(lz_lat, lz_lon, vehicle.position.alt)
            new_heading = cur_pos.bearing(dummy_waypoint)
            turning = asyncio.ensure_future(vehicle.set_heading(new_heading))
            # wait for vehicle to finish turning
            while not turning.done():
                await asyncio.sleep(0.2)

            await turning        
                    
            await vehicle.goto_coordinates(
                dummy_waypoint #, target_heading=self._default_heading
            )
        
        home_coords = Coordinate(
            vehicle.home_coords.lat, vehicle.home_coords.lon, vehicle.position.alt
        )
        
         
        
        new_heading = cur_pos.bearing(home_coords)
        turning = asyncio.ensure_future(vehicle.set_heading(new_heading))

        # wait for vehicle to finish turning
        while not turning.done():
            await asyncio.sleep(0.2)

        await turning        
                
        await vehicle.goto_coordinates(
            home_coords #, target_heading=self._default_heading
        )

        await vehicle.land()
        
        
        #After landing stop downloading data
        self.updateBS = False
        #self.start_download('0') #after completing data collection, reset the flag
        self.setBSForBroadcast("99") # reset             
        
        #print("Done!")   
        stop_time = time.time()
        #print(stop_time)
        seconds_to_complete = datetime.datetime.now() - self.start_time
        #print(seconds_to_complete)        
        #AERPAW_Platform.log_to_oeo(f"mission took {seconds_to_complete} ")
        AERPAW_Platform.log_to_oeo("Done!")
        #print('Done!')
                
