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
    waitTime = 0
    uav_altitude = 25
    angles = []
    cnt = 0
    updateBS = True
    updateWaypoint = True
    t1 = None
    max_speed = 10 #mps
    target_speed = 10
    
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

    def compute_initial_waypoints(self):       
        # Write your code here

        # If you do not use a plan file, there are a set of defualt waypoints. Please check at the end of this function to know how to use.
        # If you use a plan file from the QGroundControl and want to extract the waypoints. Use: extractWaypointsFromPlanFile function
        # This will return you the same format waypoints from your plan file used at the end of this function.
        # Example: How to extract waypoints from plan file
        myWaypoints = self.extractWaypointsFromPlanFile('aadm.plan')
        # print(myWaypoints)
                      
        # Example. How to use waypoints in the experiment
        self.waypoints = myWaypoints
        
                
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
    
    
    def update_target_bs(self):
        
        # Example. How to extract data volume with corresponding BSs/LWs. 
        if self.cnt == 0:
            self.cnt = 1         # Print only once
            data_volume = self.checkDataVolumes()
            for bs_id, volume in data_volume.items():
                print(f"BS{bs_id}>Data Volume: {volume}") #please check _vehicle_log.txt in /root/Results directory
        #else:
        #    pass
        
        # Download
        download_data = self.checkDownload()        
        for bs_id, total_received_data in download_data.items():
            print(f"BS{bs_id}>Total received data: {total_received_data}")
                
        # Example. How to extract SNRs with corresponding LWs. 
        SNRs = self.checkSNR()
        for bs_id, snr in SNRs.items():
            print(f"BS{bs_id}>SNR: {snr}") #please check _vehicle_log.txt in /root/Results directory                            
        
        
                                                   

        
        #SNRs = self.checkSignalStrengths()
        sorted_snr = sorted(SNRs.items(), key=lambda item: item[1], reverse=True)        
        
        max_bs_id, max_snr = sorted_snr[0]
        #Example. How to set the target base station to start downloading data
        self.update_bs_id = max_bs_id
        print(f'BS{self.update_bs_id} has been applied.')
    
    def update_uav_waypoints(self):
        # For autonmous trajectory, you should create a waypoint and update the waypoint in self.waypoints
        # You might change or remove the condition: self.nextWaypointIndex == self.lastWaypointIndex in go_forward state if you need.

        # Example: How to update the next waypoint. 
        self.nextWaypointIndex += 1                
        
        #Example. How to change the UAV orientation in a waypoint
        self.angles = [90, 90, 90, 90] 
        
        # set wait time
        # Example: How to set a waitTime at a waypoint. 
        self.waitTime = 5 #10 seconds     

        
        
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
                    AERPAW_Platform.log_to_oeo(f"Waypoint {self.nextWaypointIndex}")
                    self.updateWaypoint = False

                await asyncio.sleep(1)
                                                  
                    
            else:
                break
    
    
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
                
        self.compute_initial_waypoints()                
        
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
        
        
        latitude = waypoint['latitude']
        longitude = waypoint['longitude']
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

        
        # otherwise move forward to the next location
        #print("UAV goes towards the target base station")
        moving = asyncio.ensure_future(
            vehicle.goto_coordinates(next_pos) #, target_heading=self._default_heading)
        )
        
        while not moving.done(): # wait until the vehicle is done moving
            #self.update_target_bs()
            await asyncio.sleep(0.2)

        await moving
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
            return "return_to_launch_and_land"  
        #Update waypoints     
        #self.update_uav_waypoints()
        
        # print("Wait 10 seconds for data download at the current waypoint before going to the next waypoint.")
        
        #self.wait_for_download()
        
        return "wait_for_download"        
                

    @state(name="wait_for_download")
    async def wait_for_download(self, vehicle: Drone):
        #cur_pos = vehicle.position
       
        await asyncio.sleep(4) #wait four seconds
        for angle in self.angles:
            
            # Initialize new_heading with the vehicle's current heading
            heading = vehicle.heading        
            new_heading = heading + angle  # Use 'angle' from the loop
            
            turning = asyncio.ensure_future(vehicle.set_heading(new_heading))  # Correct vehicle reference

            # Wait for the vehicle to finish turning
            while not turning.done():
                #Added by Kevin to test
                #attitude = vehicle.attitude
                #print(attitude)
                await asyncio.sleep(0.1) #4 seconds at each rotation
            
            await turning

        await asyncio.sleep(self.waitTime)  # Non-blocking wait for self.waitTime  

        return "go_forward"  # Assuming you want to return this after completion
          
          
    @state(name="return_to_launch_and_land")
    async def return_to_launch_and_land(self, vehicle: Drone):
        cur_pos = vehicle.position                
        print('Return to launch and landing.')

        
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
                
