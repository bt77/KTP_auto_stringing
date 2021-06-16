import config, conductor
from span import Span
from wire import Wire

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error 
    

def extract_spans():
    # input vars
    in_Poles_shp = os.path.join(config.INPUT, config.STRUCTURE_TOP)
    lines_joined = os.path.join(config.OUTPUT, config.LINES_JOINED) 
    point_buffer = os.path.join(config.OUTPUT, config.POINT_BUFFER)
    out_centreline_cleaned = os.path.join(config.OUTPUT, config.CENTRELINE)
    out_centreline_cleaned_buf = os.path.join(config.OUTPUT, config.CENTRELINE_BUFFER_SHAPEFILE)
    out_spans = os.path.join(config.OUTPUT, config.SPAN)


    
    # Create folders
    os.makedirs(config.OUTPUT, exist_ok=True)
    
    # Extract spans of conductor points 
    print("Extracting spans ...")
    with Span(lastool_dir=config.LASTOOLS, in_dir=config.INPUT, tops_dist=config.TOPS_DIST, span_length=config.SPAN_LENGTH, centreline_buffer=config.CENTRELINE_BUFFER) as sp:                                  
        # Create Structure top shapefile
        sp.extract_structure_tops_shp(os.path.join(config.INPUT, config.CLASSIFIED_POINT_CLOUD), config.STRUCTURE, config.CONDUCTOR)
    
        # Create span shapefile
        sp.create_span_shp(in_Poles_shp, lines_joined, point_buffer, out_centreline_cleaned, out_centreline_cleaned_buf, config.BUFFER_DISTANCE)
    
        # Split Conductor by span
        sp.split_las(out_centreline_cleaned_buf, os.path.join(config.INPUT, 'Conductor.laz'), out_spans)
    
        # Select valid spans of Conductor
        sp.select_span(out_spans, config.SPAN_POINTS)
    print("Extracted spans.")
    
    
def extract_wires():    #TODO - loop through all spans in out_spans_cleaned, need to handle error when RANSCAN can't find the best fitted model
    # input vars
    out_wires = os.path.join(config.OUTPUT, config.WIRE)
    
    # Extract wires of conductor points
    print("Extracting wires ...")
    
    for span_laz in os.listdir(config.TEST_SPAN):
        with Wire(span_laz, config.TEST_SPAN, out_wires) as w:
            w.extract_wires()
    print("Extractd wires.")
            

# function to display the catenary parameters
def display_params(c):
    a_s = [c.a_gen, c.a0, c.a]
    xmin_s = [c.min_gen[0, 0], c.min0[0, 0], c.min_pred[0, 0]]
    ymin_s = [c.min_gen[0, 1], c.min0[0, 1], c.min_pred[0, 1]]
    zmin_s = [c.min_gen[0, 2], c.min0[0, 2], c.min_pred[0, 2]]
    if c.theta_gen is None:
        tg = None
    else:
        tg = np.rad2deg(c.theta_gen)
    theta_s = [tg, np.rad2deg(c.theta), np.rad2deg(c.theta)]
    length_s = [None, None, c.length]
    params = np.array(
        [a_s, xmin_s, ymin_s, zmin_s, theta_s, length_s]).T

    df = pd.DataFrame(
        params,
        columns=['a', 'x_min', 'y_min', 'z_min', 'theta', 'length']
    ).T
    df.columns = ['Generated', 'Initial', 'Predicted']

    pd.set_option("display.float_format", '{:0.3f}'.format)
    print(df)
    
def fit_catenary():     # TODO - save outputs (shapefile of vectors representing all catenaries in an area)
    print('Fitting catenary...')
    
    for wire in os.listdir(os.path.join(config.OUTPUT, config.WIRE)):
        c = conductor.Conductor()
        data_obs = np.loadtxt(os.path.join(config.OUTPUT, config.WIRE, wire))
        print(f'{wire}: {data_obs.shape[0]} points.')
        
        try:
            c.fit(data_obs)
            data_pred = c.predict(data_obs)
            rmse = np.sqrt(mean_squared_error(data_obs, data_pred))
            mve = np.mean(data_obs[:, 2] - data_pred[:, 2])
    
            message = 'Catenary: {}, \nRMSE: {:0.3f}, \nMVE: {:0.3f}, \nAdjustment success: {}, {}'
            message = message.format(c.catenary, rmse, mve, c.success, c.message)
            print(message)
            display_params(c)
        except conductor.InputError: 
            print('Failed to model conductor from {} points'.format(len(data_obs)))
    print('Fitted catenary for all wires.')

if __name__ == "__main__":
    extract_spans()
    extract_wires()
    fit_catenary()
