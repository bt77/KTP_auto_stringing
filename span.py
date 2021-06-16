from osgeo import ogr
from scipy import spatial
import numpy as np
import os, argparse, subprocess, pdal, json, shutil, laspy

from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd


class Span:
    """
    A span containing one or multiple wires
    Derived from Antonio Molina's scripts: https://drive.google.com/drive/u/0/folders/162fFj2fyrtdXxiBan8HlanqsoVb9n2qb
    """
    
    def __enter__(self):
        return self
        
    def __init__(self, lastool_dir, in_dir, tops_dist, span_length, centreline_buffer):
        self.lastool_dir=lastool_dir
        self.in_dir=in_dir
        self.tops_dist=tops_dist
        self.span_length=span_length
        self.centreline_buffer=centreline_buffer

    # # will create span shapefile using Structure top shapefile
    def create_span_shp(self, in_Poles_shp, lines_joined, point_buffer, out_centreline_cleaned, out_centreline_cleaned_buf, bufferDist):
        print('Creating Span shapefile ...')
        outDriver = ogr.GetDriverByName("ESRI Shapefile")
        ds = outDriver.CreateDataSource(lines_joined)
        out_lyr_lines = ds.CreateLayer('1', None, ogr.wkbUnknown)
        
        # # Create shapefile LINECLEANED. Get the driver to manage the shapefile
        outDriver = ogr.GetDriverByName("ESRI Shapefile")
        dslinecleaned = outDriver.CreateDataSource(out_centreline_cleaned)
        out_lyr_line_cleaned = dslinecleaned.CreateLayer('2', None, ogr.wkbUnknown)
        
        self.create_lines_from_points(in_Poles_shp, out_lyr_lines)
        ds = None
        
        buffer_obj = self.createBuffer(in_Poles_shp, point_buffer, bufferDist)
    
        # # Create_CL_buffered from the cleaned CL (without duplicates)
        outDriver = ogr.GetDriverByName("ESRI Shapefile")
        dslinecleanedbuf = outDriver.CreateDataSource(out_centreline_cleaned_buf)
        out_lyr_line_cleaned_buf = dslinecleanedbuf.CreateLayer('3', None, ogr.wkbUnknown)
        
        self.intersection(lines_joined, buffer_obj, out_lyr_line_cleaned, out_lyr_line_cleaned_buf)
    
        print(f'Span shapefile saved in {out_centreline_cleaned_buf}')


    # # will select conductor spans w/ > specified #points
    def select_span(self, in_dir, span_points):
        print('Selecting spans...')
        os.makedirs(in_dir + '_cleaned', exist_ok=True)
        for pc in os.listdir(in_dir):
            inFile = laspy.file.File(os.path.join(in_dir, pc), mode = "r")
            count = len(inFile)
            inFile.close()
            if count > span_points:
                shutil.move(os.path.join(in_dir, pc), os.path.join(in_dir + '_cleaned', pc))
        print('Selected spans.')


    # # will create SHP of estimated structure tops from 30-Classfied Tiled Point Cloud.
    def extract_structure_tops_shp(self, classified_pc_dir, structure_code, conductor_code):
        print('Creating Structure tops shapefile...')
        # # extract Structure points
        return_obj = subprocess.run([os.path.join(self.lastool_dir, 'las2las.exe'), 
            '-i', os.path.join(classified_pc_dir, '*.laz'), 
            '-merged', '-keep_class', structure_code,
            '-o', 'Structure.laz', '-odir', self.in_dir],
            capture_output=True)
        if return_obj.returncode != 0:
            error_message = f'A non 0 code was returned from las2las for {structure_code}-Structure'
            raise RuntimeError(error_message)
    
        # # extract Conductor points
        return_obj = subprocess.run([os.path.join(self.lastool_dir, 'las2las.exe'), 
            '-i', os.path.join(classified_pc_dir, '*.laz'), 
            '-merged', '-keep_class', conductor_code,
            '-o', 'Conductor.laz', '-odir', self.in_dir],
            capture_output=True)
        if return_obj.returncode != 0:
            error_message = f'A non 0 code was returned from las2las for {conductor_code}-Conductor'
            raise RuntimeError(error_message)
    
        # # create Structure polygons
        return_obj = subprocess.run([os.path.join(self.lastool_dir, 'lasboundary.exe'), 
            '-i', os.path.join(self.in_dir, 'Structure.laz'), 
            '-disjoint', '-concavity', '3',
            '-o', 'Structure.shp', '-odir', self.in_dir],
            capture_output=True)
        if return_obj.returncode != 0:
            error_message = 'A non 0 code was returned from lasboundary'
            raise RuntimeError(error_message)
    
        # # split Structure 
        self.split_las(os.path.join(self.in_dir, 'Structure.shp'), os.path.join(self.in_dir, 'Structure.laz'), os.path.join(self.in_dir, 'Structures'))
    
        # # extract Structure tops
        structures = os.path.join(self.in_dir, 'Structures', '*.laz')
        get_top={
            "pipeline": [
                structures,
                {
                    "type":"filters.locate",
                    "dimension":"Z",
                    "minmax":"max"
                },
                {
                    "type": "filters.merge"
                },
                os.path.join(self.in_dir, 'Tops.laz')
            ]
        }
        
        r = pdal.Pipeline(json.dumps(get_top))
        r.validate()
        r.execute()        
        
        # # convert to shapefile
        return_obj = subprocess.run([os.path.join(self.lastool_dir, 'las2shp.exe'), 
            '-i', os.path.join(self.in_dir, 'Tops.laz'),  
            '-o', 'Tops.shp', '-odir', self.in_dir, '-single_points'],
            capture_output=True)
        if return_obj.returncode != 0:
            error_message = 'A non 0 code was returned from las2shp'
            raise RuntimeError(error_message)
    
        print(f"Structure top shapefile saved in {os.path.join(self.in_dir, 'Top.shp')}")

    # # will create lines from points generated in the HM. Input will be a Point SHP
    def create_lines_from_points(self, in_Poles_shp, out_lyr_lines):
        print('Creating lines from points...')
        # # Open shapefile
        file = ogr.Open(in_Poles_shp, 0)
        in_shape = file.GetLayer(0)
        #print('in_shape: ', in_shape)

        # # initialise empty list
        geom_list = []

        # # loop through shapefile and create list of [x, y] list
        for i, point in enumerate(in_shape):
            #print('point: ', point)
            geom = point.GetGeometryRef()
            point_got = geom.GetPoint(0)
            #print('point_got: ', point_got)
            geom_list.append([point_got[0], point_got[1]])

        # # turn list of list [x, y] into numpy array
        geom_list_array = np.asarray(geom_list)
        #print('geom_list_array: ', geom_list_array)

        # # create kd tree of points
        tree = spatial.KDTree(geom_list_array)

        # # reset layer to be able to read again
        in_shape.ResetReading()

        #
        length_list = []
        #

        # # Loop to each point within the shp
        for i, point in enumerate(in_shape):
            geom = point.GetGeometryRef()
            point_got = geom.GetPoint(0)
            list = tree.query_ball_point([point_got[0], point_got[1]], self.tops_dist)

        # # Create lines joining points
        for item in list:
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(point_got[0], point_got[1])
            line.AddPoint(geom_list[item][0], geom_list[item][1])

            # Get rid of duplicate geometries
            length = line.Length()
            if length in length_list:
                continue
            length_list.append(length)


            # Draw lines bigger than specified meters

            if length > self.span_length:
                self.draw_centrelines(line, out_lyr_lines)

    # # Define buffer function.
    def createBuffer(self, in_Poles_shp, point_buffer, bufferDist):
        print('Creating buffer...')
        # # Open input layer
        inputds = ogr.Open(in_Poles_shp)
        inputlyr = inputds.GetLayer()
        # # Define output layer
        shpdriver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(point_buffer): #check if a file is there before doing anything with it
            shpdriver.DeleteDataSource(point_buffer) # if output already there, then delete it.
        outputBufferds = shpdriver.CreateDataSource(point_buffer) #It will create a new datasource based on the passed driver (ESRI SHP)
        bufferlyr = outputBufferds.CreateLayer(point_buffer, geom_type=ogr.wkbPolygon)
        featureDefn = bufferlyr.GetLayerDefn() #Fetch the shema information for this layer. The returned OGRFeatureDfn is owned by the OGRLayer, and should not be modified or freed by the app. It encapsulates the atrribute schema of the features of the layer.
        # # Loop into the inputlayer (Point shp)
        for feature in inputlyr:
            ingeom = feature.GetGeometryRef() #Fetch pointer to feature geometry.
            geomBuffer = ingeom.Buffer(bufferDist) #Buffer input layer to bufferDist

            outFeature = ogr.Feature(featureDefn) #Fetch a feature by its idetifies. This function will attempt to read the identified feature.
            outFeature.SetGeometry(geomBuffer)
            bufferlyr.CreateFeature(outFeature)
            outFeature = None
        return outputBufferds


    # # Intersecting polygon buffered with out_lyr_lines. inouts needed are shplines. Then loop into shplines, fetch pointer to feature geometry and Intersects with buf layer
    def intersection(self, lines_joined, point_buffer, out_centreline_cleaned, out_lyr_line_cleaned_buf):
        print('Intersecting polygon buffered w/ out_lyr_lines...')
        input_linesds = ogr.Open(lines_joined) # Opening lines
        input_linelyr = input_linesds.GetLayer()
        featureDefn = input_linelyr.GetLayerDefn()


        for line in input_linelyr: #Loop within SHP lines
            ingeom = line.GetGeometryRef() #Fetch pointer to feature geometry.


            buf_layer = point_buffer.GetLayer()
            intcounter = 0
            for buffer in buf_layer:
                temp = False
                buf_geom = buffer.GetGeometryRef()
                temp = ingeom.Intersects(buf_geom)
                if temp:
                    intcounter += 1


            if intcounter < 3:

                #print(ingeom)
                self.draw_centrelines(ingeom, out_centreline_cleaned)
                out_centreline_cleaned_buf = ingeom.Buffer(self.centreline_buffer)
                self.draw_centrelines(out_centreline_cleaned_buf, out_lyr_line_cleaned_buf)
                dslinecleanedbuf = None


            buf_layer.ResetReading()

            dslinecleaned = None


    def draw_centrelines(self, features, cline_layer):
        defn = cline_layer.GetLayerDefn()
        feat = ogr.Feature(defn)
        feat.SetGeometry(features)
        cline_layer.CreateFeature(feat)

    # # will split a point cloud using a shapefile of multiple polygons
    def split_las(self, shp, in_file, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return_obj = subprocess.run([os.path.join(self.lastool_dir, 'lasclip.exe'), 
            '-i', in_file, '-poly', shp,
            '-split', 
            '-o', f'{os.path.basename(out_dir)}.laz', '-odir', out_dir],
            capture_output=True)
        if return_obj.returncode != 0:
            error_message = 'A non 0 code was returned from lasclip'
            raise RuntimeError(error_message)
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
