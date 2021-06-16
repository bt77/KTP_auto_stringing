### File/Dir Name ###
CLASSIFIED_POINT_CLOUD = "Network"  # Classified Point Cloud
STRUCTURE_TOP = 'Tops.shp'
LINES_JOINED = '01_lines_joined.shp'   # name of lines created when joining all points
POINT_BUFFER = '02_point_buffer.shp'
CENTRELINE = '03_centreline_cleaned.shp'
CENTRELINE_BUFFER_SHAPEFILE = '04_centreline_cleaned_buf.shp'
SPAN = 'Spans'
SPAN_CLEANED = 'Spans_cleaned'
WIRE = 'Wires'




### Paths ####
# LASTools dir
LASTOOLS = (r"D:\Code\LAStools\bin")
# Input dir
INPUT = (r"D:\Code\auto_stringing\Input_test1")
# Output dir
OUTPUT = (r"D:\Code\auto_stringing\Output_test1")
# Test span dir for extract_wires()
TEST_SPAN = (r"D:\Code\auto_stringing\Output_test1\test_span")    


### Other Variables###
STRUCTURE = '15'    # Feature code of the Structure class in classified point cloud 
CONDUCTOR = '16'    # Feature code of the Conductor class in classified point cloud 
BUFFER_DISTANCE = 5.0
SPAN_LENGTH = 10    # Threshold for drawing centrelines
TOPS_DIST = 250     # Distance threshold for drawing centre lines
SPAN_POINTS = 65    # Minimum #points required for a span, used to select valid spans
CENTRELINE_BUFFER = 1