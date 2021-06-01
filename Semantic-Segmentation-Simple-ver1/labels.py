from collections import namedtuple

Label = namedtuple( 'Label' , ['name', 'id', 'color'] )

labels = [
    #       name                     Id       color
    
    Label(  'Pole'                 , 0       ,   (  0,  0, 64)     ),
    Label(  'Pole'                 , 0       ,   (192,192,128)     ),
    Label(  'SignSymbol'           , 1       ,   (128,128, 64)     ),
    Label(  'SignSymbol'           , 1       ,   (192,128,128)     ),
    Label(  'SignSymbol'           , 1       ,   (  0, 64, 64)     ),
    
    Label(  'Bicyclist'            , 2       ,   (  0,128,192)     ),
    Label(  'Bicyclist'            , 2       ,   (192,  0,192)     ),
    Label(  'Pedestrian'           , 3       ,   ( 64, 64,  0)     ),
    Label(  'Pedestrian'           , 3       ,   (192,128, 64)     ),
    Label(  'Pedestrian'           , 3       ,   ( 64,128, 64)     ),
    Label(  'Pedestrian'           , 3       ,   ( 64,  0,192)     ),
    
    Label(  'Building'             , 4       ,   ( 64,  0, 64)     ),
    Label(  'Building'             , 4       ,   (128,  0,  0)     ),
    Label(  'Building'             , 4       ,   (192,  0,128)     ),
    Label(  'Building'             , 4       ,   ( 64,192,  0)     ),
    Label(  'Building'             , 4       ,   (  0,128, 64)     ),
    Label(  'Fence'                , 5       ,   ( 64, 64,128)     ),
    
    Label(  'Pavement'             , 6       ,   ( 64,192,128)     ),
    Label(  'Pavement'             , 6       ,   (128,128,192)     ),
    Label(  'Pavement'             , 6       ,   (  0,  0,192)     ),
    Label(  'Road'                 , 7       ,   (128, 64,128)     ),
    Label(  'Road'                 , 7       ,   (128,  0,192)     ),
    
    Label(  'Car'                  , 8       ,   ( 64,128,192)     ),
    Label(  'Car'                  , 8       ,   (128, 64, 64)     ),
    Label(  'Car'                  , 8       ,   (192,128,192)     ),
    Label(  'Car'                  , 8       ,   ( 64,  0,128)     ),
    
    Label(  'Sky'                  , 9       ,   (128,128,128)     ),
    
    Label(  'Tree'                 , 10      ,   (192,192,  0)     ),
    Label(  'Tree'                 , 10      ,   (128,128,  0)     ),
]