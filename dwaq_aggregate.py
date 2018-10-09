"""
Sample driver script for aggregation of dwaq hydrodynamics
"""

import matplotlib
import os
import datetime
import stompy.model.delft.waq_scenario as waq

# Inputs:
agg_grid_shp="Agg2Shapefile_edited/agg_shapefile_edited.shp"
hyd_fn="/hpcvol1/public/sfb_dfm_v2/wy2013c/DFM_DELWAQ_wy2013c_adj/wy2013c.hyd"
output_fn="wy2013c-agg/wy2013c-agg.hyd"


#----------

# Clean inputs
def clean_shapefile(shp_in):
    """
    break multipolygons into individual polygons.
    """
    from stompy.spatial import wkb2shp
    geoms=wkb2shp.shp2geom(agg_grid_shp)

    multi_count=0
    
    new_geoms=[]
    for fi,feat in enumerate(geoms):
        if feat['geom'].type=='Polygon':
            new_geoms.append(feat['geom'])
        else:
            multi_count+=1
            for g in feat['geom'].geoms:
                new_geoms.append(g)
    if multi_count:
        cleaned=agg_grid_shp.replace('.shp','-cleaned.shp')
        assert cleaned!=agg_grid_shp
        wkb2shp.wkb2shp(cleaned,new_geoms,overwrite=True)

        return cleaned
    else:
        return shp_in


# Processing:

# remove multipolygons from inputs 
shp=clean_shapefile(agg_grid_shp)    

# open the original dwaq hydrodynamics
hydro_orig=waq.HydroFiles(hyd_fn)

# create object representing aggregated hydrodynamics
# sparse_layers: for z-layer inputs this can be True, in which cases cells are only output for the
#    layers in which they are above the bed.  Usually a bad idea.  Parts of DWAQ assume
#    each 2D cell exists across all layers
# agg_boundaries: if True, multiple boundary inputs entering a single aggregated cell will be
#   merged into a single boundary input.  Generally best to keep this as False.
hydro_agg=waq.HydroAggregator(hydro_in=hydro_orig,
                              agg_shp=shp,
                              sparse_layers=False,
                              agg_boundaries=False)


# The code to write dwaq hydro is wrapped up in the code to write a dwaq model inp file,
# so we pretend to set up a dwaq simulation, even though the goal is just to write
# the hydro.
name=os.path.basename(output_fn.replace('.hyd',''))
class Writer(waq.Scenario):
    name=name
    desc=(name,
          agg_grid_shp,
          'aggregated')
    # output directory inferred from output hyd path
    base_path=os.path.dirname(output_fn)

# Define the subset of timesteps to write out, in this case the
# whole run.
sec=datetime.timedelta(seconds=1)
start_time=hydro_agg.time0+hydro_agg.t_secs[ 0]*sec
stop_time =hydro_agg.time0+hydro_agg.t_secs[-1]*sec

# probably would have been better to just pass name, desc, base_path in here,
# rather than using a shell subclass.
writer=Writer(hydro=hydro_agg,
              start_time=start_time,
              stop_time=stop_time)

# This step is super slow.  Watch the output directory for progress.
# Takes ~20 hours on HPC for the full wy2013 run.
writer.cmd_write_hydro()
