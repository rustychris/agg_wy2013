import stompy.model.delft.waq_scenario as waq
import six
six.moves.reload_module(waq)

##

agg_grid_shp="Agg2Shapefile/wy2013c_waqgeomAgg2.shp"
hyd_fn="/hpcvol1/public/sfb_dfm_v2/wy2013c/DFM_DELWAQ_wy2013c_adj/wy2013c.hyd"
output_fn="wy2013c-agg/wy2013c-agg.hyd"

##

# Clean inputs
def clean_shapefile(shp_in):
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

shp=clean_shapefile(agg_grid_shp)    


hydro_orig=waq.HydroFiles(hyd_fn)

hydro_agg=waq.HydroAggregator(hydro_in=hydro_orig,
                              agg_shp=shp,
                              sparse_layers=False,
                              agg_boundaries=False)

name=os.path.basename(output_fn.replace('.hyd',''))
class Writer(waq.Scenario):
    name=name
    desc=(name,
          agg_grid_shp,
          'aggregated')
    base_path=os.path.dirname(output_fn)

sec=datetime.timedelta(seconds=1)
if 0:
    # short run for testing: start after some hydro spinup:
    start_time=hydro.time0+sec*hydro.t_secs[100]
    # and run for 1.5 days..
    stop_time=start_time + 4*24*3600*sec
else:
    start_time=hydro.time0+hydro.t_secs[ 0]*sec
    stop_time =hydro.time0+hydro.t_secs[-1]*sec

writer=Writer(hydro=hydro_agg,
              start_time=start_time,
              stop_time=stop_time)

writer.cmd_write_hydro()

