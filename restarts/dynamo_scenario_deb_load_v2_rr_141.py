"""
2018-03-01: modified from dynamo_scenario_deb.py but instead of imposing POTWs 
as boundary concentration, nutrient loads are imposed. 

2020-03-18: RH: Test restart capabilities with a nutrient run
"""

import os
import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import numpy as np
import xarray as xr
import pandas as pd
import shapely.wkt
from six import iteritems
from importlib import reload

from stompy import utils
from stompy.model.delft import waq_scenario
reload(waq_scenario)

##
utils.path("/hpcvol1/Mugunthan/Scripts/RunLaunchers")
import BayDynamoMin_v2_rr_141 as BayDynamo
reload(BayDynamo) # just in case I made some modification to dynamo
## 

# Specify locations of outside data sources:

# POTW data, checked out in subdirectory:
potw_fn='/hpcvol1/zhenlin/sfbay/common/sfbay_potw/outputs/sfbay_potw.nc'

# Selection of hydrodynamics data:
# this has spatially variable shear stresses output from the hydrorun
rel_hyd_path="/hpcvol1/Mugunthan/Inputs/hydro/WY2013/agg141_tau-lp-pass_params/com-agg141_tau-lp.hyd"

# Load those external datasets:
sfbay_potw=xr.open_dataset(potw_fn)

# External reference to radiance data defined in radsurf() in Scen() class

hydro=waq_scenario.HydroFiles(hyd_path=rel_hyd_path,enable_write_symlink=True)

## 

PC=waq_scenario.ParameterConstant
Sub=waq_scenario.Substance
IC=waq_scenario.Initial

class Scen(BayDynamo.BayDynamo):
    name="sfbay_dynamo000"
    desc=('sfbay_dynamo000',
          'wy2013',
          'full run')
    base_x_dispersion = 0. # m**2/s - constant horizontal dispersion coefficient 
    integration_option="""16.62 ;
    LOWER-ORDER-AT-BOUND NODISP-AT-BOUND
    BALANCES-OLD-STYLE BALANCES-GPP-STYLE 
    BAL_NOLUMPPROCESSES BAL_NOLUMPLOADS BAL_NOLUMPTRANSPORT
    BAL_NOSUPPRESSSPACE BAL_NOSUPPRESSTIME
    """
    _base_path='auto'
    delwaq2_path = '/home/zhenlin/src/delft3d/delft3d/src/bin/delwaq2_0.2T'
    time_step=1000 # matches the hydro # dwaq HHMMSS integer format; make half of 1000 because the model crushed possibly due to long time step. 

    map_formats=['binary']
    
    #storm sources are generally unknown, so will apply a constant boundary nutrient condition now
    storm_sources=['SCLARAVW2_flow',
                   'SCLARAVW1_flow',
                   'SCLARAVW4_flow',
                   'SCLARAVW3_flow',
                   'UALAMEDA_flow',
                   'EBAYS_flow',
                   'COYOTE_flow',
                   'PENINSULb1_flow',
                   'EBAYCc3_flow',
                   'USANLORZ_flow',
                   'PENINSULb3_flow',
                   'PENINSULb4_flow',
                   'EBAYCc2_flow',
                   'PENINSULb6_flow',
                   'PENINSULb2_flow',
                   'PENINSULb7_flow',
                   'PENINSULb5_flow',
                   'SCLARAVCc_flow',
                   'SCLARAVW5_flow',
                   'MARINS1_flow',
                   'EBAYCc6_flow',
                   'EBAYCc1_flow',
                   'EBAYCc5_flow',
                   'EBAYCc4_flow',
                   'MARINN_flow',
                   'NAPA_flow',
                   'CCOSTAW2_flow',
                   'CCOSTAW3_flow',
                   'MARINS3_flow',
                   'MARINS2_flow',
                   'PETALUMA_flow',
                   'SONOMA_flow',
                   'CCOSTAW1_flow',
                   'SOLANOWc_flow',
                   'CCOSTAC2_flow',
                   'EBAYN1_flow',
                   'EBAYN4_flow',
                   'EBAYN2_flow',
                   'EBAYN3_flow',
                   'SOLANOWa_flow',
                   'SOLANOWb_flow',
                   'CCOSTAC3_flow',
                   'CCOSTAC1_flow',
                   'CCOSTAC4_flow']
    
    delta_sources=['Jersey_flow',
                   'RioVista_flow']

    sea_sources=[ 'Sea_ssh' ]
    
    # some of the potw sources are defnitely rivers, but since we will apply
    # nutrients at the boundary (not in the bottom) anyway for now, I will not
    # distinguish between the two. 
    
    potw_sources = ['american','benicia','calistoga','cccsd','central_marin',
                    'ch','chevron', 'ddsd', 'ebda', 'ebmud', 'fs', 'lg', 
                    'marin5', 'millbrae', 'mt_view', 'napa', 'novato', 
                    'palo_alto', 'petaluma', 'phillips66', 'pinole', 'rodeo',
                    'san_jose', 'san_mateo', 'sausalito', 'sf_southeast', 
                    'sfo', 'shell', 'sonoma_valley', 'south_bayside', 
                    'south_sf', 'st_helena', 'sunnyvale', 'tesoro', 
                    'treasure_island', 'valero', 'vallejo', 
                    'west_county_richmond', 'yountville']   
    # The above four sources complete all boundary sources for the run
    all_sources = storm_sources+ delta_sources+sea_sources+potw_sources
    
    def add_potw_loads(self):
        """
        Add POTWs as point loads in the model.

        Read in source_locations.csv, the output of select_source_locations.py,
        to choose the subset of POTWs.
        Create a discharge for each, which is assumed to be at the bed.
        Create a substance for each, assigned to its discharge, based on flow.
        """        
        boundaries=self.hydro.boundary_defs()
        allitems = [boundary.decode("utf-8") for boundary in set(boundaries['type'])]
        group_boundary_links = self.hydro.group_boundary_links() # read boundary location and name information from DFM .bnd file
          
        g=self.hydro.grid()
        self.hydro.infer_2d_elements()
        
        source_segs={} # name => discharge id
        for k in allitems:

            if k not in self.potw_sources: # the river source will be added as boundary condition rather than loads                
                continue 
            
            if k=='millbrae':
                site_name = ['burlingame','millbrae'] # burlingame and millbrae merges into millbrae. 
            elif k=='marin5':
                site_name = ['sasm','marin5'] # sasm and marin5 merges into marin5
            else:
                site_name=k   
           
            bdn = np.nonzero(group_boundary_links['name']==k) #getting index for boundary
            assert(len(bdn)==1)                
            bdn = np.asscalar(np.asarray(bdn)) #bdn is in an annoying tuple type
            line = shapely.wkt.loads(group_boundary_links['attrs'][bdn]['geom'])
            xutm = line.xy[0][0]
            yutm = line.xy[1][0]                

            xy=np.array( [xutm, yutm] )
            elt=g.select_cells_nearest(xy)
            # put everybody at the bed - 
            seg=np.nonzero( self.hydro.seg_to_2d_element==elt )[0][-1]
    
            # the same segment can receive multiple loads, so stick to seg-<id>
            # for naming here, as opposed to naming discharge points after a
            # specific source.
            source_segs[k]="seg-%d"%seg
    
            self.add_discharge(seg_id=seg,load_id=source_segs[k],on_exists='ignore')            
            
            # name the substance after the source, and the discharge already
            # named after the source.
            if isinstance(site_name,str):# one loading per site
                ds_site=sfbay_potw.sel(site=site_name.encode())
            else: #two loading per site
                ds_site=sfbay_potw.sel(site= [site_name[0].encode(),site_name[1].encode()])
            
            # replace nan values with something else
            if np.any(np.isnan(ds_site.NOx_load.values)):
                print("Site %s: has nan NO3"%site_name)
                ds_site.NOx_load.values=ds_site.NOx_load.values
                assert np.all(np.isfinite(ds_site.NOx_load.values))
               
            if np.any(np.isnan(ds_site.NH3_load.values)):
                print("Site %s: has nan NH4"%site_name)
                ds_site.NH3_load.values=0*ds_site.flow.values

            if np.any(np.isnan(ds_site.PO4_load.values)):             
                    # based on plots from potw_add_po4.py,
                    # choose constant concentration of
                    # 2mg/L.
                conc_mg_l=2.0
                self.log.warning("Site %s: nan PO4, will use 2.0 mg/l P"%site_name)     
                ds_site.PO4_load.values = conc_mg_l*ds_site.flow*86400/1000.    
            
            
            if isinstance(site_name,str):   # one loading per site

                # data comes in with units of kg/day, but d-waq wants g/s.
                nh3_load_g_s = (ds_site.NH3_load*1000/86400.).to_series() #this already comes with time because ds_site is a panda array
                no3_load_g_s = (ds_site.NOx_load*1000/86400.).to_series()
                po4_load_g_s = (ds_site.PO4_load*1000/86400.).to_series()
            else:# two loadings per site; This is probably not the best way to deal with it. 
                sites = ds_site.site
                nh3_load_g_s = (ds_site.sel(site=sites[0]).NH3_load*1000/86400.).to_series() + \
                (ds_site.sel(site=sites[1]).NH3_load*1000/86400.).to_series()
                no3_load_g_s = (ds_site.sel(site=sites[0]).NOx_load*1000/86400.).to_series() + \
                (ds_site.sel(site=sites[1]).NOx_load*1000/86400.).to_series()
                po4_load_g_s = (ds_site.sel(site=sites[0]).PO4_load*1000/86400.).to_series() + \
                (ds_site.sel(site=sites[1]).PO4_load*1000/86400.).to_series()                
                

            self.add_load(source_segs[k],substances='NH4',data=nh3_load_g_s)
            self.add_load(source_segs[k],substances='NO3',data=no3_load_g_s)
            self.add_load(source_segs[k],substances='PO4',data=po4_load_g_s)                
               
                
            # Conservative versions:
            if 'NH4cons' in self.substances:
                self.add_load(source_segs[k],substances='NH4_cons',data=nh3_load_g_s)
            if 'NO3cons' in self.substances:
                self.add_load(source_segs[k],substances='NO3_cons',data=no3_load_g_s)    

    def init_loads(self):
        super(Scen,self).init_loads()
        self.add_potw_loads()
        
    
    def init_substances(self):
        subs=super(Scen,self).init_substances()

        # first cut at unifying the handling of src_tags 
        
        subs['OXY']=Sub(initial=IC(default=8.0)) # start DO at 8mg/L
        #subs['continuity']=Sub(initial=IC(default=1.0))

        # Based on Nutrient Conceptual model data, (pg 48)
        # NH4 in April: 3uM in Central, 4ish in SSFB and SPB,
        # so punt with 2uM coastal.
        # similarly 20uM NO3 in Central.  30-35 uM is upper bound on ocean
        # NO3, based on freshly upwelled water.  More common is 10uM in the
        # coastal ocean.

        # call the initial condition spatially constant 10uM.

        # 1 umol/L NH4 * 14 gN/mol * 1milli/1000micro
        subs['NH4']        = Sub(initial=IC(default=2 * 14/1000.))
        # had been 1.1 - i.e. lower south bay ambient 70uM.
        subs['NO3']        = Sub(initial=IC(default=10 * 14/1000.))
        # Follow Largier and Stacey - coastal phosphate roughly Redfield
        # to the nitrate.
        subs['PO4']        = Sub(initial=IC(default=10/16. * 31/1000.))

        # based on SFB Nutrient Conceptual model fig 6.7, choose a nominal
        # 100uM Si initial condition
        subs['Si']         = Sub(initial=IC(default=100.*28/1000))
        
        add_cons=False # whether to add conserved nutrient mimics
        if add_cons:
            subs['NH4cons']=subs['NH4'].copy()
            subs['NO3cons']=subs['NO3'].copy()

        boundaries=self.hydro.boundary_defs()
        allitems = [boundary.decode("utf-8") for boundary in set(boundaries['type'])]        
         
        # continuity tracer
        self.src_tags.append(dict(tracer='continuity',
                                  items=self.all_sources,
                                  value=1.0))
        
        # Setting boundary condition for OXY
        # Rivers and the ocean get background DO, but leave potws with 0 DO.
        self.src_tags.append(dict(tracer='oxy',items=self.storm_sources + self.delta_sources + self.sea_sources,
                                  value=subs['oxy'].initial.default))


        # Setting boundary condition for Si
        # Rivers get a base Si concentration, adapted from Peterson, 1978
        # Discharges stick with 0 Si.
        self.src_tags.append(dict(tracer='si',items=self.delta_sources+self.storm_sources,
                                  value=200 * 28./1000 ) )
        self.src_tags.append(dict(tracer='si',items=self.sea_sources,
                                  value=30 * 28./1000 ) )
 
       
        #setting boundary condition for nutrient sources
        #1) the ocean and river gets the initial condition
        for tracer in ['NO3','NH4','PO4']:
            self.src_tags.append(dict(tracer=tracer,
                                      items=self.sea_sources+self.storm_sources,
                                      value=subs[tracer].initial.default))
            
        #2) the delta and potws get the nutrient values from sfbay_potws

        def to_series(fld,site_name):
            try:
                return sfbay_potw[fld].sel(site=site_name).to_dataframe()[fld]
            except KeyError: # annoying py3 bytes vs. string
                return sfbay_potw[fld].sel(site=site_name.encode()).to_dataframe()[fld]  
                    
        for k in self.delta_sources: 
            
            if k=='RioVista_flow':
                site_name = 'false_sac'
            elif k=='Jersey_flow':
                site_name = 'false_sj'       
            else:
                site_name=k                    
                
            items=k
            nox_conc=to_series('NOx_conc',site_name)    
            nh3_conc=to_series('NH3_conc',site_name)    
            po4_conc=to_series('PO4_conc',site_name)
     
            self.src_tags.append(dict(tracer='NO3',
                                      items=items,
                                      value=nox_conc))
            
            
            self.src_tags.append(dict(tracer='NH4',
                                      items=items,
                                      value=nh3_conc))               
    
            self.src_tags.append(dict(tracer='PO4',
                                      items=items,
                                      value=po4_conc))      
           
        return subs

    def init_parameters(self):
        # choose which processes are enabled.  Includes some
        # parameters which are not currently used.
        params=super(Scen,self).init_parameters()
        
        params['NOTHREADS']=PC(8) # better to keep it to real cores?
        # params['RefDay']=PC(274.) # what's reference day??

        # looser than this and the errors are quite visible.  This already feels lenient,
        # but in 2016-06 tighter tolerances led to non-convergence.
        # 2017-03-17: This had been 1.0e-5, but I would like to see if it can handle a slightly tighter
        # bound.
        # 2018-02-02: failed at 1.0-6, so I will change the tolerance to 1.0e-5
        # 2018-03-08: 1.0e-5 shows some weird oscillation, so change it back to 1.0e-6

        params['Tolerance']=PC(1.0e-6)
        
        # if convergence becomes an issue, there *might* be more info here:
        # params['Iteration Report']=PC(1)
        params['TimMultBl']=PC(48) # daily bloom step for a 0.5h waq step.

        return params
        
    def cmd_default(self):
        self.cmd_write_hydro()
        self.cmd_write_inp()
        self.cmd_delwaq1()
        self.cmd_delwaq2()        
        self.cmd_write_nc()
        
    def __init__(self,*a,**k):
        super(Scen,self).__init__(*a,**k)

        extra_fields=('salinity',
                      'temp',
                      'TotalDepth',
                      'volume',
                      'depth',
                      'tau',
                      'velocity')
        self.map_output+=extra_fields
        #self.hist_output+=extra_fields
        
        #bio_fields=('fPPDiat','LimRadDiat')
        bio_fields=('fPPDiat','LimNutDiat','LimRadDiat','fBurS1DetC','fBurS2DetC')
        #self.map_output+=bio_fields
        self.hist_output+=bio_fields
        self.hist_output+=('volume','salinity','temp',)
        
        self.stat_output +=bio_fields
        self.stat_output +=('DZ_Diat','dPPDiat','dSedPOC1','dSedDiat')        
#        self.stat_output +=('DZ_Diat','dPPDiat','GroMrt_Diat','dcPPDiat',\ #diat
#                            'dNITRIF','dMinPON1','dNH4Upt','dNH4Aut','dZ_NRes',\ #NH4
#                            'dDenitSed','dDenitWat','dNITRIF','dNO3Upt','dNiDen')\ #NO3
        

        DAILY=1000000
        self.map_time_step=DAILY # daily
        self.mon_time_step=6000 # every 60 min. -- format here mmss - this is not an integer specification of time!!
        self.hist_time_step=6000 # every 60 min        
        #self.add_usgs_transect_monitor() 
        self.add_usgs_monitor_areas()
        self.add_disp()
		
        """ add subembayment transects from a defined shapefile
        """
        #shp_fn_lines = "/hpcvol1/zhenlin/sfbay/best_runs/shapefiles/Agg_exchange_lines.shp"
#        shp_fn_lines = "/hpcvol1/Mugunthan/Inputs/Grid/Shapefiles/Agg_exchange_lines_141.shp"
#        #shp_fn_polys = "/hpcvol1/zhenlin/sfbay/best_runs/shapefiles/Agg_mod_contiguous.shp"
#        shp_fn_polys = "/hpcvol1/Mugunthan/Inputs/Grid/Shapefiles/Agg_mod_contiguous_141.shp"
#        self.add_transects_from_shp(shp_fn_lines,clip_to_poly=False, on_edge=True)
#        self.add_monitor_from_shp(shp_fn_polys)
        
    def add_disp(self):
        # Add externally specified dispersion coefficients 
        K = np.loadtxt('/hpcvol1/rusty/dwaq/agg141_tau-lp/dispersions.txt')
        self.dispersions['anisoK']=waq_scenario.DispArray(substances=".*",data=K)

    def add_usgs_transect_monitor(self):
        """ lump all the usgs sample locations into one monitoring area to
        get broad-brush balances.
        This will give an area averaged value in the hist file
        """
        # pulled from usgs transect
        xy=np.array( [[  582676.89874631,  4147608.43485955],
                      [  581341.46840116,  4148519.88353528],
                      [  580146.58169605,  4150172.63870161],
                      [  577628.08464891,  4151628.13229029],
                      [  576586.78934572,  4152728.03349741],
                      [  574367.58634256,  4153817.4355253 ],
                      [  571544.03202531,  4156751.47189957],
                      [  569029.11834886,  4158209.56929831],
                      [  566663.74310412,  4159484.68975885],
                      [  564437.61852413,  4161871.12435409],
                      [  562511.08020062,  4163705.51662575],
                      [  560585.48008233,  4165540.46266197],
                      [  559528.12680624,  4169416.15506171],
                      [  558330.08584425,  4172551.26246545],
                      [  558453.38267663,  4175880.75757463],
                      [  556516.2466788 ,  4179935.62097193],
                      [  556498.48564466,  4182524.40826308],
                      [  553393.91771774,  4186017.28803373],
                      [  550881.84804991,  4188960.18742818],
                      [  550418.09174599,  4192840.75604696],
                      [  548637.36614153,  4196528.48190827],
                      [  549476.11342052,  4203190.88711621],
                      [  552234.90982132,  4206721.74706252],
                      [  555291.3004305 ,  4209145.90400182],
                      [  560391.72612233,  4211771.20928497],
                      [  564623.62825589,  4212913.26912799],
                      [  569450.11460174,  4212767.9811421 ],
                      [  572530.45877571,  4211685.15653964],
                      [  574451.68089209,  4209483.16829452],
                      [  579258.93975789,  4211562.78942538],
                      [  584650.55476478,  4213466.4860191 ],
                      [  589481.4076512 ,  4212963.23308994],
                      [  593444.05879313,  4211712.976593  ],
                      [  598413.99062463,  4211956.42813898],
                      [  600443.0439842 ,  4213460.80109604],
                      [  605270.37332549,  4213336.82658049],
                      [  614930.91266961,  4223085.77808305]])

        # hydro=waq_scenario.HydroFiles(os.path.join(dwaq_dir,'com-sfbay_dynamo000.hyd'))

        self.hydro.infer_2d_elements()
        g=self.hydro.grid()

        elt_sel=[g.select_cells_nearest(pnt,inside=False)
                 for pnt in xy]

        segs=[ np.nonzero(elt==hydro.seg_to_2d_element)[0] 
               for elt in np.unique(elt_sel) ]
        segs=np.concatenate(segs)
        self.monitor_areas=self.monitor_areas + (('usgs_transect',segs),)


    def add_usgs_monitor_areas(self):

#        # pulled from usgs transect station 34
#        xy= [  580146.58169605,  4150172.63870161]
#        self.hydro.infer_2d_elements()
#        g=self.hydro.grid()
#
#        elt_sel=g.select_cells_nearest(xy,inside=False)                
#
#        #This gives you depth average value of the monitoring location
#        #segs=np.nonzero(elt_sel==hydro.seg_to_2d_element)[0]                
#        #segs=np.concatenate(segs)
#        # This gives you surface values of the monitoring location
#        segs = np.nonzero(elt_sel==hydro.seg_to_2d_element)[0][0]        
#        self.monitor_areas=self.monitor_areas + (('usgs_34',[segs]),) #convert the int, segs, to list so that we can apply len() on it. 

        
        dfs = pd.read_csv('/hpcvol1/zhenlin/sfbay/sfbay_fullrun/stations.csv')
        xy = np.array([dfs['utm_x'],dfs['utm_y']]).transpose()
        
        self.hydro.infer_2d_elements()
        g=self.hydro.grid()

        elt_sel=[g.select_cells_nearest(pnt,inside=False)
                 for pnt in xy]

#        segs=[ [np.squeeze(np.nonzero(elt==hydro.seg_to_2d_element))[0]] # output only the surface level
#               for elt in elt_sel ]
#        #segs=np.concatenate(segs)  
#                 
#        for i in np.arange(len(segs)):
#            self.monitor_areas=self.monitor_areas + ((dfs['Station'].values[i],segs[i]),)

        segs=[ np.squeeze(np.nonzero(elt==hydro.seg_to_2d_element)) # output the whole water column
               for elt in elt_sel ]

        for i in np.arange(len(segs)):
            self.monitor_areas=self.monitor_areas + ((dfs['Station'].values[i]+'_column',segs[i]),)

sec=datetime.timedelta(seconds=1)

if 1:  # short run for testing:
    start_time=hydro.time0+hydro.t_secs[0]*sec
    # and run for 20 days
    stop_time=start_time + 24*3600*sec
    map_time_step=3000 # half hour
if 0: # long run
    start_time=hydro.time0+hydro.t_secs[ 0]*sec
    #stop_time =start_time+265*24*3600*sec   #run for 265 days
    stop_time = hydro.time0+hydro.t_secs[-2]*sec #run for entire water year
    map_time_step=1000000 # daily

scen=Scen(hydro=hydro,
          start_time=start_time,
          stop_time=stop_time,
          base_path = '/hpcvol1/rusty/dwaq/agg_wy2013/restarts/dwaq-original/',
          overwrite=True)
scen.map_time_step=map_time_step
    
scen.cmd_default()

##

print("-"*30 + "RESTART" + "-"*30)

scen_restart=Scen(hydro=hydro,
                  start_time=start_time,
                  stop_time=stop_time,
                  base_path = '/hpcvol1/rusty/dwaq/agg_wy2013/restarts/dwaq-restart/',
                  overwrite=True,
                  restart_file=os.path.join(scen.base_path,scen.name+"_res.map"))
scen_restart.map_time_step=map_time_step
scen_restart.cmd_default()
