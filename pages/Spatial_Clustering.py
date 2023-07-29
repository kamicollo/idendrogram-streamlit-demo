import streamlit as st
import geopandas as gpd
import pydeck as pdk
import pickle
import idendrogram
from idendrogram_streamlit_component import StreamlitConverter
import numpy as np
import altair as alt
from collections import defaultdict
import pandas as pd

st.set_page_config(layout='wide')


st.markdown("""
    ## Spatially constrained clustering

In this example, we'll use `idendrogram` Streamlit component to visually explore sub-clusters on a map. This demo is based on the spatial clustering case study in 
[idendrogram documentation](https://kamicollo.github.io/idendrogram/case-studies/spatial-clustering/), where a couple of demographic variables are used to identify continuous clusters in Vilnius, Lithuania. 

In the case study, we end up with 3 main clusters and build separate dendrograms to visualize their components. Here we do the same - 
you can choose which cluster you are interested in, and then interact with the dendrograms to see where the subclusters fall on the map.
""")

@st.cache_data()
def get_data():
    return gpd.read_parquet("demo/data/vilnius.parquet")

@st.cache_data()
def get_model():
    with open("demo/data/sp_cluster_model.pickle", "rb") as f:
        return pickle.load(f)

def get_selected_nodes_and_links(cl_data):

    
    #setup idendrogram objects
    idd = idendrogram.idendrogram()    
    idd.set_cluster_info(cl_data)

    rel_clusters = np.where(data_for_clustering.groupby('cluster')['POP'].sum() > 1_000)[0]

    colors = [
        '#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99',
        '#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99'
    ]

    color_map = {k: list(int(v.lstrip("#")[i:i+2], 16) 
        for i in (0, 2, 4)) for k,v in zip(rel_clusters, colors)}
    
    clusters = defaultdict(lambda: {'nodes': [], 'links': []})

    #create another dendrogram that goes deeper
    dendrogram = idd.create_dendrogram(
        truncate_mode='lastp', p=100, 
        node_hover_func=my_tooltip, 
        leaf_label_func= lambda *x: "",
        link_color_func=idendrogram.callbacks.link_painter(
            colors = {k: v for k,v in zip(rel_clusters, colors)},
            above_threshold='grey'
        )
    )

    #adjust radius
    for n in dendrogram.nodes:
        _, nodelist = cl_data.get_tree()
        original_ids = nodelist[n.id].pre_order(lambda x: x.id)

        population = data_for_clustering.loc[original_ids, 'POP'].sum()        
        n.radius = np.sqrt(population / 1000) * 1   
        n.fillcolor = n.edgecolor

    selected_node_ids = []

    #iterate through the nodes and save them to appropriate sub-lists based on cluster assignment
    for n in dendrogram.nodes:
        if n.type != 'supercluster':
            _, nodelist = cl_data.get_tree()
            #find first leaf node
            original_id = nodelist[n.id].pre_order(lambda x: x.id)[0]
            #obtain cluster label
            cluster = data_for_clustering.loc[original_id, 'cluster']
            #save this node to the right key in the main dictionary
            clusters[cluster]['nodes'].append(n)
            #also save the node ID to a list - we'll need it for finding links
            selected_node_ids.append(n.id)
    
    #find links that connect nodes of interest (skip any links that link to super clusters)
    selected_links = [l for l in dendrogram.links 
        if (l.children_id[0] in selected_node_ids or l.children_id[1] in selected_node_ids) 
        and l.id in selected_node_ids
    ]

    #iterate through the links and place them into the right dictionary keys
    for l in selected_links:
        _, nodelist = cl_data.get_tree()
        original_id = nodelist[l.id].pre_order(lambda x: x.id)[0]
        cluster = data_for_clustering.loc[original_id, 'cluster']
        clusters[cluster]['links'].append(l)

    return clusters, dendrogram

def form_dendrogram(vals, dendrogram):
    # for plotting purposes, we need to know min/max X and Y coordinates
    # we can obtain that from links list
    x_coord, y_coord = zip(*[(l.x, l.y) for l in vals['links']])
    min_x = np.array(x_coord).flatten().min()
    max_x = np.array(x_coord).flatten().max()
    min_y = np.array(y_coord).flatten().min()
    max_y = np.array(y_coord).flatten().max()

    # picking up axis labels (blank in this case - just for completeness)
    relevant_labels = [a for a in dendrogram.axis_labels if a.x >= min_x and a.x <= max_x]

    # forming a dendrogram object manually
    return idendrogram.Dendrogram(
        links = vals['links'],
        nodes = vals['nodes'],
        x_domain= (min_x, max_x),
        y_domain= (min_y, max_y),
        axis_labels= relevant_labels,
        computed_nodes= True
    )

#custom tooltip functions
def my_tooltip(data, linkage_id):

    #get all original observation IDs that roll up to this node
    _, nodelist = data.get_tree()
    original_ids = nodelist[linkage_id].pre_order(lambda x: x.id)

    #get the associated dataframe rows
    cells = data_for_clustering.loc[original_ids, :]

    #form basic tooltip with total population & number of leaf nodes
    tooltip = {        
        'number of cells': str(nodelist[linkage_id].get_count()),           
        'Total Population':  int(cells['POP'].values.sum()),                
    }

    col_names = [ 
        'x', 'y',
        'pct_children', 
        'pct_retired',          
        'pct_manager_specialist',     
    ]

    #calculate weighted average KPIs for this node
    for var in col_names[2:]:
        pct = (cells[var] * cells['POP']).sum() / cells['POP'].sum()
        ref = (
            data_for_clustering[var] * data_for_clustering['POP']
        ).sum() / data_for_clustering['POP'].sum()
        tooltip[var] = str(int(round(pct - ref,2) * 100)) + '%'

    return tooltip

def delete_selection():
    if 'selection' in st.session_state:
        del st.session_state['selection']

cluster = st.selectbox("Select cluster of interest", options=[0,1,2], format_func= lambda x: ["Center", "Northeast", "West"][x], on_change = delete_selection)

if cluster is not None:    
    data_for_clustering = get_data()        
    model = get_model()
    cl_data = idendrogram.ScikitLearnClusteringData(model)
    clusters, o_dendrogram = get_selected_nodes_and_links(cl_data=cl_data)    
    dendrogram = form_dendrogram(clusters[cluster], o_dendrogram)    

    converter = StreamlitConverter(release=False)

    col1, col2 = st.columns((4,2))

    with col1:
        
        p = converter.convert(dendrogram, width=800, height=600, 
            orientation="bottom", scale="symlog", show_nodes=True, key=cluster,
            margins={'top': 50, 'bottom': 50, 'left': 50, 'right': 50}
        )
        if p is not None:
            st.write(p)
            st.session_state['selection'] = p            

    if 'selection' in st.session_state:        
        _, nodelist = cl_data.get_tree()
        p = st.session_state['selection']
        original_ids = nodelist[p.id].pre_order(lambda x: x.id)
        to_plot = data_for_clustering.iloc[original_ids,:].copy()
    else:        
        to_plot = data_for_clustering[data_for_clustering['cluster'] == cluster].copy()
        p = dendrogram.nodes[0]

    color = "[" + ",".join([str(int(p.fillcolor.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4)]) + "]"
        
    layers = [    
        pdk.Layer(
            "GeoJsonLayer",
            data=to_plot.reset_index(names='foo'),
            get_fill_color=color,
            pickable=True,
            auto_highlight = True,
        ),
    ]

    deck = pdk.Deck(
        layers,
        tooltip=True,
        initial_view_state=pdk.ViewState(
            longitude=25.25,
            latitude=54.7,
            zoom=10,
            min_zoom=5,
            max_zoom=15,    
            map_style=None
        )
    )

    col2.pydeck_chart(deck, use_container_width=True)
    