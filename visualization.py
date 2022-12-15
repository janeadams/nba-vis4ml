#!/usr/bin/env python
# coding: utf-8

# # Explainable Neural Binary Analysis
# 
# CS7295 Visualization for Machine Learning
# 
# Jane Adams and Michael Davinroy
# 
# 
# **IMPORTANT**: This notebook will not work in Google Drive. Please download and run locally.

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install -r requirements.txt')


# In[2]:


import networkx as nx
import json
import numpy as np
import pandas as pd
from EoN import hierarchy_pos
import plotly.express as px
import plotly.graph_objects as go
import itertools
from urllib.request import urlopen


# In[3]:


# Data: https://github.com/Cisco-Talos/binary_function_similarity
# GNN: https://github.com/deepmind/deepmind-research/tree/master/graph_matching_networks


# In[4]:


data_dict = {
    'node':'Function Address',
    'bb_len': 'Length of Basic Blocks',
    'instruction_count': 'Number of Instructions',
    'bb_mnems': 'Instruction Mnemonics',
    'source_count': 'Departures from this function', 
    'target_count': 'Arrivals to this function'
}


# In[5]:


debug=False
dataset = 'Dataset-1'
testtrain = 'testing'
cutoff = 50
rel_path = 'http://dev.universalities.com/nba/'


# In[6]:


def get_simDF(dataset=dataset, testtrain=testtrain, cutoff=cutoff, rank=True, subset='OPC-200'):
    sim_dfs = []
    for posneg in ['pos','neg']:
        path = f'{rel_path}sim_scores/{posneg}{"_rank" if rank else ""}_{testtrain}_{dataset}_GGSNN_{subset}_e10.csv'
        print(f'Accessing similarity data path {path}...')
        df = pd.read_csv(path)
        df['class'] = posneg
        df['distance']=[-s for s in df['sim']]
        df = df.sample(n=cutoff).sort_values(by='distance').reset_index().drop(columns='index')
        sim_dfs.append(df)
    simDF = pd.concat(sim_dfs).sort_index(kind='merge').reset_index().drop(columns='index')
    return simDF


# In[7]:


#simDF = get_simDF(rank=False)
simDF = pd.read_csv(f'{rel_path}sample_sim.csv')


# In[8]:


def visualize_simDF(simDF):
    fig = px.histogram(simDF, x='distance', color='class', range_x=[0,100])
    fig.update_layout(
        template='plotly_white',
        title=f'Histogram of {dataset} Euclidean Distance by Classification',
        #width=600,
        #height=400
    )
    fig.update_xaxes(title='Euclidean Distance')
    fig.update_yaxes(title='Count')
    return fig


# In[9]:


model_explorer = visualize_simDF(simDF)


# In[10]:


def get_feature_paths(idb, t='acfg_disasm', db=dataset, testtrain='testing'):
    stripped = idb.split('/')[-1][:-4]
    if 'acfg' in t:
        return f'{rel_path}DBs/{db}/features/{testtrain}/{t}_{db}_{testtrain}/{stripped}_{t}.json'
    if t == 'fss':
        return f'{rel_path}DBs/{db}/features/{testtrain}/{t}_{db}_{testtrain}/{stripped}_Capstone_True_fss.json'
    else: return 'ERR'


# In[11]:


def get_full_binary(filepath):
    print(f'Accessing similarity data path {filepath}...')
    response = urlopen(filepath)
    o = json.loads(response.read())
    l = len(list(o.keys()))
    if (l > 1):
        print(f'Warning! Code contains {l} keys')
    code = o[list(o.keys())[0]]
    #print(f'Binary contains {len(list(code.keys()))} functions')
    return code


# In[12]:


def filter_edges(function):
    problem_targets = []
    for e_item in list(itertools.chain(*function['edges'])):
        if e_item not in function['nodes']:
            print()
            print("*** WARNING ****")
            print(f'{e_item} not in node list')
            problem_targets.append(e_item)
            print()
    new_edges = []
    for edge in function['edges']:
        for problem in problem_targets:
            if problem in edge:
                if debug: print(f'Removed {edge} from network')
            else:
                new_edges.append(edge)
    function['edges']=new_edges
    return function


# In[13]:


def extract_network(address, binary, filter_e=False):
    function = binary[address]
    if filter_e:
        function = filter_edges(function)
    return function


# In[14]:


def remove_components(G, nodeDF, edgeDF):
    components = list(nx.connected_components(G))
    if debug:
        print(f'Found {len(list(components))} components in the graph')
    if len(list(components))>1:
        if debug: print(f'Filtering graph of size {len(list(G.nodes()))}...')
        trees = []
        subgraphs = []
        for i, node_list in enumerate(components):
            if debug:
                #print(f'node_list: {list(node_list)}')
                print(f'Component #{i} has {len(node_list)} nodes')
            Gc = G.subgraph(list(node_list))
            if nx.is_tree(Gc):
                if debug: print(f'...and is tree')
                trees.append(Gc)
            else:
                if debug: print(f'...and is NOT a tree')
                subgraphs.append(Gc)
            if debug: print()
        if debug:
            print(f'Found {len(trees)} tree(s) and {len(subgraphs)} non-tree subgraphs')
        if len(trees)>0:
            G = trees[0].copy()
        else:
            G = subgraphs[0].copy()
        if debug: print(f'Filtered graph to size {len(list(G.nodes()))}')
        nodeDF = nodeDF.filter(items = list(G.nodes()), axis=0)
        for i in nodeDF.index:
            if i not in list(G.nodes()):
                if debug: print(f'Removing {i} from nodeDF')
        edgeDF = edgeDF[edgeDF[['source','target']].isin(list(G.nodes())).any(1)]
    return G, nodeDF, edgeDF


# In[15]:


def parse_network(network, debug=True, remove_c=False):
    data_cols = ['node','bb_len', 'bb_mnems', 'bb_norm', 'bb_disasm', 'b64_bytes', 'bb_heads', 'source_count', 'target_count']
    edgeDF = pd.DataFrame(network["edges"], columns=['source','target'])
    nodeDF = pd.DataFrame(columns=data_cols)
    for n in network["nodes"]:
        metadata = network['basic_blocks'][str(n)].copy()
        metadata['node'] = n
        metadata['source_count'] = list(edgeDF['source']).count(n)
        metadata['target_count'] = list(edgeDF['target']).count(n)
        nodeDF = pd.concat([nodeDF, pd.DataFrame.from_records([metadata])])
    nodeDF['instruction_count'] = [len(m) for m in nodeDF['bb_mnems']]
    nodeDF = nodeDF.sort_values(by='target_count').set_index('node')
    G = nx.from_pandas_edgelist(edgeDF)
    if remove_c:
        G, nodeDF, edgeDF = remove_components(G, nodeDF, edgeDF)
    root = nodeDF.sort_values(by='source_count').index[0]
    if debug: print(f'Root node: {root}')
    try:
        pos = hierarchy_pos(G, root=root)
        if debug: print('Hierarchical positioning worked!')
    except:
        pos=nx.kamada_kawai_layout(G)
        if debug: print('Hierarchical positioning failed; using spring layout instead')
    
    def get_pos(node):
        try:
            return pos[node]
        except:
            return (-1,-1)
    
    nodeDF[['y','x']] = [get_pos(n) for n in nodeDF.index]
    nodeLookup = nodeDF.to_dict(orient='index')
    
    def get_xy(s, t, xy, debug=True):
        x = None
        y = None
        try:
            x = nodeLookup[s][xy]
        except:
            if debug: print(f'Failed to find source {s} in node list!')
            x = x
        try:
            y = nodeLookup[t][xy]
        except:
            if debug: print(f'Failed to find target {t} in node list!')
            y = y
        return [x, y]
    
    edgeDF['x'] = [get_xy(s,t,'x') for s,t in zip(edgeDF['source'], edgeDF['target'])]
    edgeDF['y'] = [get_xy(s,t,'y') for s,t in zip(edgeDF['source'], edgeDF['target'])]
    edges = {'x':[], 'y':[]}
    for i,e in edgeDF.iterrows():
        edges['x'].extend(e['x'])
        edges['x'].append(None)
        edges['y'].extend(e['y'])
        edges['y'].append(None)
    return nodeDF, edgeDF, edges


# In[16]:


def make_network_fig(nodeDF, edges, meta=None):
    root = nodeDF.sort_values(by='source_count').index[0]
    fig = go.Figure(layout=go.Layout(
                    title='Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.add_trace(
        go.Scatter(x=edges['x'], y=edges['y'], mode='lines',
                   line=dict(width=2, color='#888'),
                   hoverinfo='none'
                  )
    )
    fig.add_trace(
        go.Scatter(x=nodeDF['x'], y=nodeDF['y'], text=[str(m) for m in nodeDF['bb_mnems']], mode='markers',
                   marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        reversescale=False,
                        color=nodeDF['instruction_count'],
                        size=20,
                        colorbar=dict(
                            thickness=15,
                            title='Instruction Count',
                            xanchor='left',
                            titleside='right'
                        ),
                        line=dict(
                            color=['red' if c==0 else 'black' for c  in nodeDF['target_count']],
                            width=4
                        ))
                ))
    fig.update_layout(showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white'
    )
    if meta is not None:
        fig.update_layout(title=f"Function: {meta['function']}<br>IDB Path: {meta['path']}<br>Arch: {meta['arch']}")
    return fig


# In[17]:


def compare_plot(node_set, col='bb_len'):
    label = data_dict[col]
    fig = px.histogram(node_set, y=col, x='function', color=col, title=label, barmode='group')
    fig.update_xaxes(title='Function')
    fig.update_yaxes(title=label)
    fig.update_layout(template='plotly_white')
    return fig


# In[18]:


def setup(i, feature='bb_len', debug=True):
    row = simDF.iloc[i]
    networks = []
    node_sets = []
    for i in range(1,3):
        meta = {'function': row[f'fva_{i}'], 'path':row[f'idb_path_{i}']}
        meta['arch'] = meta['path'].split('/')[-1][:-4]
        if debug:
            print(meta)
            print(f"---- Looking up function {meta['function']} ----")
        binary = get_full_binary(get_feature_paths(meta['path']))
        network = extract_network(meta['function'], binary)
        nodeDF, edgeDF, edges = parse_network(network, debug=debug)
        networks.append(make_network_fig(nodeDF, edges, meta=meta))
        nodeDF['function'] = meta['function']
        node_sets.append(nodeDF)
    compare = compare_plot(pd.concat(node_sets), col=feature)
    return networks, compare


# In[19]:


map_class = {
    'pos': 'MATCH',
    'neg': 'NOT MATCH'
}


# In[20]:


def format_pair(i):
    label = f"{list(simDF['fva_1'])[i]} vs. {list(simDF['fva_2'])[i]} ({map_class[list(simDF['class'])[i]]}, {np.round(list(simDF['distance'])[i],2)})"
    return {'label': label, 'value': i}


# In[21]:


feature_options = []
for var_name, explanation in data_dict.items():
    if var_name in ['bb_len', 'instruction_count','source_count','target_count']:
        feature_options.append({'label': explanation, 'value': var_name})


# In[22]:


from jupyter_dash import JupyterDash
from dash import Dash, dash_table, html, dcc, Input, Output
import dash_daq as daq


# In[23]:


app = JupyterDash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1('Function Similarity Explorer'),
        html.P('by Jane Adams & Michael Davinroy'),
        dcc.Dropdown([format_pair(i) for i in simDF.index], 1, id='sim_index')
    ]),
    html.Div([
        html.Div([
            html.H1('Function #1'),
            dcc.Graph(id='network_1'), 
        ], style={'width': '45%', 'display': 'block', 'float':'left'}),
        html.Div([
            html.H1('Function #2'),
            dcc.Graph(id='network_2'), 
        ], style={'width': '45%', 'display': 'block', 'float':'right'})
    ], style={'width': '100%', 'display': 'block'}),
    html.Div([
        html.Div([
            html.H1('Feature Explorer'),
            dcc.Dropdown(feature_options, 'bb_len', id='feature'),
            dcc.Graph(id='importance'),
        ], style={'width': '45%', 'display': 'block', 'float':'left'}),
        html.Div([
            html.H1('Model Explorer'),
            dcc.Graph(id='model-explorer', figure=model_explorer),
            #html.Div(id='table-container')
        ], style={'width': '45%', 'display': 'block', 'float':'right'})
    ], style={'width': '100%', 'display': 'block'})
             
], style={'marginBottom': 50, 'marginTop': 25, 'marginLeft': 50, 'marginRight':50})

@app.callback(
    [Output(component_id='network_1', component_property='figure'),
     Output(component_id='network_2', component_property='figure'),
     Output(component_id='importance', component_property='figure'),
    ],
    [Input(component_id='sim_index', component_property='value'),
     Input(component_id='feature', component_property='value')
    ]
)
def update_outputs(sim_index, feature):
    
    networks, importance = setup(sim_index, debug=debug, feature=feature)
    
    network_1, network_2 = networks[0], networks[1]
    
    table = dash_table.DataTable(
        data=simDF.astype(str).to_dict('records'), # data
        sort_action='native',
        columns = [{"name": i, "id": i} for i in list(simDF.columns)], # columns
        id='table',
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_table={'overflowX': 'auto'},
        style_data_conditional=[
        {
            'if': {
                'column_id': 'index',
            },
            'backgroundColor': 'yellow',
            'color': 'black'
        }]
    )
    
    return network_1, network_2, importance

app.run_server(debug=debug)


# # By default, dashboard will be available at [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

# In[ ]:




