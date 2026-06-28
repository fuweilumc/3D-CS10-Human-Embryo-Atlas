"""
3D Cell Communication Viewer
Integrated cell-connection visualization, based on the data structure of whole_embryo_hvg_res_sub.py
Removed gene-expression visualization, enlarged the display area
"""

import dash
from dash import html, dcc, Output, Input, State, callback_context
from dash.dependencies import ALL
import plotly.graph_objects as go
import plotly.express as px
import scanpy as sc
import numpy as np
import pandas as pd
import json, os, re

# ==================== Data loading ====================
print("Loading data...")
mtx = sc.read_h5ad("nt_nmp_nc_2026_02_09.h5ad")

# Use the new column names
mtx.obs['whole_leiden_str'] = mtx.obs['leiden_hvg_1'].astype(str)
mtx.obs['sub_leiden_str'] = mtx.obs['leiden_hvg_sub'].astype(str)

# Ensure the coordinate columns exist
for col in ['x_centroid', 'y_centroid', 'z_centroid']:
    if col not in mtx.obs.columns:
        mtx.obs[col] = 0

# Check for kde_aligned coordinates
has_kde = 'x_kde_aligned' in mtx.obs.columns and 'y_kde_aligned' in mtx.obs.columns
print(f"Using coordinate system: {'kde_aligned' if has_kde else 'centroid'}")

# ==================== Load cell-connection data ====================
df_connections = None
try:
    df_connections = pd.read_csv("df.csv")
    print(f"✓ Successfully loaded connection data: {len(df_connections)} connections")
    print(f"  Columns: {df_connections.columns.tolist()}")
    
    required_conn_cols = ['from_cell', 'to_cell']
    if not all(col in df_connections.columns for col in required_conn_cols):
        print(f"⚠️ df.csv must contain 'from_cell' and 'to_cell' columns")
        df_connections = None
    else:
        if 'pair' not in df_connections.columns:
            df_connections['pair'] = 'default'
            print("  'pair' column not found, using a default pair")
        
        unique_pairs = df_connections['pair'].unique()
        print(f"  Unique pair types: {unique_pairs.tolist()}")
        
        # Check cell-ID matching
        all_from_cells = set(df_connections['from_cell'].astype(str).unique())
        all_to_cells = set(df_connections['to_cell'].astype(str).unique())
        all_df_cells = all_from_cells | all_to_cells
        adata_cells = set(mtx.obs.index.astype(str))
        
        matched_cells = all_df_cells & adata_cells
        unmatched_cells = all_df_cells - adata_cells
        
        print(f"  Total cells in df: {len(all_df_cells)}")
        print(f"  Cells matching adata: {len(matched_cells)}")
        print(f"  Unmatched cells: {len(unmatched_cells)}")
        
        if unmatched_cells:
            print(f"  Example unmatched cells: {list(unmatched_cells)[:5]}")
            print(f"  Example adata.obs.index: {list(adata_cells)[:5]}")
        
except FileNotFoundError:
    print("⚠️ df.csv not found; cell connections will not be displayed")
except Exception as e:
    print(f"⚠️ Error loading df.csv: {e}")

# ==================== Cluster configuration ====================
cluster_labels = sorted(mtx.obs['leiden_hvg_1'].astype(str).unique(), 
                       key=lambda x: int(x) if x.isdigit() else 999)
e_clusters = [cl for cl in cluster_labels]

section_labels = sorted(mtx.obs['adjusted_cure_clustering'].unique()) if 'adjusted_cure_clustering' in mtx.obs.columns else []

# Colour generation
def generate_distinct_colors(n):
    colors = []
    color_scales = [
        px.colors.qualitative.Plotly,
        px.colors.qualitative.D3,
        px.colors.qualitative.G10,
        px.colors.qualitative.T10,
        px.colors.qualitative.Alphabet,
        px.colors.qualitative.Dark24,
    ]
    for scale in color_scales:
        colors.extend(scale)
        if len(colors) >= n:
            break
    return colors[:n]

def sanitize_color(c):
    """Plotly Scatter3d does not support 8-digit hex (#rrggbbaa); truncate to 6 digits (#rrggbb)"""
    if isinstance(c, str) and c.startswith('#') and len(c) == 9:
        return c[:7]
    return c

cluster_colors_list = mtx.uns.get('leiden_hvg_1_colors', generate_distinct_colors(len(cluster_labels)))
cluster_colors_list = [sanitize_color(c) for c in cluster_colors_list]
cluster_colors = {str(c): cluster_colors_list[i] for i, c in enumerate(cluster_labels)}

# Parent-sub mapping
parent_to_sub = {}
for parent in e_clusters:
    subs = (
        mtx.obs.loc[mtx.obs['whole_leiden_str'] == parent, 'leiden_hvg_sub']
        .replace("", np.nan).dropna().unique()
    )
    parent_to_sub[parent] = sorted(map(str, subs)) if len(subs) else []

parent_map = {sub: parent for parent, subs in parent_to_sub.items() for sub in subs}
subcluster_colors = {sub: cluster_colors.get(parent_map[sub], '#808080') for sub in parent_map}

# ==================== Mesh data ====================
mesh_folder = "scaffolds"
mesh_data = {}

def _cluster_key_from_fname(fname: str):
    m = re.match(r"(.+)_mesh\.json$", fname)
    if m: return m.group(1)
    m = re.match(r"mesh_(.+)\.json$", fname)
    if m: return m.group(1)
    return None

if os.path.isdir(mesh_folder):
    for fname in os.listdir(mesh_folder):
        if not fname.endswith(".json"): continue
        key = _cluster_key_from_fname(fname)
        if key is None: continue
        if key.startswith("p_"): continue
        try:
            with open(os.path.join(mesh_folder, fname), "r") as f:
                mesh_data[key] = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load {fname}: {e}")

print(f"Loaded meshes: {sorted(mesh_data.keys())}")
print("Data loaded successfully!")

# ==================== Connection configuration ====================
DEFAULT_PAIR_COLORS = {
    'default': '#FF0000',
    'pair1': '#00FF00',
    'pair2': '#0000FF',
    'pair3': '#FFFF00',
    'pair4': '#FF00FF',
    'pair5': '#00FFFF',
}

ARROW_STYLES = {
    'cone': {'label': 'Cone', 'anchor': 'tip'},
    'arrow': {'label': 'Arrow', 'anchor': 'center'},
}

# ==================== UI Helpers ====================
def mesh_checkbox(id_dict):
    return dcc.Checklist(
        id=id_dict,
        options=[{'label': '', 'value': 'mesh'}],
        value=[],
        className="mesh-only"
    )

def control_rows(clusters, parent_to_sub, cluster_colors, subcluster_colors, mesh_data):
    blocks = []
    for parent in clusters:
        prow = html.Div([
            dcc.Checklist(
                id={'type':'cluster-check','index':parent},
                options=[{'label': f"Cluster {parent}", 'value': parent}],
                value=[]
            ),
            dcc.Input(
                id={'type':'cluster-color-input','index':parent},
                type='color',
                value=cluster_colors.get(parent,'#808080'),
                style={'width':'40px','height':'24px','padding':'0','border':'none'}
            ),
            mesh_checkbox({'type':'mesh-toggle','index':parent}) if parent in mesh_data else html.Div()
        ], className="mesh-row")

        srows = []
        for sub in parent_to_sub.get(parent, []):
            srows.append(
                html.Div([
                    dcc.Checklist(
                        id={'type':'subcluster-check','index':sub},
                        options=[{'label': sub, 'value': sub}],
                        value=[]
                    ),
                    dcc.Input(
                        id={'type':'subcluster-color-input','index':sub},
                        type='color',
                        value=subcluster_colors.get(sub, cluster_colors.get(parent,'#808080')),
                        style={'width':'36px','height':'22px','padding':'0','border':'none'}
                    ),
                    mesh_checkbox({'type':'mesh-toggle','index':sub}) if sub in mesh_data else html.Div()
                ], className="sub-row")
            )
        blocks.append(html.Div([prow, html.Div(srows, style={'paddingLeft':'20px'})]))
    return blocks

def create_connection_controls():
    """Create the connection-configuration controls"""
    if df_connections is None:
        return html.Div("No connection data loaded", style={'color': 'gray', 'fontSize': '12px'})
    
    unique_pairs = df_connections['pair'].unique()
    controls = []
    
    for pair in unique_pairs:
        count = len(df_connections[df_connections['pair'] == pair])
        default_color = DEFAULT_PAIR_COLORS.get(pair, '#FF0000')
        
        controls.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'pair-check', 'index': pair},
                    options=[{'label': f'{pair} ({count})', 'value': pair}],
                    value=[],  # unselected by default
                    style={'display': 'inline-block', 'marginRight': '10px'}
                ),
                dcc.Input(
                    id={'type': 'pair-color', 'index': pair},
                    type='color',
                    value=default_color,
                    style={'width': '40px', 'height': '24px', 'padding': '0', 
                           'border': 'none', 'marginRight': '10px'}
                ),
                html.Label('Width:', style={'fontSize': '11px', 'marginRight': '5px'}),
                dcc.Input(
                    id={'type': 'pair-width', 'index': pair},
                    type='number',
                    value=2.0,
                    min=0.5, max=10, step=0.5,
                    style={'width': '50px'}
                ),
            ], style={'marginBottom': '8px'})
        )
    
    return html.Div(controls)

# ==================== Dash App ====================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("3D Cell Communication Viewer", style={'textAlign': 'center', 'marginBottom': '10px'}),

    # Top settings bar
    html.Div([
        # Basic settings
        html.Div([
            html.Label("Marker size:", style={'fontSize': '12px', 'fontWeight': 'bold'}),
            dcc.Slider(id='size_slider', min=0.5, max=10, step=0.5, value=2,
                       marks={0.5: '0.5', 2: '2', 5: '5', 10: '10'},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Mesh Opacity:", style={'fontSize': '12px'}),
            dcc.Slider(id='mesh_opacity', min=0.05, max=1.0, step=0.05, value=0.25,
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Label("Z Zoom:", style={'fontSize': '12px'}),
            dcc.Slider(id='z_zoom_slider', min=0.25, max=2.0, step=0.05, value=1.0,
                       tooltip={"placement": "bottom", "always_visible": False}),
        ], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),

        # df-cell specific settings
        html.Div([
            html.Label("df connected-cell settings", style={'fontWeight': 'bold', 'fontSize': '13px', 'color': '#d62728'}),
            html.Div("Sender=red | Receiver=green | Both=yellow", 
                     style={'fontSize': '10px', 'color': '#666', 'marginBottom': '5px'}),
            html.Label("Connected-cell size:", style={'fontSize': '12px', 'marginTop': '5px'}),
            dcc.Slider(id='connected_cell_size_slider', min=1, max=20, step=0.5, value=5,
                       marks={1: '1', 5: '5', 10: '10', 20: '20'},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Connected-cell colour:", style={'fontSize': '12px'}),
            dcc.Input(id='connected_cell_color', type='color', value='#FF0000',
                      style={'width': '60px', 'height': '30px', 'padding': '0', 'border': 'none'}),
            html.Label("Display mode:", style={'fontSize': '12px', 'marginTop': '5px'}),
            dcc.RadioItems(
                id='df_cell_display_mode',
                options=[
                    {'label': 'Use original cluster colours', 'value': 'cluster_color'},
                    {'label': 'Use a uniform colour', 'value': 'uniform_color'},
                ],
                value='uniform_color',
                style={'fontSize': '11px'}
            ),
        ], style={'flex': '1', 'minWidth': '220px', 'padding': '0 10px', 
                  'backgroundColor': '#fff3f3', 'borderRadius': '5px', 'paddingTop': '5px', 'paddingBottom': '10px'}),

        # Section selection
        html.Div([
            html.Label("Select Sections:", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='section_selector',
                options=[{'label': str(sec), 'value': sec} for sec in section_labels],
                value=section_labels,
                inline=True,
                style={'fontSize': '11px'}
            ) if section_labels else html.Div("No section data")
        ], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),

        # Highlight settings
        html.Div([
            html.Label("Highlight Connected Cells:", style={'fontWeight': 'bold', 'fontSize': '12px'}),
            dcc.Checklist(
                id='highlight_connected_cells',
                options=[{'label': 'Enable highlighting (grey out non-connected cells)', 'value': 'highlight'}],
                value=['highlight'],
                style={'marginBottom': '8px'}
            ),
            html.Label("Highlight Subclusters:", style={'fontSize': '12px'}),
            dcc.Dropdown(
                id='highlight_subcluster_selector',
                options=[{'label': sub, 'value': sub} for sub in sorted(parent_map.keys())],
                multi=True, value=[],
                placeholder="Select subclusters",
                style={'width': '100%'}
            ),
        ], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
    ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px', 'alignItems': 'flex-start'}),

    # Main content area
    html.Div([
        # Left: cluster control panel
        html.Div([
            html.H4("Clusters", style={'margin': '0 0 8px 0'}),
            html.Div(
                control_rows(e_clusters, parent_to_sub, cluster_colors, subcluster_colors, mesh_data),
                style={'maxHeight': '85vh', 'overflowY': 'auto', 'overflowX': 'hidden'}
            )
        ], style={'width': '200px', 'padding': '10px', 'borderRight': '1px solid #ccc', 
                  'overflowY': 'auto', 'height': '90vh'}),

        # Centre: 3D plot (enlarged display area)
        html.Div([
            dcc.Graph(
                id='scatter3d_plot', 
                style={'height': '90vh', 'width': '100%'},
                config={'toImageButtonOptions': {'height': 1000, 'width': 1000, 'scale': 2}}
            ),
        ], style={'flex': '1', 'minWidth': '0'}),

        # Right: connection control panel
        html.Div([
            html.H4("Cell-connection configuration", style={'margin': '0 0 8px 0'}),
            
            # Arrow settings
            html.Div([
                html.Label("Show arrows:", style={'fontSize': '11px'}),
                dcc.Checklist(
                    id='show_arrows',
                    options=[{'label': 'Show', 'value': 'show'}],
                    value=['show'],
                    style={'display': 'inline-block', 'marginLeft': '10px'}
                ),
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label("Arrow style:", style={'fontSize': '11px'}),
                dcc.Dropdown(
                    id='arrow_style',
                    options=[{'label': v['label'], 'value': k} for k, v in ARROW_STYLES.items()],
                    value='cone',
                    clearable=False,
                    style={'width': '140px'}
                ),
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label("Arrow size:", style={'fontSize': '11px'}),
                dcc.Slider(
                    id='arrow_size_slider',
                    min=0.01, max=0.2, step=0.01, value=0.05,
                    marks={0.01: 'Small', 0.1: 'Medium', 0.2: 'Large'},
                    tooltip={"placement": "bottom"}
                ),
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label("Arrow position:", style={'fontSize': '11px'}),
                dcc.Slider(
                    id='arrow_position_slider',
                    min=0.0, max=0.5, step=0.05, value=0.1,
                    marks={0.0: 'End', 0.25: 'Medium', 0.5: 'Far'},
                    tooltip={"placement": "bottom"}
                ),
            ], style={'marginBottom': '15px'}),
            
            html.Hr(),
            html.Div(id='connection_controls', children=create_connection_controls()),
        ], style={'width': '220px', 'padding': '10px', 'borderLeft': '1px solid #ccc',
                  'overflowY': 'auto', 'height': '90vh'}),
    ], style={'display': 'flex', 'flexDirection': 'row', 'height': '90vh'}),

], style={'fontFamily': 'Arial, sans-serif', 'padding': '10px'})

# CSS
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        .mesh-row, .sub-row {
            display: grid; 
            grid-template-columns: 1fr auto auto; 
            gap: 6px; 
            align-items: center; 
            padding: 2px 0;
        }
        .mesh-only { margin: 0; padding: 0; }
        .sub-row { font-size: 0.92em; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

# ==================== Callbacks ====================

# Auto-select sub-clusters when a parent cluster is selected
@app.callback(
    Output({'type':'subcluster-check','index':ALL}, 'value'),
    [Input({'type':'cluster-check','index':ALL}, 'value')],
    [State({'type':'cluster-check','index':ALL}, 'id'),
     State({'type':'subcluster-check','index':ALL}, 'id'),
     State({'type':'subcluster-check','index':ALL}, 'value')],
    prevent_initial_call=True
)
def sync_parent_to_children(cluster_vals, cluster_ids, subcluster_ids, current_sub_vals):
    if not cluster_vals or not cluster_ids:
        return current_sub_vals
    
    ctx = callback_context
    if not ctx.triggered:
        return current_sub_vals
    
    triggered_id = ctx.triggered[0]['prop_id']
    
    try:
        id_str = triggered_id.split('.')[0]
        triggered_dict = json.loads(id_str)
        triggered_cluster = triggered_dict['index']
    except:
        return current_sub_vals
    
    triggered_cluster_selected = False
    for cid, val in zip(cluster_ids, cluster_vals):
        if cid['index'] == triggered_cluster:
            triggered_cluster_selected = bool(val)
            break
    
    new_sub_vals = []
    for sub_id, current_val in zip(subcluster_ids, current_sub_vals):
        sub_name = sub_id['index']
        parent = parent_map.get(sub_name)
        
        if parent == triggered_cluster:
            if triggered_cluster_selected:
                new_sub_vals.append([sub_name])
            else:
                new_sub_vals.append([])
        else:
            new_sub_vals.append(current_val)
    
    return new_sub_vals


def compute_cell_roles(df_conn, pair_configs):
    """Compute cell roles (without drawing connections); returns (connected_cells, sender_cells, receiver_cells, both_cells)"""
    if df_conn is None or df_conn.empty:
        return set(), set(), set(), set()
    
    connected_cells = set()
    all_senders = set()
    all_receivers = set()
    
    for pair, config in pair_configs.items():
        if not config['show']:
            continue
            
        pair_df = df_conn[df_conn['pair'] == pair]
        
        for _, row in pair_df.iterrows():
            from_cell = str(row['from_cell'])
            to_cell = str(row['to_cell'])
            
            if from_cell not in mtx.obs.index or to_cell not in mtx.obs.index:
                continue
            
            connected_cells.add(from_cell)
            connected_cells.add(to_cell)
            all_senders.add(from_cell)
            all_receivers.add(to_cell)
    
    # Classify: both sender and receiver -> both; otherwise sender-only or receiver-only
    both_cells = all_senders & all_receivers
    sender_cells = all_senders - both_cells
    receiver_cells = all_receivers - both_cells
    
    return connected_cells, sender_cells, receiver_cells, both_cells


@app.callback(
    Output('scatter3d_plot', 'figure'),
    [Input('size_slider', 'value'),
     Input('connected_cell_size_slider', 'value'),
     Input('connected_cell_color', 'value'),
     Input('df_cell_display_mode', 'value'),
     Input('mesh_opacity', 'value'),
     Input('z_zoom_slider', 'value'),
     Input('section_selector', 'value'),
     Input('highlight_connected_cells', 'value'),
     Input('highlight_subcluster_selector', 'value'),
     Input({'type': 'cluster-check', 'index': ALL}, 'value'),
     Input({'type': 'subcluster-check', 'index': ALL}, 'value'),
     Input({'type': 'mesh-toggle', 'index': ALL}, 'value'),
     Input({'type': 'cluster-color-input', 'index': ALL}, 'value'),
     Input({'type': 'subcluster-color-input', 'index': ALL}, 'value'),
     Input({'type': 'pair-check', 'index': ALL}, 'value'),
     Input({'type': 'pair-color', 'index': ALL}, 'value'),
     Input({'type': 'pair-width', 'index': ALL}, 'value'),
     Input('show_arrows', 'value'),
     Input('arrow_style', 'value'),
     Input('arrow_size_slider', 'value'),
     Input('arrow_position_slider', 'value')],
    [State({'type': 'cluster-check', 'index': ALL}, 'id'),
     State({'type': 'subcluster-check', 'index': ALL}, 'id'),
     State({'type': 'mesh-toggle', 'index': ALL}, 'id'),
     State({'type': 'cluster-color-input', 'index': ALL}, 'id'),
     State({'type': 'subcluster-color-input', 'index': ALL}, 'id'),
     State({'type': 'pair-check', 'index': ALL}, 'id')]
)
def update_plot(marker_size, connected_cell_size, connected_cell_color, df_cell_display_mode,
                mesh_alpha, z_zoom,
                selected_sections, highlight_connected_list, highlight_subs,
                cluster_vals, subcluster_vals, mesh_vals,
                cluster_color_vals, subcluster_color_vals,
                pair_checks, pair_colors, pair_widths,
                show_arrows_list, arrow_style, arrow_size, arrow_position,
                cluster_ids, subcluster_ids, mesh_ids,
                cluster_color_ids, subcluster_color_ids, pair_ids):
    
    fig = go.Figure()
    
    # Parse the selection
    selected_clusters = {cid['index'] for cid, val in zip(cluster_ids, cluster_vals) if val}
    selected_subs = {sid['index'] for sid, val in zip(subcluster_ids, subcluster_vals) if val}
    highlight_subs_set = set(highlight_subs) if highlight_subs else set()
    mesh_enabled = {mid['index'] for mid, val in zip(mesh_ids, mesh_vals) if val}
    
    # Colour mapping (sanitize to prevent 8-digit hex)
    live_cluster_colors = {cid['index']: sanitize_color(col) for cid, col in zip(cluster_color_ids, cluster_color_vals)}
    live_subcluster_colors = {sid['index']: sanitize_color(col) for sid, col in zip(subcluster_color_ids, subcluster_color_vals)}
    
    # Section filtering
    if section_labels and selected_sections:
        section_mask = mtx.obs['adjusted_cure_clustering'].isin(selected_sections)
    else:
        section_mask = np.ones(mtx.n_obs, dtype=bool)
    
    scale_y = lambda arr: np.array(arr) * z_zoom if hasattr(arr, '__iter__') else arr * z_zoom
    
    # Parse the connection configuration
    pair_configs = {}
    if df_connections is not None and pair_ids:
        for check_list, color, width, id_dict in zip(pair_checks, pair_colors, pair_widths, pair_ids):
            pair = id_dict['index']
            pair_configs[pair] = {
                'show': pair in check_list if check_list else False,
                'color': color,
                'width': width
            }
    
    # Compute cell roles (without drawing connections)
    connected_cells, sender_cells, receiver_cells, both_cells = compute_cell_roles(
        df_connections, pair_configs
    )
    
    highlight_mode = 'highlight' in highlight_connected_list
    use_uniform_color = df_cell_display_mode == 'uniform_color'
    print(f"Connected cells: {len(connected_cells)} (sender: {len(sender_cells)}, receiver: {len(receiver_cells)}, both: {len(both_cells)}), highlight mode: {highlight_mode}, uniform colour: {use_uniform_color}")
    
    # Cell-role colour mapping: sender=red, receiver=green, both=yellow
    ROLE_COLORS = {'sender': '#FF0000', 'receiver': '#00FF00', 'both': '#FFFF00'}
    
    # Draw cells
    traces_normal = []
    traces_highlight = []
    
    # If no cluster is selected but there are connected cells, show all cells (grey background + role-coloured connected cells)
    if not selected_clusters and connected_cells:
        # First draw all cells as a grey background
        all_cell_ids = mtx.obs.loc[section_mask].index.astype(str).to_numpy()
        is_connected_all = np.array([cid in connected_cells for cid in all_cell_ids])
        
        if has_kde:
            x_all = mtx.obs.loc[section_mask, 'x_kde_aligned'].to_numpy()
            y_all = scale_y(mtx.obs.loc[section_mask, 'z_centroid'].to_numpy())
            z_all = mtx.obs.loc[section_mask, 'y_kde_aligned'].to_numpy()
        else:
            x_all = mtx.obs.loc[section_mask, 'x_centroid'].to_numpy()
            y_all = scale_y(mtx.obs.loc[section_mask, 'z_centroid'].to_numpy())
            z_all = mtx.obs.loc[section_mask, 'y_centroid'].to_numpy()
        
        # Non-connected cells -> grey
        if np.any(~is_connected_all):
            traces_normal.append(go.Scatter3d(
                x=x_all[~is_connected_all],
                y=y_all[~is_connected_all],
                z=z_all[~is_connected_all],
                mode='markers',
                marker=dict(size=marker_size, color='#000000', opacity=0.3),
                name='Other Cells',
                showlegend=False
            ))
        
        # Draw connected cells grouped by role
        if np.any(is_connected_all):
            conn_ids = all_cell_ids[is_connected_all]
            conn_x = x_all[is_connected_all]
            conn_y = y_all[is_connected_all]
            conn_z = z_all[is_connected_all]
            
            is_sender = np.array([cid in sender_cells for cid in conn_ids])
            is_receiver = np.array([cid in receiver_cells for cid in conn_ids])
            is_both = np.array([cid in both_cells for cid in conn_ids])
            
            for role_mask, role_color, role_name in [
                (is_sender, ROLE_COLORS['sender'], 'Sender Cells'),
                (is_receiver, ROLE_COLORS['receiver'], 'Receiver Cells'),
                (is_both, ROLE_COLORS['both'], 'Sender & Receiver Cells')
            ]:
                if np.any(role_mask):
                    traces_highlight.append(go.Scatter3d(
                        x=conn_x[role_mask],
                        y=conn_y[role_mask],
                        z=conn_z[role_mask],
                        mode='markers',
                        marker=dict(size=connected_cell_size, color=role_color, opacity=0.9,
                                   line=dict(width=0.5, color='white')),
                        name=role_name
                    ))
    
    for cluster in selected_clusters:
        mask_cells = section_mask & (mtx.obs['whole_leiden_str'] == cluster)
        if not mask_cells.any():
            continue
        
        # Get coordinates
        if has_kde:
            x_arr = mtx.obs.loc[mask_cells, 'x_kde_aligned'].to_numpy()
            y_arr = scale_y(mtx.obs.loc[mask_cells, 'z_centroid'].to_numpy())
            z_arr = mtx.obs.loc[mask_cells, 'y_kde_aligned'].to_numpy()
        else:
            x_arr = mtx.obs.loc[mask_cells, 'x_centroid'].to_numpy()
            y_arr = scale_y(mtx.obs.loc[mask_cells, 'z_centroid'].to_numpy())
            z_arr = mtx.obs.loc[mask_cells, 'y_centroid'].to_numpy()
        
        all_child_subs = parent_to_sub.get(cluster, [])
        child_subs = [s for s in all_child_subs if s in selected_subs]
        
        if child_subs:
            for sub in child_subs:
                sub_mask = mask_cells & (mtx.obs['sub_leiden_str'] == sub)
                if not sub_mask.any():
                    continue
                
                cell_ids = mtx.obs.loc[sub_mask].index.astype(str).to_numpy()
                
                if has_kde:
                    x_sub = mtx.obs.loc[sub_mask, 'x_kde_aligned'].to_numpy()
                    y_sub = scale_y(mtx.obs.loc[sub_mask, 'z_centroid'].to_numpy())
                    z_sub = mtx.obs.loc[sub_mask, 'y_kde_aligned'].to_numpy()
                else:
                    x_sub = mtx.obs.loc[sub_mask, 'x_centroid'].to_numpy()
                    y_sub = scale_y(mtx.obs.loc[sub_mask, 'z_centroid'].to_numpy())
                    z_sub = mtx.obs.loc[sub_mask, 'y_centroid'].to_numpy()
                
                color = live_subcluster_colors.get(sub, live_cluster_colors.get(cluster, '#808080'))
                is_highlight_sub = sub in highlight_subs_set
                
                if highlight_mode and connected_cells:
                    is_connected = np.array([cid in connected_cells for cid in cell_ids])
                    
                    # Non-connected cells (grey, small, low opacity)
                    if np.any(~is_connected):
                        traces_normal.append(go.Scatter3d(
                            x=x_sub[~is_connected],
                            y=y_sub[~is_connected],
                            z=z_sub[~is_connected],
                            mode='markers',
                            marker=dict(size=marker_size, color='#000000', opacity=0.3),
                            name=f'{sub} (other)',
                            showlegend=False
                        ))
                    
                    # Connected cells: coloured by role sender=red, receiver=green, both=yellow
                    if np.any(is_connected):
                        if use_uniform_color:
                            connected_ids = cell_ids[is_connected]
                            connected_x = x_sub[is_connected]
                            connected_y = y_sub[is_connected]
                            connected_z = z_sub[is_connected]
                            
                            # Create a mask for each of the three roles
                            is_sender = np.array([cid in sender_cells for cid in connected_ids])
                            is_receiver = np.array([cid in receiver_cells for cid in connected_ids])
                            is_both = np.array([cid in both_cells for cid in connected_ids])
                            
                            for role_mask, role_color, role_name in [
                                (is_sender, ROLE_COLORS['sender'], 'Sender'),
                                (is_receiver, ROLE_COLORS['receiver'], 'Receiver'),
                                (is_both, ROLE_COLORS['both'], 'Both')
                            ]:
                                if np.any(role_mask):
                                    traces_highlight.append(go.Scatter3d(
                                        x=connected_x[role_mask],
                                        y=connected_y[role_mask],
                                        z=connected_z[role_mask],
                                        mode='markers',
                                        marker=dict(size=connected_cell_size, color=role_color, opacity=0.9,
                                                   line=dict(width=0.5, color='white')),
                                        name=f'{sub} ({role_name})'
                                    ))
                        else:
                            traces_highlight.append(go.Scatter3d(
                                x=x_sub[is_connected],
                                y=y_sub[is_connected],
                                z=z_sub[is_connected],
                                mode='markers',
                                marker=dict(size=connected_cell_size, color=color, opacity=0.9,
                                           line=dict(width=0.5, color='white')),
                                name=f'{sub} (connected)'
                            ))
                else:
                    # Normal mode
                    size = connected_cell_size if is_highlight_sub else marker_size
                    opacity = 0.95 if is_highlight_sub else 0.85
                    
                    trace = go.Scatter3d(
                        x=x_sub, y=y_sub, z=z_sub,
                        mode='markers',
                        marker=dict(size=size, color=color, opacity=opacity,
                                   line=dict(width=0.5, color='white') if is_highlight_sub else dict(width=0)),
                        name=f'{sub}' + (' [highlighted]' if is_highlight_sub else '')
                    )
                    
                    if is_highlight_sub:
                        traces_highlight.append(trace)
                    else:
                        traces_normal.append(trace)
                        
        elif not all_child_subs:
            # Case with no sub-clusters
            cell_ids = mtx.obs.loc[mask_cells].index.astype(str).to_numpy()
            color = live_cluster_colors.get(cluster, '#808080')
            
            if highlight_mode and connected_cells:
                is_connected = np.array([cid in connected_cells for cid in cell_ids])
                
                if np.any(~is_connected):
                    traces_normal.append(go.Scatter3d(
                        x=x_arr[~is_connected],
                        y=y_arr[~is_connected],
                        z=z_arr[~is_connected],
                        mode='markers',
                        marker=dict(size=marker_size, color='#000000', opacity=0.3),
                        name=f'Cluster {cluster} (other)',
                        showlegend=False
                    ))
                
                if np.any(is_connected):
                    if use_uniform_color:
                        connected_ids = cell_ids[is_connected]
                        connected_x = x_arr[is_connected]
                        connected_y = y_arr[is_connected]
                        connected_z = z_arr[is_connected]
                        
                        is_sender = np.array([cid in sender_cells for cid in connected_ids])
                        is_receiver = np.array([cid in receiver_cells for cid in connected_ids])
                        is_both = np.array([cid in both_cells for cid in connected_ids])
                        
                        for role_mask, role_color, role_name in [
                            (is_sender, ROLE_COLORS['sender'], 'Sender'),
                            (is_receiver, ROLE_COLORS['receiver'], 'Receiver'),
                            (is_both, ROLE_COLORS['both'], 'Both')
                        ]:
                            if np.any(role_mask):
                                traces_highlight.append(go.Scatter3d(
                                    x=connected_x[role_mask],
                                    y=connected_y[role_mask],
                                    z=connected_z[role_mask],
                                    mode='markers',
                                    marker=dict(size=connected_cell_size, color=role_color, opacity=0.9,
                                               line=dict(width=0.5, color='white')),
                                    name=f'Cluster {cluster} ({role_name})'
                                ))
                    else:
                        traces_highlight.append(go.Scatter3d(
                            x=x_arr[is_connected],
                            y=y_arr[is_connected],
                            z=z_arr[is_connected],
                            mode='markers',
                            marker=dict(size=connected_cell_size, color=color, opacity=0.9,
                                       line=dict(width=0.5, color='white')),
                            name=f'Cluster {cluster} (connected)'
                        ))
            else:
                traces_normal.append(go.Scatter3d(
                    x=x_arr, y=y_arr, z=z_arr,
                    mode='markers',
                    marker=dict(size=marker_size, color=color, opacity=0.85),
                    name=f'Cluster {cluster}'
                ))
        
        # Add mesh
        if cluster in mesh_enabled and cluster in mesh_data:
            mdata = mesh_data[cluster]
            if 'vertices' in mdata and 'faces' in mdata:
                verts = np.array(mdata['vertices'])
                faces = np.array(mdata['faces'])
                mesh_x = verts[:, 0]
                mesh_y = scale_y(verts[:, 2])
                mesh_z = verts[:, 1]
                
                traces_normal.append(go.Mesh3d(
                    x=mesh_x, y=mesh_y, z=mesh_z,
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color=live_cluster_colors.get(cluster, '#808080'),
                    opacity=mesh_alpha,
                    name=f'Mesh {cluster}'
                ))
    
    # Subcluster meshes
    for sub in selected_subs:
        if sub in mesh_enabled and sub in mesh_data:
            parent = parent_map.get(sub)
            mdata = mesh_data[sub]
            if 'vertices' in mdata and 'faces' in mdata:
                verts = np.array(mdata['vertices'])
                faces = np.array(mdata['faces'])
                mesh_x = verts[:, 0]
                mesh_y = scale_y(verts[:, 2])
                mesh_z = verts[:, 1]
                
                traces_normal.append(go.Mesh3d(
                    x=mesh_x, y=mesh_y, z=mesh_z,
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color=live_subcluster_colors.get(sub, live_cluster_colors.get(parent, '#808080')),
                    opacity=mesh_alpha,
                    name=f'Mesh {sub}'
                ))
    
    # Add normal traces first, then highlighted traces
    for trace in traces_normal:
        fig.add_trace(trace)
    for trace in traces_highlight:
        fig.add_trace(trace)
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='', showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(title='', showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(title='', showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        uirevision='main'
    )
    
    return fig


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8051)