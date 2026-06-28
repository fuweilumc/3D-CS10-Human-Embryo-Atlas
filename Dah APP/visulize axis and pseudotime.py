import dash
from dash import html, dcc, Output, Input, State
from dash.dependencies import ALL
import plotly.graph_objects as go
import plotly.express as px
import scanpy as sc
import numpy as np
import json, os, re

# ---------- load data ----------
mtx = sc.read_h5ad("nt_nmp_nc_2026_02_09.h5ad")
mtx.obs['whole_leiden_str'] = mtx.obs['leiden_hvg_1'].astype(str)
mtx.obs['sub_leiden_str'] = mtx.obs['leiden_hvg_sub'].astype(str)

# Check for and add AP/LR/DV coordinates
required_cols = ['AP', 'LR', 'DV']
missing_cols = [col for col in required_cols if col not in mtx.obs.columns]
if missing_cols:
    print(f"⚠️ Missing columns: {missing_cols}")
    if 'AP' not in mtx.obs.columns and 'new_x_arc_length' in mtx.obs.columns:
        mtx.obs['AP'] = mtx.obs['new_x_arc_length']
    if 'LR' not in mtx.obs.columns and 'new_y_dist_from_surface' in mtx.obs.columns:
        mtx.obs['LR'] = mtx.obs['new_y_dist_from_surface']
    if 'DV' not in mtx.obs.columns and 'new_z_dist_from_center' in mtx.obs.columns:
        mtx.obs['DV'] = mtx.obs['new_z_dist_from_center']

# If still missing, fall back to the original coordinates
for col in required_cols:
    if col not in mtx.obs.columns:
        if col == 'AP' and 'x_kde_aligned' in mtx.obs.columns:
            mtx.obs['AP'] = mtx.obs['x_kde_aligned']
        elif col == 'LR' and 'y_kde_aligned' in mtx.obs.columns:
            mtx.obs['LR'] = mtx.obs['y_kde_aligned']
        elif col == 'DV' and 'z_centroid' in mtx.obs.columns:
            mtx.obs['DV'] = mtx.obs['z_centroid']
        else:
            mtx.obs[col] = 0

print(f"Coordinate ranges:")
print(f"  AP: {mtx.obs['AP'].min():.2f} to {mtx.obs['AP'].max():.2f}")
print(f"  LR: {mtx.obs['LR'].min():.2f} to {mtx.obs['LR'].max():.2f}")
print(f"  DV: {mtx.obs['DV'].min():.2f} to {mtx.obs['DV'].max():.2f}")

# leiden_hvg_1 is purely numeric, not prefixed with e_ or p_
cluster_labels = sorted(mtx.obs['leiden_hvg_1'].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else float('inf'))
# leiden_hvg_sub is prefixed with e (e.g. e0_0, e12_1)
e_clusters = cluster_labels  # all are embryo clusters
p_clusters = []  # no placenta clusters

section_labels = sorted(mtx.obs['adjusted_cure_clustering'].unique()) if 'adjusted_cure_clustering' in mtx.obs.columns else []

cluster_colors_list = mtx.uns.get('leiden_hvg_1_colors', px.colors.qualitative.Plotly * (len(cluster_labels) // 10 + 1))
cluster_colors = {str(c): cluster_colors_list[i] for i, c in enumerate(cluster_labels)}

gene_list = mtx.var_names.tolist() if hasattr(mtx, 'var_names') else []

# Find the sub-clusters for each parent cluster
parent_to_sub = {}
for parent in e_clusters:
    subs = (
        mtx.obs.loc[mtx.obs['whole_leiden_str'] == parent, 'leiden_hvg_sub']
        .replace("", np.nan).dropna().unique()
    )
    parent_to_sub[parent] = sorted(map(str, subs)) if len(subs) else []

parent_map = {sub: parent for parent, subs in parent_to_sub.items() for sub in subs}

# Custom colour mapping for leiden_hvg_sub
sub_cluster_palette = {
    # --- Cluster 0 subtypes (red/blue/yellow/green/purple/cyan, high contrast) ---
    "e0_0": "#ff002a",  # bright red
    "e0_1": "#000080",  # dark blue
    "e0_2": "#e5e022",  # bright yellow
    "e0_3": "#009203",  # dark green
    "e0_4": "#a8329e",  # purple
    "e0_5": "#00FFFF",  # cyan

    # --- Cluster 12 subtypes (orange/deep-purple/pink/brown) ---
    "e12_0": "#FF8C00", # orange
    "e12_1": "#565DFD", # blue-purple
    "e12_2": "#FF00FF", # deep pink
    "e12_3": "#8B4513", # brown

    # --- Cluster 20 subtypes (bright-green/dark-red/teal) ---
    "e20_0": "#66c102", # bright green
    "e20_1": "#8e0119", # dark red
    "e20_2": "#10686f", # teal

    # --- Cluster 25 subtypes (gold/grey/magenta) ---
    "e25_0": "#FF1493", # gold
    "e25_1": "#9f50f9", # grey
    "e25_2": "#32a88e", # magenta
    "0": "#FF1493",
    "1": "#9f50f9",
    "2": "#32a88e",
}

# Use sub_cluster_palette directly; fall back to the parent cluster's colour if absent
subcluster_colors = {
    sub: sub_cluster_palette.get(sub, cluster_colors.get(parent_map.get(sub, ''), '#808080')) 
    for sub in parent_map
}

# ---------- include scaffolds ----------
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
        try:
            with open(os.path.join(mesh_folder, fname), "r") as f:
                mesh_data[key] = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load {fname}: {e}")
print("Loaded meshes for:", sorted(mesh_data.keys()))

# ---------- Section-plane configuration ----------
SLICE_POSITIONS = [280, 480, 840, 2100, 2600]
DEFAULT_COLORS = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']

# ---------- UI helpers ----------
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
                options=[{'label': parent, 'value': parent}],
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

# ---------- Dash app ----------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("3D Cluster Viewer with Slicing Planes (AP-LR-DV)", style={'textAlign': 'center'}),

    # --------- Settings (top row) ----------
    html.Div([
        html.Div([
            html.Label("Gamma (GE Color Intensity):", style={'fontSize': '12px'}),
            dcc.Slider(id='gamma_slider', min=0.1, max=2.0, step=0.1, value=1.0,
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Label("Marker Size:", style={'fontSize': '12px'}),
            dcc.Slider(id='size_slider', min=0.1, max=5, step=0.1, value=0.3,
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Label("Mesh Opacity:", style={'fontSize': '12px'}),
            dcc.Slider(id='mesh_opacity', min=0.05, max=1.0, step=0.05, value=0.25,
                       tooltip={"placement": "bottom", "always_visible": False}),
        ], style={'flex': '1', 'minWidth': '220px'}),

         html.Div([
            html.Label("Select Section to Display:") if section_labels else html.Div(),
            dcc.Checklist(
                id='section_selector',
                options=[{'label': str(sec), 'value': sec} for sec in section_labels],
                value=section_labels, inline=True,
                style={'margin-bottom': '10px'}
            ) if section_labels else html.Div(),
            html.Label("Select up to 4 Genes (for RGBC coloring):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='gene_selector',
                options=[{'label': g, 'value': g} for g in gene_list],
                multi=True, value=[],
                placeholder="Pick genes to color cells",
                style={'width': '100%'}
            ),
            # ========== Export instructions ==========
            html.Div([
                html.Hr(style={'margin': '15px 0 10px 0'}),
                html.Strong("Export the current view:", style={'fontSize': '13px', 'color': '#2c3e50'}),
                html.Ol([
                    html.Li("Rotate the plot to the desired angle", style={'fontSize': '11px', 'margin': '3px 0'}),
                    html.Li("Press F12 -> Console tab", style={'fontSize': '11px', 'margin': '3px 0'}),
                    html.Li("Type: allow pasting (press Enter)", style={'fontSize': '11px', 'margin': '3px 0'}),
                    html.Li("Copy-paste the code below and press Enter:", style={'fontSize': '11px', 'margin': '3px 0'}),
                ], style={'paddingLeft': '20px', 'margin': '5px 0'}),
                html.Div([
                    html.Code(
                        "var gd=document.getElementById('cluster_3d_plot');var r=gd.getBoundingClientRect();Plotly.toImage(gd,{format:'png',width:Math.round(r.width*5),height:Math.round(r.height*5)}).then(function(u){var a=document.createElement('a');a.download='cluster_highres.png';a.href=u;a.click()});",
                        style={'display': 'block', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'fontSize': '9px', 'wordBreak': 'break-all', 'border': '1px solid #dee2e6', 'borderRadius': '4px', 'fontFamily': 'monospace', 'lineHeight': '1.4'}
                    ),
                    html.Div("Right-hand plot: change 'cluster_3d_plot' to 'gene_expression_plot'", style={'fontSize': '10px', 'color': '#6c757d', 'marginTop': '5px', 'fontStyle': 'italic'})
                ])
            ], style={'backgroundColor': '#e7f3ff', 'padding': '12px', 'borderRadius': '6px', 'border': '1px solid #b3d9ff', 'marginTop': '10px'})
            # ==========================================
        ], style={'flex': '1', 'minWidth': '280px', 'padding':'0 20px'})
    ], style={'display': 'flex', 'gap':'20px', 'margin-bottom': '16px', 'alignItems': 'flex-start'}),
    
    # --------- Section-plane control panel ----------
    html.Div([
        html.H3("Section-plane control (perpendicular to the AP axis, expanding from LR=0 to both sides)", style={'textAlign': 'center', 'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.Div([
                    html.Strong(f"Section plane {i+1}: AP = {pos}", style={'fontSize': '14px'}),
                    html.Br(),
                    dcc.Checklist(
                        id=f'slice_toggle_{i}',
                        options=[{'label': ' Show section plane', 'value': 'show'}],
                        value=[],
                        style={'marginTop': '5px'}
                    ),
                    html.Label("Colour:", style={'fontSize': '12px', 'marginTop': '8px'}),
                    dcc.Input(
                        id=f'slice_color_{i}',
                        type='text',
                        value=DEFAULT_COLORS[i],
                        placeholder='e.g. #FF5733',
                        style={'width': '100%', 'fontSize': '12px', 'padding': '4px'}
                    ),
                    html.Label("LR-axis width (from 0 to both sides):", style={'fontSize': '12px', 'marginTop': '8px'}),
                    dcc.Slider(
                        id=f'slice_width_{i}',
                        min=0,
                        max=1000,
                        step=10,
                        value=400,
                        marks={0: '0', 250: '250', 500: '500', 750: '750', 1000: '1000'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Label("Opacity:", style={'fontSize': '12px', 'marginTop': '8px'}),
                    dcc.Slider(
                        id=f'slice_opacity_{i}',
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        value=0.3,
                        marks={0.1: '0.1', 0.5: '0.5', 1.0: '1.0'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'borderRadius': '8px',
                    'backgroundColor': '#fafafa'
                })
            ], style={'flex': '1', 'margin': '5px'})
            for i, pos in enumerate(SLICE_POSITIONS)
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'})
    ], style={'backgroundColor': '#f8f8f8', 'padding': '15px', 'marginBottom': '20px', 'borderRadius': '10px'}),

    # --------- Main content ----------
    html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='cluster_3d_plot', style={'height': '600px', 'width': '100%'},
                          config={'toImageButtonOptions': {'height': 1080, 'width': 1920, 'scale': 2}})
            ], style={'flex': '1', 'minWidth':'0', 'marginRight':'8px'}),
            html.Div([
                dcc.Graph(id='gene_expression_plot', style={'height': '600px', 'width': '100%'},
                          config={'toImageButtonOptions': {'height': 1080, 'width': 1920, 'scale': 2}}),
                html.Div(id='rgb_legend', style={'textAlign': 'center', 'paddingTop': '6px'})
            ], style={'flex': '1', 'minWidth':'0', 'marginLeft':'8px'}),
        ], style={'flex': '3', 'display': 'flex', 'flexDirection': 'row', 'minWidth':'0'}),
        html.Div([
            # Embryo column
            html.Div([
                html.Div([
                    html.Div("Clusters",   style={'fontWeight':'bold'}),
                    html.Div("Surface",  style={'textAlign':'center','fontWeight':'bold'}),
                ], style={
                    'display':'grid',
                    'gridTemplateColumns':'auto 34px 18px',
                    'columnGap':'2px',
                    'marginBottom':'4px'
                }),
                dcc.Checklist(
                    id='toggle_all_e',
                    options=[{'label':'Toggle All','value':'all'}],
                    value=[],
                    style={'marginBottom':'6px'}
                ),
                html.Div(control_rows(e_clusters, parent_to_sub, cluster_colors, subcluster_colors, mesh_data))
            ], style={'flex':'1', 'minWidth':'180px'}),

        ], style={'flex':'1', 'display':'flex', 'flexDirection':'row',
                  'maxHeight':'600px', 'overflowY':'auto', 'paddingLeft':'4px'})

    ], style={'display': 'flex', 'gap':'4px', 'alignItems': 'flex-start'}),

    dcc.Store(id='cluster_selector', data=cluster_labels),
    dcc.Store(id='cluster_colors_store', data=cluster_colors),
    dcc.Store(id='subcluster_colors_store', data=subcluster_colors)
])

# --------- Callbacks ----------

@app.callback(
    Output('cluster_selector', 'data'),
    Input({'type': 'cluster-check', 'index': ALL}, 'value'),
)
def sync_selected_clusters(values):
    selected_clusters = [val for v in (values or []) for val in (v or [])]
    return selected_clusters

@app.callback(
    Output({'type': 'cluster-check', 'index': ALL}, 'value'),
    Output('toggle_all_e', 'value'),
    Input('toggle_all_e', 'value'),
    State({'type': 'cluster-check', 'index': ALL}, 'id')
)
def toggle_all(e_toggle, ids):
    e_on = 'all' in (e_toggle or [])
    out = []
    for cid in ids:
        idx = cid['index']
        if idx in e_clusters:
            out.append([idx] if e_on else [])
        else:
            out.append([])
    return out, [] if e_on else e_toggle

@app.callback(
    Output('cluster_3d_plot', 'figure'),
    Output('gene_expression_plot', 'figure'),
    Output('rgb_legend', 'children'),
    Input('gamma_slider', 'value'),
    Input('size_slider', 'value'),
    Input('mesh_opacity', 'value'),
    Input('section_selector', 'value'),
    Input('gene_selector', 'value'),
    Input({'type': 'cluster-check', 'index': ALL}, 'value'),
    Input({'type': 'cluster-color-input', 'index': ALL}, 'value'),
    Input({'type': 'subcluster-check', 'index': ALL}, 'value'),
    Input({'type': 'subcluster-color-input', 'index': ALL}, 'value'),
    Input({'type': 'mesh-toggle', 'index': ALL}, 'value'),
    *[Input(f'slice_toggle_{i}', 'value') for i in range(len(SLICE_POSITIONS))],
    *[Input(f'slice_color_{i}', 'value') for i in range(len(SLICE_POSITIONS))],
    *[Input(f'slice_width_{i}', 'value') for i in range(len(SLICE_POSITIONS))],
    *[Input(f'slice_opacity_{i}', 'value') for i in range(len(SLICE_POSITIONS))],
    State({'type': 'cluster-check', 'index': ALL}, 'id'),
    State({'type': 'subcluster-check', 'index': ALL}, 'id'),
    State({'type': 'mesh-toggle', 'index': ALL}, 'id'),
)
def update_plots(
    gamma, marker_size, mesh_opacity_val, selected_sections, selected_genes,
    cluster_vals, cluster_cols, subcluster_vals, subcluster_cols, mesh_vals,
    *slice_args,
    cluster_ids, subcluster_ids, mesh_ids
):
    # Parse section-plane parameters
    n_slices = len(SLICE_POSITIONS)
    slice_toggles = slice_args[:n_slices]
    slice_colors = slice_args[n_slices:2*n_slices]
    slice_widths = slice_args[2*n_slices:3*n_slices]
    slice_opacities = slice_args[3*n_slices:]

    # Determine the selected clusters and sub-clusters
    selected_parent = {cid['index'] for cid, vals in zip(cluster_ids, cluster_vals) if cid['index'] in vals}
    selected_subs = {sid['index'] for sid, vals in zip(subcluster_ids, subcluster_vals) if sid['index'] in vals}

    # Colour mapping
    cluster_color_map = {cid['index']: col for cid, col in zip(cluster_ids, cluster_cols)}
    subcluster_color_map = {sid['index']: col for sid, col in zip(subcluster_ids, subcluster_cols)}

    # Mesh display
    mesh_show = {mid['index']: ('mesh' in vals) for mid, vals in zip(mesh_ids, mesh_vals)}

    # Filter sections (if any)
    if section_labels and selected_sections:
        section_mask = mtx.obs['adjusted_cure_clustering'].isin(selected_sections)
    else:
        section_mask = pd.Series([True] * mtx.n_obs, index=mtx.obs.index)

    # Compute the DV range for the plane size
    dv_min = mtx.obs['DV'].min()
    dv_max = mtx.obs['DV'].max()

    def add_slice_plane(fig, ap_pos, color, width, opacity):
        lr_min, lr_max = -width, width
        dv_min_val, dv_max_val = dv_min, dv_max
        
        # Create the grid
        lr_grid = np.linspace(lr_min, lr_max, 2)
        dv_grid = np.linspace(dv_min_val, dv_max_val, 2)
        LR, DV = np.meshgrid(lr_grid, dv_grid)
        AP = np.full_like(LR, ap_pos)
        
        fig.add_trace(go.Surface(
            x=AP, y=LR, z=DV,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=opacity,
            name=f'Slice AP={ap_pos}',
            hoverinfo='skip'
        ))

    # ---------- Figure 1: Cluster plot ----------
    fig1 = go.Figure()

    for parent in selected_parent:
        parent_subs = parent_to_sub.get(parent, [])
        child_subs = [s for s in parent_subs if s in selected_subs]
        all_child_subs = (len(child_subs) == len(parent_subs)) if parent_subs else False

        mask_cells = (mtx.obs['whole_leiden_str'] == parent) & section_mask
        if mask_cells.sum() == 0:
            continue

        x_arr = mtx.obs.loc[mask_cells, 'AP'].to_numpy()
        y_arr = mtx.obs.loc[mask_cells, 'LR'].to_numpy()
        z_arr = mtx.obs.loc[mask_cells, 'DV'].to_numpy()

        if child_subs:
            for sub in child_subs:
                sub_mask = mask_cells & (mtx.obs['sub_leiden_str'] == sub)
                x_sub = mtx.obs.loc[sub_mask, 'AP'].to_numpy()
                y_sub = mtx.obs.loc[sub_mask, 'LR'].to_numpy()
                z_sub = mtx.obs.loc[sub_mask, 'DV'].to_numpy()
                if x_sub.size == 0:
                    continue
                sub_col = subcluster_color_map.get(sub, cluster_color_map.get(parent, '#808080'))
                fig1.add_trace(go.Scatter3d(
                    x=x_sub, y=y_sub, z=z_sub,
                    mode='markers',
                    marker=dict(size=marker_size, color=sub_col, opacity=0.8),
                    name=sub
                ))
        elif not all_child_subs:
            pcol = cluster_color_map.get(parent, '#808080')
            fig1.add_trace(go.Scatter3d(
                x=x_arr, y=y_arr, z=z_arr,
                mode='markers',
                marker=dict(size=marker_size, color=pcol, opacity=0.8),
                name=parent
            ))

        # Mesh
        if mesh_show.get(parent, False) and parent in mesh_data:
            mdata = mesh_data[parent]
            fig1.add_trace(go.Mesh3d(
                x=mdata['x'], y=mdata['y'], z=mdata['z'],
                i=mdata['i'], j=mdata['j'], k=mdata['k'],
                color=cluster_color_map.get(parent, '#808080'),
                opacity=mesh_opacity_val,
                name=f"{parent}_mesh"
            ))

        for sub in child_subs:
            if mesh_show.get(sub, False) and sub in mesh_data:
                mdata = mesh_data[sub]
                fig1.add_trace(go.Mesh3d(
                    x=mdata['x'], y=mdata['y'], z=mdata['z'],
                    i=mdata['i'], j=mdata['j'], k=mdata['k'],
                    color=subcluster_color_map.get(sub, cluster_color_map.get(parent, '#808080')),
                    opacity=mesh_opacity_val,
                    name=f"{sub}_mesh"
                ))

    # Add section planes to plot 1
    for i, (toggle, color, width, opacity) in enumerate(zip(slice_toggles, slice_colors, slice_widths, slice_opacities)):
        if 'show' in toggle:
            add_slice_plane(fig1, SLICE_POSITIONS[i], color, width, opacity)

    fig1.update_layout(
        scene=dict(
            xaxis=dict(title='AP', showbackground=True),
            yaxis=dict(title='LR', showbackground=True),
            zaxis=dict(title='DV', showbackground=True),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False,
        uirevision='clusters'
    )

    # ---------- Figure 2: Gene expression plot ----------
    fig2 = go.Figure()

    def _gene_vec(adata, gene):
        if gene in adata.var_names:
            idx = list(adata.var_names).index(gene)
            if hasattr(adata.X, 'toarray'):
                return adata.X[:, idx].toarray().flatten()
            else:
                return adata.X[:, idx].flatten()
        return np.zeros(adata.n_obs, dtype=float)

    if not selected_genes:
        selected_genes = []

    # Draw gene expression
    for parent in selected_parent:
        parent_subs = parent_to_sub.get(parent, [])
        child_subs = [s for s in parent_subs if s in selected_subs]
        all_child_subs = (len(child_subs) == len(parent_subs)) if parent_subs else False

        mask_cells = (mtx.obs['whole_leiden_str'] == parent) & section_mask
        if mask_cells.sum() == 0:
            continue

        x_arr = mtx.obs.loc[mask_cells, 'AP'].to_numpy()
        y_arr = mtx.obs.loc[mask_cells, 'LR'].to_numpy()
        z_arr = mtx.obs.loc[mask_cells, 'DV'].to_numpy()

        # Single gene -> Viridis colormap
        if len(selected_genes) == 1:
            gene = selected_genes[0]
            if gene in mtx.var_names:
                gv_full = _gene_vec(mtx, gene)
                if child_subs:
                    for sub in child_subs:
                        sub_mask = mask_cells & (mtx.obs['sub_leiden_str'] == sub)
                        x_sub = mtx.obs.loc[sub_mask, 'AP'].to_numpy()
                        y_sub = mtx.obs.loc[sub_mask, 'LR'].to_numpy()
                        z_sub = mtx.obs.loc[sub_mask, 'DV'].to_numpy()
                        gv = gv_full[sub_mask.to_numpy()]
                        if gv.size and np.nanmax(gv) > 0:
                            gv = np.log1p(gv)
                            gmin, gmax = float(np.nanmin(gv)), float(np.nanmax(gv))
                            norm = ((gv - gmin) / (gmax - gmin + 1e-9)) ** gamma
                        else:
                            norm = np.zeros_like(gv, dtype=float)

                        fig2.add_trace(go.Scatter3d(
                            x=x_sub, y=y_sub, z=z_sub,
                            mode='markers',
                            marker=dict(size=marker_size, color=norm,
                                        colorscale='Viridis', cmin=0, cmax=1, opacity=0.9,
                                        colorbar=dict(title=f"{gene} expr", len=0.5)),
                            name=f"{gene} in {sub}"
                        ))
                elif not all_child_subs:
                    gv = gv_full[mask_cells.to_numpy()]
                    if gv.size and np.nanmax(gv) > 0:
                        gv = np.log1p(gv)
                        gmin, gmax = float(np.nanmin(gv)), float(np.nanmax(gv))
                        norm = ((gv - gmin) / (gmax - gmin + 1e-9)) ** gamma
                    else:
                        norm = np.zeros_like(gv, dtype=float)

                    fig2.add_trace(go.Scatter3d(
                        x=x_arr, y=y_arr, z=z_arr,
                        mode='markers',
                        marker=dict(size=marker_size, color=norm,
                                    colorscale='Viridis', cmin=0, cmax=1, opacity=0.9,
                                    colorbar=dict(title=f"{gene} expr", len=0.5)),
                        name=f"{gene} expression"
                    ))
            continue

        # 2–4 genes → Multi-channel coloring
        genes_multi = [g for g in selected_genes[:4] if g in mtx.var_names]
        if len(genes_multi) >= 1:
            gv_by_gene = {g: _gene_vec(mtx, g) for g in genes_multi}
            
            # Choose the number of colour channels based on the number of genes
            n_channels = min(len(genes_multi), 4)
            
            if child_subs:
                for sub in child_subs:
                    sub_mask = mask_cells & (mtx.obs['sub_leiden_str'] == sub)
                    sub_np = sub_mask.to_numpy()

                    x_sub = mtx.obs.loc[sub_mask, 'AP'].to_numpy()
                    y_sub = mtx.obs.loc[sub_mask, 'LR'].to_numpy()
                    z_sub = mtx.obs.loc[sub_mask, 'DV'].to_numpy()

                    N = x_sub.shape[0]
                    if N == 0: continue

                    # Use 4 channels or 3 channels (RGB mode)
                    if n_channels == 4:
                        # 4-gene mode: Red, Green, Blue, Cyan
                        channels = np.zeros((N, 4), dtype=float)
                        alpha = np.zeros(N, dtype=float)
                        for chan, gene in enumerate(genes_multi):
                            gv = gv_by_gene[gene][sub_np]
                            if gv.size and np.nanmax(gv) > 0:
                                gv = np.log1p(gv)
                                gmin, gmax = float(np.nanmin(gv)), float(np.nanmax(gv))
                                norm = ((gv - gmin) / (gmax - gmin + 1e-9)) ** gamma
                            else:
                                norm = np.zeros_like(gv)
                            channels[:, chan] = norm
                            alpha += norm
                        
                        # Convert to RGB: Red(1,0,0), Green(0,1,0), Blue(0,0,1), Cyan(0,1,1)
                        rgb = np.zeros((N, 3), dtype=float)
                        rgb[:, 0] = channels[:, 0]               # R = Red
                        rgb[:, 1] = channels[:, 1] + channels[:, 3]  # G = Green + Cyan
                        rgb[:, 2] = channels[:, 2] + channels[:, 3]  # B = Blue + Cyan
                        rgb = np.clip(rgb, 0, 1)
                        
                        alpha = np.clip(alpha / n_channels, 0.10, 1.0)
                    else:
                        # RGB mode for 1-3 genes
                        rgb = np.zeros((N, 3), dtype=float)
                        alpha = np.zeros(N, dtype=float)
                        for chan, gene in enumerate(genes_multi):
                            gv = gv_by_gene[gene][sub_np]
                            if gv.size and np.nanmax(gv) > 0:
                                gv = np.log1p(gv)
                                gmin, gmax = float(np.nanmin(gv)), float(np.nanmax(gv))
                                norm = ((gv - gmin) / (gmax - gmin + 1e-9)) ** gamma
                            else:
                                norm = np.zeros_like(gv)
                            rgb[:, chan] = norm
                            alpha += norm

                        alpha = np.clip(alpha / max(1, len(genes_multi)), 0.10, 1.0)
                    
                    rgba = [f'rgba({int(255*r)},{int(255*g)},{int(255*b)},{a:.3f})'
                            for (r, g, b), a in zip(rgb, alpha)]

                    fig2.add_trace(go.Scatter3d(
                        x=x_sub, y=y_sub, z=z_sub,
                        mode='markers',
                        marker=dict(size=marker_size, color=rgba),
                        name=f"Multi-gene in {sub}"
                    ))
            elif not all_child_subs:
                N = x_arr.shape[0]
                mask_np = mask_cells.to_numpy()
                
                if n_channels == 4:
                    # 4-gene mode: Red, Green, Blue, Cyan
                    channels = np.zeros((N, 4), dtype=float)
                    alpha = np.zeros(N, dtype=float)
                    for chan, gene in enumerate(genes_multi):
                        gv = gv_by_gene[gene][mask_np]
                        if gv.size and np.nanmax(gv) > 0:
                            gv = np.log1p(gv)
                            gmin, gmax = float(np.nanmin(gv)), float(np.nanmax(gv))
                            norm = ((gv - gmin) / (gmax - gmin + 1e-9)) ** gamma
                        else:
                            norm = np.zeros_like(gv)
                        channels[:, chan] = norm
                        alpha += norm
                    
                    rgb = np.zeros((N, 3), dtype=float)
                    rgb[:, 0] = channels[:, 0]               # R = Red
                    rgb[:, 1] = channels[:, 1] + channels[:, 3]  # G = Green + Cyan
                    rgb[:, 2] = channels[:, 2] + channels[:, 3]  # B = Blue + Cyan
                    rgb = np.clip(rgb, 0, 1)
                    
                    alpha = np.clip(alpha / n_channels, 0.10, 1.0)
                else:
                    # RGB mode
                    rgb = np.zeros((N, 3), dtype=float)
                    alpha = np.zeros(N, dtype=float)
                    for chan, gene in enumerate(genes_multi):
                        gv = gv_by_gene[gene][mask_np]
                        if gv.size and np.nanmax(gv) > 0:
                            gv = np.log1p(gv)
                            gmin, gmax = float(np.nanmin(gv)), float(np.nanmax(gv))
                            norm = ((gv - gmin) / (gmax - gmin + 1e-9)) ** gamma
                        else:
                            norm = np.zeros_like(gv)
                        rgb[:, chan] = norm
                        alpha += norm

                    alpha = np.clip(alpha / max(1, len(genes_multi)), 0.10, 1.0)

                rgba = [f'rgba({int(255*r)},{int(255*g)},{int(255*b)},{a:.3f})'
                        for (r, g, b), a in zip(rgb, alpha)]

                fig2.add_trace(go.Scatter3d(
                    x=x_arr, y=y_arr, z=z_arr,
                    mode='markers',
                    marker=dict(size=marker_size, color=rgba),
                    name="Multi-gene Expression"
                ))

    # Add section planes to plot 2
    for i, (toggle, color, width, opacity) in enumerate(zip(slice_toggles, slice_colors, slice_widths, slice_opacities)):
        if 'show' in toggle:
            add_slice_plane(fig2, SLICE_POSITIONS[i], color, width, opacity)

    fig2.update_layout(
        scene=dict(
            xaxis=dict(title='AP', showbackground=True),
            yaxis=dict(title='LR', showbackground=True),
            zaxis=dict(title='DV', showbackground=True),
            aspectmode='data'
        ),
        margin=dict(l=0,r=0,b=0,t=30),
        showlegend=False,
        uirevision='genes'
    )

    # Multi-gene legend
    if len(selected_genes) > 1:
        if len(selected_genes) == 4:
            # 4-gene mode: Red, Green, Blue, Cyan
            color_names = ['Red', 'Green', 'Blue', 'Cyan']
            color_codes = ['red', 'green', 'blue', 'cyan']
        else:
            # RGB mode
            color_names = ['Red', 'Green', 'Blue']
            color_codes = ['red', 'green', 'blue']
        
        items = []
        for i, gene in enumerate(selected_genes[:len(color_names)]):
            items.append(html.Span([
                html.Span(style={'display':'inline-block','width':'16px','height':'16px',
                                 'backgroundColor': color_codes[i],'marginRight':'8px',
                                 'border': '1px solid #999'}),
                f"{color_names[i]}: {gene}"
            ], style={'marginRight':'16px'}))
        mode_label = "RGBC" if len(selected_genes) == 4 else "RGB"
        rgb_legend = html.Div([html.Strong(f"{mode_label} mapping: "), *items])
    else:
        rgb_legend = ""

    return fig1, fig2, rgb_legend


if __name__ == '__main__':
    app.run(debug=False, port=8050)