import dash
from dash import html, dcc, Output, Input, State
from dash.dependencies import ALL
import plotly.graph_objects as go
import plotly.express as px
import scanpy as sc
import numpy as np
import json, os, re

# ---------- load data ----------
mtx = sc.read_h5ad("final_mtx_v3_Fu_adjustedz.h5ad")
mtx.obs['whole_leiden_str'] = mtx.obs['whole_leiden'].astype(str)
mtx.obs['sub_leiden_str'] = mtx.obs['sub_leiden'].astype(str)

for col in ['x_centroid', 'y_centroid', 'z_centroid']:
    if col not in mtx.obs.columns:
        mtx.obs[col] = 0  # Dummy fallback

cluster_labels = sorted(mtx.obs['whole_leiden'].astype(str).unique())
e_clusters = sorted([cl for cl in cluster_labels if cl.startswith("e_")], key=lambda s: (s.split("_")[0], int(s.split("_")[1])))
p_clusters = sorted([cl for cl in cluster_labels if cl.startswith("p_")], key=lambda s: (s.split("_")[0], int(s.split("_")[1])))

section_labels = sorted(mtx.obs['adjusted_cure_clustering'].unique())

cluster_colors_list = mtx.uns.get('whole_leiden_colors', px.colors.qualitative.Plotly * (len(cluster_labels) // 10 + 1))
cluster_colors = {str(c): cluster_colors_list[i] for i, c in enumerate(cluster_labels)}

gene_list = mtx.var_names.tolist() if hasattr(mtx, 'var_names') else []

parent_to_sub = {}
for parent in e_clusters:
    subs = (
        mtx.obs.loc[mtx.obs['whole_leiden_str'] == parent, 'sub_leiden']
        .replace("", np.nan).dropna().unique()
    )
    parent_to_sub[parent] = sorted(map(str, subs)) if len(subs) else []

parent_map = {sub: parent for parent, subs in parent_to_sub.items() for sub in subs}
subcluster_colors = {sub: cluster_colors.get(parent_map[sub], '#808080') for sub in parent_map}
# ---------- include scaffolds ----------
mesh_folder = "/Users/jamilla/Documents/LUMC/hEmbryo/xenium/DASH_APP/scaffolds"
mesh_data = {}

def _cluster_key_from_fname(fname: str):
    m = re.match(r"(.+)_mesh\.json$", fname)  # "e_0_mesh.json"
    if m: return m.group(1)
    m = re.match(r"mesh_(.+)\.json$", fname)  # "mesh_e_0.json"
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

# ---------- UI helpers ----------
def mesh_checkbox(id_dict):
    return dcc.Checklist(
        id=id_dict,
        options=[{'label': '', 'value': 'mesh'}],
        value=[],
        className="mesh-only"
    )

def control_rows(clusters, parent_to_sub, cluster_colors, subcluster_colors, mesh_data):
    """Build tidy 3-col grid rows: [label+check] [color] [mesh] for parents + subclusters."""
    blocks = []
    for parent in clusters:
        # parent row
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

        # subcluster rows
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
    html.H1("3D Cluster Viewer", style={'textAlign': 'center'}),

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
            dcc.Slider(id='mesh_opacity', min=0.05, max=0.9, step=0.05, value=0.25,
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Label("Z Zoom:", style={'fontSize': '12px'}),
            dcc.Slider(id='z_zoom_slider', min=500, max=5000, step=250, value=2500,
                       tooltip={"placement": "bottom", "always_visible": False})
        ], style={'flex': '1', 'minWidth': '220px'}),

        html.Div([
            html.Label("Select Section to Display:"),
            dcc.Checklist(
                id='section_selector',
                options=[{'label': str(sec), 'value': sec} for sec in section_labels],
                value=section_labels, inline=True,
                style={'margin-bottom': '10px'}
            ),
            html.Label("Select up to 3 Genes (for RGB coloring):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='gene_selector',
                options=[{'label': g, 'value': g} for g in gene_list],
                multi=True, value=[],
                placeholder="Pick genes to color cells",
                style={'width': '100%'}
            )
        ], style={'flex': '1', 'minWidth': '280px', 'padding':'0 20px'})
    ], style={'display': 'flex', 'gap':'20px', 'margin-bottom': '16px', 'alignItems': 'flex-start'}),

    # --------- Main content: left = 2 graphs, right = 2 columns of controls ----------
    html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='cluster_3d_plot', style={'height': '600px', 'width': '100%'})
            ], style={'flex': '1', 'minWidth':'0', 'marginRight':'8px'}),
            html.Div([
                dcc.Graph(id='gene_expression_plot', style={'height': '600px', 'width': '100%'}),
                html.Div(id='rgb_legend', style={'textAlign': 'center', 'paddingTop': '6px'})
            ], style={'flex': '1', 'minWidth':'0', 'marginLeft':'8px'}),
        ], style={'flex': '3', 'display': 'flex', 'flexDirection': 'row', 'minWidth':'0'}),
        html.Div([
            # Embryo column
            html.Div([
                html.Div([
                    html.Div("Embryo",   style={'fontWeight':'bold'}),
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

            # Placenta column
            html.Div([
                html.Div([
                    html.Div("Placenta", style={'fontWeight':'bold'}),
                    html.Div("Surface",  style={'textAlign':'center','fontWeight':'bold'}),
                ], style={
                    'display':'grid',
                    'gridTemplateColumns':'auto 34px 18px',
                    'columnGap':'2px',
                    'marginBottom':'4px'
                }),

                dcc.Checklist(
                    id='toggle_all_p',
                    options=[{'label':'Toggle All','value':'all'}],
                    value=[],
                    style={'marginBottom':'6px'}
                ),

                html.Div(control_rows(p_clusters, {}, cluster_colors, {}, mesh_data))
            ], style={'flex':'1', 'minWidth':'180px', 'paddingLeft':'8px'})

        ], style={'flex':'1', 'display':'flex', 'flexDirection':'row',
                  'maxHeight':'600px', 'overflowY':'auto', 'paddingLeft':'4px'})



    ], style={'display': 'flex', 'gap':'4px', 'alignItems': 'flex-start'}),

    dcc.Store(id='cluster_selector', data=cluster_labels),
    dcc.Store(id='cluster_colors_store', data=cluster_colors),
    dcc.Store(id='subcluster_colors_store', data=subcluster_colors)
])

## --------- Debugging --------
@app.callback(
    Output('cluster_selector', 'data', allow_duplicate=True),
    Input('cluster_selector', 'data'),
    prevent_initial_call=True
)
def debug_cluster_selector(data):
    print("cluster_selector store value:", data)
    return dash.no_update

@app.callback(
    Output('cluster_colors_store', 'data', allow_duplicate=True),
    Input('cluster_colors_store', 'data'),
    prevent_initial_call=True
)
def debug_cluster_colors(data):
    print("cluster_colors_store value:", data)
    return dash.no_update
@app.callback(
    Output('cluster_selector', 'data'),
    Input({'type': 'cluster-check', 'index': ALL}, 'value'),
)
def sync_selected_clusters(values):
    selected_clusters = [val for v in (values or []) for val in (v or [])]
    return selected_clusters

@app.callback(
    Output('gene_expression_plot', 'figure', allow_duplicate=True),
    Input('gene_expression_plot', 'figure'),
    prevent_initial_call=True
)
def debug_gene_expression(fig):
    print("gene_expression_plot figure updated")
    return dash.no_update
# -----------------------------------------------


# --- Toggle all clusters ---
@app.callback(
    Output({'type': 'cluster-check', 'index': ALL}, 'value'),
    Input('toggle_all_e', 'value'),
    Input('toggle_all_p', 'value'),
    State({'type': 'cluster-check', 'index': ALL}, 'id')
)
def toggle_all(e_toggle, p_toggle, ids):
    e_on = 'all' in (e_toggle or [])
    p_on = 'all' in (p_toggle or [])
    out = []
    for cid in ids:
        idx = cid['index']
        if idx in e_clusters:
            out.append([idx] if e_on else [])
        elif idx in p_clusters:
            out.append([idx] if p_on else [])
        else:
            out.append([])
    return out

@app.callback(
    Output({'type': 'subcluster-check', 'index': ALL}, 'value'),
    Input({'type': 'cluster-check', 'index': ALL}, 'value'),
    State({'type': 'cluster-check', 'index': ALL}, 'id'),
    State({'type': 'subcluster-check', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def parent_toggles_children(parent_values, parent_ids, sub_ids):
    """
    If a parent cluster is checked, select ALL its subclusters.
    If a parent cluster is unchecked, deselect ALL its subclusters.
    """
    # Which parents are currently checked?
    checked_parents = {pid['index']
                       for val, pid in zip(parent_values or [], parent_ids or [])
                       if (val and len(val) > 0)}

    out = []
    for sid in sub_ids or []:
        sub = sid['index']
        parent = parent_map.get(sub)  # e.g., "e_4"
        if parent in checked_parents:
            out.append([sub])          # selected
        else:
            out.append([])             # deselected
    return out

# ----------------------------------------

# --- Update cluster colors ---
@app.callback(
    Output('cluster_colors_store', 'data'),
    Input({'type': 'cluster-color-input', 'index': ALL}, 'value'),
    State({'type': 'cluster-color-input', 'index': ALL}, 'id'),
    State('cluster_colors_store', 'data')
)
def update_cluster_colors(colors, ids, current_colors):
    base = dict(current_colors or {})
    if colors is None or ids is None:
        return base
    for color, cid in zip(colors, ids):
        if color and cid and 'index' in cid:
            base[cid['index']] = color
    return base

@app.callback(
    Output('subcluster_colors_store', 'data'),
    Input({'type': 'subcluster-color-input', 'index': ALL}, 'value'),
    State({'type': 'subcluster-color-input', 'index': ALL}, 'id'),
    State('subcluster_colors_store', 'data'),
    prevent_initial_call=True
)
def update_subcluster_colors(colors, ids, current):
    base = dict(current or {})
    for color, cid in zip(colors or [], ids or []):
        if color and 'index' in cid:
            base[cid['index']] = color
    return base

try:
    from dash import ctx
except Exception:
    ctx = dash.callback_context

@app.callback(
    Output({'type': 'subcluster-color-input', 'index': ALL}, 'value'),
    Input({'type': 'cluster-color-input', 'index': ALL}, 'value'),
    State({'type': 'cluster-color-input', 'index': ALL}, 'id'),
    State({'type': 'subcluster-color-input', 'index': ALL}, 'id'),
)
def apply_parent_color_to_subs(parent_colors, parent_ids, sub_ids):
    if not parent_colors or not parent_ids:
        return [dash.no_update] * len(sub_ids or [])

    triggered = getattr(ctx, "triggered_id", None)
    if not triggered:
        t = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        triggered = json.loads(t) if t else None

    if not triggered or triggered.get('type') != 'cluster-color-input':
        return [dash.no_update] * len(sub_ids or [])

    changed_parent = triggered['index']
    parent_color_map = {pid['index']: col for col, pid in zip(parent_colors, parent_ids)}
    new_color = parent_color_map.get(changed_parent)

    if not new_color:
        return [dash.no_update] * len(sub_ids or [])

    out = []
    for sid in sub_ids or []:
        sub = sid['index']
        if parent_map.get(sub) == changed_parent:
            out.append(new_color)
        else:
            out.append(dash.no_update)
    return out

# ----------------------------------------

def _gene_vec(adata, gene):
    """Return 1-D numpy array of expression for 'gene' across all cells."""
    Xg = adata[:, gene].X
    if hasattr(Xg, "toarray"):
        arr = Xg.toarray()
    else:
        arr = np.asarray(Xg)
    return np.ravel(arr)



@app.callback(
    Output('cluster_3d_plot', 'figure'),
    Output('gene_expression_plot', 'figure'),
    Output('rgb_legend', 'children'),
    Input('cluster_selector', 'data'),
    Input('section_selector', 'value'),
    Input('gene_selector', 'value'),
    Input('gamma_slider', 'value'),
    Input('size_slider', 'value'),
    Input('mesh_opacity', 'value'),
    Input('z_zoom_slider', 'value'),
    Input({'type': 'mesh-toggle', 'index': ALL}, 'value'),
    Input({'type': 'subcluster-check', 'index': ALL}, 'value'),
    Input({'type': 'cluster-color-input', 'index': ALL}, 'value'),
    Input({'type': 'subcluster-color-input', 'index': ALL}, 'value'),

    State({'type': 'mesh-toggle', 'index': ALL}, 'id'),
    State({'type': 'subcluster-check', 'index': ALL}, 'id'),
    State({'type': 'cluster-color-input', 'index': ALL}, 'id'),
    State({'type': 'subcluster-color-input', 'index': ALL}, 'id'),
)
def update_figures(
    selected_clusters, selected_sections, selected_genes,
    gamma, marker_size, mesh_opacity, z_zoom,
    mesh_values, sub_values, parent_colors, sub_colors,
    mesh_ids, sub_ids, parent_color_ids, sub_color_ids
):
    selected_clusters = selected_clusters or []

    mesh_toggle_dict = {
        cid['index']: val
        for cid, val in zip(mesh_ids or [], mesh_values or [])
    }

    live_cluster_colors = dict(cluster_colors)
    if parent_colors and parent_color_ids:
        for col, cid in zip(parent_colors, parent_color_ids):
            if col and cid and 'index' in cid:
                live_cluster_colors[cid['index']] = col

    live_subcluster_colors = dict(subcluster_colors)
    if sub_colors and sub_color_ids:
        for col, cid in zip(sub_colors, sub_color_ids):
            if col and cid and 'index' in cid:
                live_subcluster_colors[cid['index']] = col

    # Which subclusters are currently checked?
    selected_subs = {
        sid['index']
        for val, sid in zip(sub_values or [], sub_ids or [])
        if (val and len(val) > 0)
    }

    if selected_genes is None:
        selected_genes = []
    elif isinstance(selected_genes, str):
        selected_genes = [selected_genes]
    else:
        # It may be a tuple/np array; coerce and drop falsy entries
        selected_genes = [g for g in list(selected_genes) if g]


    # ---- Figure 1: clusters ----
    fig1 = go.Figure()
    for parent in selected_clusters:
        parent_mask = (
            (mtx.obs['whole_leiden_str'] == parent) &
            (mtx.obs['adjusted_cure_clustering'].isin(selected_sections))
        )
        # subs of this parent that are selected
        child_subs = [s for s in parent_to_sub.get(parent, []) if s in selected_subs]

        for sub in child_subs:
            mask = parent_mask & (mtx.obs['sub_leiden_str'] == sub)
            fig1.add_trace(go.Scatter3d(
                x=mtx.obs.loc[mask, 'x_kde_aligned'],
                y=mtx.obs.loc[mask, 'z_centroid'],
                z=mtx.obs.loc[mask, 'y_kde_aligned'],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=live_subcluster_colors.get(sub, live_cluster_colors.get(parent, '#808080'))
                ),
                name=f'{sub} cells',
                legendgroup=parent
            ))

            # optional sub mesh
            if ('mesh' in mesh_toggle_dict.get(sub, [])) and (sub in mesh_data):
                mesh = mesh_data[sub]
                verts = np.array(mesh['verts']); faces = np.array(mesh['faces'], dtype=int)
                if faces.size > 0 and verts.size > 0:
                    i, j, k = faces.T
                    fig1.add_trace(go.Mesh3d(
                        x=verts[:,0], y=verts[:,1], z=verts[:,2],
                        i=i, j=j, k=k,
                        color=live_subcluster_colors.get(sub, live_cluster_colors.get(parent, '#808080')),
                        opacity=mesh_opacity,
                        flatshading=False,
                        name=f"{sub} mesh",
                        legendgroup=parent,
                        showlegend=True
                    ))

        # optional parent mesh (kept if you want)
        if ('mesh' in mesh_toggle_dict.get(parent, [])) and (parent in mesh_data):
            mesh = mesh_data[parent]
            verts = np.array(mesh['verts']); faces = np.array(mesh['faces'], dtype=int)
            if faces.size > 0 and verts.size > 0:
                i, j, k = faces.T
                fig1.add_trace(go.Mesh3d(
                    x=verts[:,0], y=verts[:,1], z=verts[:,2],
                    i=i, j=j, k=k,
                    color=live_cluster_colors.get(parent, '#808080'),
                    opacity=mesh_opacity,
                    flatshading=False,
                    name=f"{parent} mesh",
                    legendgroup=parent,
                    showlegend=True
                ))


    fig1.update_layout(
        scene=dict(
            xaxis=dict(title='', showbackground=False, showticklabels=False),
            yaxis=dict(title='', showbackground=False, showticklabels=False, range=[-z_zoom, z_zoom]),
            zaxis=dict(title='', showbackground=False, showticklabels=False),
            aspectmode='data'
        ),
        margin=dict(l=0,r=0,b=0,t=30),
        showlegend=False,
        uirevision='clusters'
    )

    # ---- Figure 2: gene expression ----
    fig2 = go.Figure()
    rgb_legend = []

    for cluster in selected_clusters:
        mask_cells = (
            (mtx.obs['whole_leiden_str'] == cluster) &
            (mtx.obs['adjusted_cure_clustering'].isin(selected_sections))
        )
        mask_np = mask_cells.to_numpy()

        if ('mesh' in mesh_toggle_dict.get(cluster, [])) and (cluster in mesh_data):
            mesh = mesh_data[cluster]
            verts = np.array(mesh['verts'])
            faces = np.array(mesh['faces'], dtype=int)
            if faces.size > 0 and verts.size > 0:
                i, j, k = faces.T
                fig2.add_trace(go.Mesh3d(
                    x=verts[:,0], y=verts[:,1], z=verts[:,2],
                    i=i, j=j, k=k,
                    color='lightgrey',
                    opacity=0.12,
                    flatshading=False,
                    showlegend=False
                ))

        if not mask_cells.any():
            continue

        x = mtx.obs.loc[mask_cells, 'x_kde_aligned'].to_numpy()
        y = mtx.obs.loc[mask_cells, 'z_centroid'   ].to_numpy()
        z = mtx.obs.loc[mask_cells, 'y_kde_aligned'].to_numpy()

        if len(selected_genes) == 0:
            child_subs = [s for s in parent_to_sub.get(cluster, []) if s in selected_subs]
            for sub in child_subs:
                sub_mask = mask_cells & (mtx.obs['sub_leiden_str'] == sub)
                fig2.add_trace(go.Scatter3d(
                    x=mtx.obs.loc[sub_mask, 'x_kde_aligned'],
                    y=mtx.obs.loc[sub_mask, 'z_centroid'],
                    z=mtx.obs.loc[sub_mask, 'y_kde_aligned'],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=live_subcluster_colors.get(sub, live_cluster_colors.get(cluster, '#808080')),
                        opacity=0.85
                    ),
                    name=f'{sub} cells'
                ))
            continue 

        elif len(selected_genes) == 1:
            gene = selected_genes[0]
            if gene in mtx.var_names:
                gv_full = _gene_vec(mtx, gene)
                child_subs = [s for s in parent_to_sub.get(cluster, []) if s in selected_subs]
                for sub in child_subs:
                    sub_mask = mask_cells & (mtx.obs['sub_leiden_str'] == sub)
                    sub_np = sub_mask.to_numpy()
                    gv = gv_full[sub_np]

                    if gv.size and np.nanmax(gv) > 0:
                        gv = np.log1p(gv)
                        gmin, gmax = float(np.nanmin(gv)), float(np.nanmax(gv))
                        norm = ((gv - gmin) / (gmax - gmin + 1e-9)) ** gamma
                    else:
                        norm = np.zeros_like(gv, dtype=float)

                    fig2.add_trace(go.Scatter3d(
                        x=mtx.obs.loc[sub_mask, 'x_kde_aligned'],
                        y=mtx.obs.loc[sub_mask, 'z_centroid'],
                        z=mtx.obs.loc[sub_mask, 'y_kde_aligned'],
                        mode='markers',
                        marker=dict(
                            size=marker_size,
                            color=norm,
                            colorscale='Viridis',
                            cmin=0, cmax=1,
                            opacity=0.9,
                            colorbar=dict(title=f"{gene} expr", len=0.5)
                        ),
                        name=f"{gene} in {sub}"
                    ))


        else:
            genes_rgb = [g for g in selected_genes[:3] if g in mtx.var_names]
            if len(genes_rgb) >= 1:
                N = x.shape[0]
                rgb = np.zeros((N, 3), dtype=float)
                alpha = np.zeros(N, dtype=float)

                for chan, gene in enumerate(genes_rgb):
                    gv = mtx[:, gene].X
                    gv = gv.toarray().ravel() if hasattr(gv, 'toarray') else np.ravel(gv)
                    gv = gv[mask_np]
                    if gv.size and np.nanmax(gv) > 0:
                        gv = np.log1p(gv)
                        gmin, gmax = float(np.nanmin(gv)), float(np.nanmax(gv))
                        norm = ((gv - gmin) / (gmax - gmin + 1e-9)) ** gamma
                    else:
                        norm = np.zeros_like(gv)
                    rgb[:, chan] = norm
                    alpha += norm

                alpha = np.clip(alpha / max(1, len(genes_rgb)), 0.10, 1.0)
                rgba = [f'rgba({int(255*r)},{int(255*g)},{int(255*b)},{a:.3f})'
                        for (r,g,b), a in zip(rgb, alpha)]

                fig2.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(size=marker_size, color=rgba),
                    name="RGB Expression"
                ))

    fig2.update_layout(
        scene=dict(
            xaxis=dict(title='', showbackground=False, showticklabels=False),
            yaxis=dict(title='', showbackground=False, showticklabels=False, range=[-z_zoom, z_zoom]),
            zaxis=dict(title='', showbackground=False, showticklabels=False),
            aspectmode='data'
        ),
        margin=dict(l=0,r=0,b=0,t=30),
        showlegend=False,
        uirevision='genes'
    )

    # RGB legend
    if len(selected_genes) > 1:
        color_names = ['Red','Green','Blue']
        items = []
        for i, gene in enumerate(selected_genes[:3]):
            items.append(html.Span([
                html.Span(style={'display':'inline-block','width':'16px','height':'16px',
                                 'backgroundColor': color_names[i].lower(),'marginRight':'8px'}),
                f"{color_names[i]}: {gene}"
            ], style={'marginRight':'16px'}))
        rgb_legend = html.Div([html.Strong("RGB mapping: "), *items])
    else:
        rgb_legend = ""

    return fig1, fig2, rgb_legend


if __name__ == '__main__':
    app.run(debug=False)
