import sys

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("../domain/")
sys.path.append("vis")
from domain.config import games


def viewInd(ind, taskName):
    env = games[taskName]
    if isinstance(ind, str):
        ind = np.loadtxt(ind, delimiter=",")
        wMat = ind[:, :-1]
        aVec = ind[:, -1]
    else:
        wMat = ind.wMat
        # Use aVec from ind if available, otherwise create zeros
        if hasattr(ind, 'aVec') and ind.aVec is not None and len(ind.aVec) > 0:
            aVec = ind.aVec
        else:
            aVec = np.zeros((np.shape(wMat)[0]))
    print("# of Connections in ANN: ", np.sum(wMat != 0))

    # Create Graph
    nIn = env.input_size + 1  # bias
    nOut = env.output_size
    G, layer = ind2graph(wMat, nIn, nOut)
    pos = getNodeCoord(G, layer, taskName)

    # Draw Graph
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    drawEdge(G, pos, wMat, layer)
    nx.draw_networkx_nodes(
        G, pos, node_color="lightblue", node_shape="o", cmap="terrain", vmin=0, vmax=6
    )
    drawNodeLabels(G, pos, aVec)
    labelInOut(pos, env)

    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelleft=False,
        labelbottom=False,
    )  # labels along the bottom edge are off

    return fig, ax


def ind2graph(wMat, nIn, nOut):
    hMat = wMat[nIn:-nOut, nIn:-nOut]
    hLay = getLayer(hMat) + 1

    if len(hLay) > 0:
        lastLayer = max(hLay) + 1
    else:
        lastLayer = 1
    L = np.r_[np.zeros(nIn), hLay, np.full((nOut), lastLayer)]

    layer = L
    order = layer.argsort()
    layer = layer[order]

    wMat = wMat[np.ix_(order, order)]
    nLayer = layer[-1]

    # Convert wMat to Full Network Graph
    rows, cols = np.where(wMat != 0)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G, layer


def getNodeCoord(G, layer, taskName):
    env = games[taskName]

    # Calculate positions of input and output
    nIn = env.input_size + 1
    nOut = env.output_size
    nNode = len(G.nodes)
    fixed_pos = np.empty((nNode, 2))
    fixed_nodes = np.r_[np.arange(0, nIn), np.arange(nNode - nOut, nNode)]

    # Set Figure dimensions
    fig_wide = 10
    fig_long = 5

    # Assign x and y coordinates per layer
    x = np.ones((1, nNode)) * layer  # Assign x coord by layer
    x = (x / np.max(x)) * fig_wide  # Normalize

    _, nPerLayer = np.unique(layer, return_counts=True)

    y = cLinspace(-2, fig_long + 2, nPerLayer[0])
    for i in range(1, len(nPerLayer)):
        if i % 2 == 0:
            y = np.r_[y, cLinspace(0, fig_long, nPerLayer[i])]
        else:
            y = np.r_[y, cLinspace(-1, fig_long + 1, nPerLayer[i])]

    fixed_pos = np.c_[x.T, y.T]
    pos = dict(enumerate(fixed_pos.tolist()))

    return pos


def labelInOut(pos, env):
    nIn = env.input_size + 1
    nOut = env.output_size
    nNode = len(pos)
    fixed_nodes = np.r_[np.arange(0, nIn), np.arange(nNode - nOut, nNode)]

    if len(env.in_out_labels) > 0:
        stateLabels = ["bias"] + env.in_out_labels
        labelDict = {}
    for i in range(len(stateLabels)):
        labelDict[fixed_nodes[i]] = stateLabels[i]

    for i in range(nIn):
        plt.annotate(
            labelDict[i],
            xy=(pos[i][0] - 0.5, pos[i][1]),
            xytext=(pos[i][0] - 2.5, pos[i][1] - 0.5),
            arrowprops=dict(arrowstyle="->", color="k", connectionstyle="angle"),
        )

    for i in range(nNode - nOut, nNode):
        plt.annotate(
            labelDict[i],
            xy=(pos[i][0] + 0.1, pos[i][1]),
            xytext=(pos[i][0] + 1.5, pos[i][1] + 1.0),
            arrowprops=dict(arrowstyle="<-", color="k", connectionstyle="angle"),
        )


def drawNodeLabels(G, pos, aVec):
    # Complete activation function labels with more readable names
    # Mapping: actId -> label
    actLabel = np.array(
        (
            [
                "",           # 0 - unused
                "Linear",     # 1 - Linear
                "Step",       # 2 - Unsigned Step Function
                "Sin",        # 3 - Sin
                "Gauss",      # 4 - Gaussian
                "Tanh",       # 5 - Hyperbolic Tangent
                "Sigmoid",    # 6 - Sigmoid
                "Inverse",    # 7 - Inverse
                "Abs",        # 8 - Absolute Value
                "ReLU",       # 9 - Relu
                "Cos",        # 10 - Cosine
                "Squared",    # 11 - Squared
            ]
        )
    )
    # Color mapping for activation functions
    actColors = {
        0: 'gray',      # unused
        1: 'black',     # Linear - black
        2: 'red',       # Step - red
        3: 'blue',      # Sin - blue
        4: 'purple',    # Gauss - purple
        5: 'green',     # Tanh - green
        6: 'orange',    # Sigmoid - orange
        7: 'brown',    # Inverse - brown
        8: 'cyan',      # Abs - cyan
        9: 'magenta',   # ReLU - magenta
        10: 'navy',     # Cos - navy blue
        11: 'olive',    # Squared - olive
    }
    
    # Handle out-of-bounds indices (shouldn't happen, but be safe)
    aVec_int = aVec.astype(int)
    aVec_int = np.clip(aVec_int, 0, len(actLabel) - 1)
    listLabel = actLabel[aVec_int]
    label = dict(enumerate(listLabel))
    
    # Create color mapping for each node
    label_colors = {}
    for node_id, act_id in enumerate(aVec_int):
        label_colors[node_id] = actColors.get(act_id, 'black')
    
    # Draw labels with color coding - need to draw individually to set different colors
    for node_id, label_text in label.items():
        if label_text:  # Only draw if label is not empty
            nx.draw_networkx_labels(
                G,
                pos,
                labels={node_id: label_text},
                font_size=8,
                font_weight='bold',
                font_color=label_colors[node_id]
            )


def drawEdge(G, pos, wMat, layer):
    wMat[np.isnan(wMat)] = 0
    # Organize edges by layer
    _, nPerLayer = np.unique(layer, return_counts=True)
    edgeLayer = []
    edgeWeights = []  # Store weights for each layer (signed, not absolute)
    edgeSigns = []  # Store sign info for each layer (True = positive, False = negative)
    layBord = np.cumsum(nPerLayer)
    for i in range(0, len(layBord)):
        tmpMat = np.copy(wMat)
        start = layBord[-i]
        end = layBord[-i + 1]
        tmpMat[:, :start] *= 0
        tmpMat[:, end:] *= 0
        rows, cols = np.where(tmpMat != 0)
        edges = list(zip(rows.tolist(), cols.tolist()))  # Convert to list to avoid zip exhaustion
        edgeLayer.append(nx.DiGraph())
        edgeLayer[-1].add_edges_from(edges)
        # Store weights and signs for edges in this layer
        weights = [tmpMat[r, c] for r, c in edges]
        signs = [w >= 0 for w in weights]  # True for positive, False for negative
        edgeWeights.append([abs(w) for w in weights])  # Store absolute values for width calculation
        edgeSigns.append(signs)
    edgeLayer.append(edgeLayer.pop(0))  # move first layer to correct position
    edgeWeights.append(edgeWeights.pop(0))  # move weights to match
    edgeSigns.append(edgeSigns.pop(0))  # move signs to match

    # Calculate weight scaling for line thickness
    # Get all non-zero weights for normalization
    all_weights = [w for weights in edgeWeights for w in weights]
    if len(all_weights) > 0:
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        # Scale weights to line width range: 0.5 (thin) to 5.0 (thick)
        # Use absolute value since we want magnitude
        if max_weight > min_weight:
            weight_range = max_weight - min_weight
        else:
            weight_range = 1.0  # Avoid division by zero
    else:
        min_weight = 0
        weight_range = 1.0

    # Layer Colors
    for i in range(len(edgeLayer)):
        C = [i / len(edgeLayer)] * len(edgeLayer[i].edges)
        # Calculate line widths based on weight magnitudes
        if len(edgeWeights[i]) > 0:
            # Normalize weights to [0.5, 5.0] range
            widths = []
            for w in edgeWeights[i]:
                if weight_range > 0:
                    normalized = (w - min_weight) / weight_range
                    width = 0.5 + normalized * 4.5  # Scale to [0.5, 5.0]
                else:
                    width = 1.0  # Default if all weights are same
                widths.append(width)
        else:
            widths = [1.0] * len(edgeLayer[i].edges)
        
        # Determine line styles: solid for positive, dashed for negative
        styles = ['solid' if sign else 'dashed' for sign in edgeSigns[i]]
        
        # Draw edges with different styles for positive/negative weights
        # NetworkX doesn't support per-edge styles directly, so we need to draw in groups
        positive_edges = [edge for j, edge in enumerate(edgeLayer[i].edges) if edgeSigns[i][j]]
        negative_edges = [edge for j, edge in enumerate(edgeLayer[i].edges) if not edgeSigns[i][j]]
        positive_widths = [widths[j] for j in range(len(widths)) if edgeSigns[i][j]]
        negative_widths = [widths[j] for j in range(len(widths)) if not edgeSigns[i][j]]
        positive_colors = [C[j] for j in range(len(C)) if edgeSigns[i][j]]
        negative_colors = [C[j] for j in range(len(C)) if not edgeSigns[i][j]]
        
        # Draw positive weights (solid lines)
        if len(positive_edges) > 0:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=positive_edges,
                alpha=0.75,
                width=positive_widths,
                edge_color=positive_colors,
                edge_cmap=plt.cm.viridis,
                edge_vmin=0.0,
                edge_vmax=1.0,
                arrowsize=8,
                style='solid',
            )
        
        # Draw negative weights (dashed lines)
        if len(negative_edges) > 0:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=negative_edges,
                alpha=0.75,
                width=negative_widths,
                edge_color=negative_colors,
                edge_cmap=plt.cm.viridis,
                edge_vmin=0.0,
                edge_vmax=1.0,
                arrowsize=8,
                style='dashed',
            )


def getLayer(wMat):
    """
    Traverse wMat by row, collecting layer of all nodes that connect to you (X).
    Your layer is max(X)+1
    """
    wMat[np.isnan(wMat)] = 0
    wMat[wMat != 0] = 1
    nNode = np.shape(wMat)[0]
    layer = np.zeros((nNode))
    while True:  # Loop until sorting doesn't help any more
        prevOrder = np.copy(layer)
        for curr in range(nNode):
            srcLayer = np.zeros((nNode))
            for src in range(nNode):
                srcLayer[src] = layer[src] * wMat[src, curr]
            layer[curr] = np.max(srcLayer) + 1
        if all(prevOrder == layer):
            break
    return layer - 1


def cLinspace(start, end, N):
    if N == 1:
        return np.mean([start, end])
    else:
        return np.linspace(start, end, N)


def lload(fileName):
    return np.loadtxt(fileName, delimiter=",")
